from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.stat_scores import StatScores
from transformers.models.auto.modeling_auto import \
    AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup

from src.loss import SoftTargetCrossEntropy
from src.mixup import Mixup
from src.coordinatedmemory.coordinateMemory import RecurrentCoordinateMemLayer, CoordinateMemLayer

from src.HMA.model import HMA

from timm import create_model



class TimmClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'vit_base_patch32_224.augreg_in21k',
        use_pretrained = True,
        ckpt = '/baai-memory-llm/qiuzihan/VIT/pretrained_ckpt/vit_base_patch32_224.augreg_in21k.pth', #None,
        optimizer: str = "sgd",
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 50,
        n_classes: int = 10,
        channels_last: bool = False,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        tune_head: bool = False,
        tune_cls: bool = False,
        image_size: int = 224,
        weights: Optional[str] = None,
        global_pool='token', # can use 'avg'
        # memory settings
        mem_config=None,
        mem_type=None,
        add_mem_num=5,
        **kwargs,
        # mixup: bool = False,
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup epochs
            n_classes: Number of target class.
            channels_last: Change to channels last memory format for possible training speed up
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            linear_probe: Only train the classifier and keep other layers frozen
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes
        self.channels_last = channels_last
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        
        self.tune_head = tune_head
        self.tune_cls = tune_cls
        
        self.image_size = image_size
        self.weights = weights
        # self.mixup = mixup
        
        self.mem_type = mem_type

        self.mem_config = mem_config

        # Initialize network
        self.net = create_model(model_name=model_name, 
                                pretrained=use_pretrained and (ckpt is None),
                                # pretrained=True,
                                # checkpoint_path=ckpt,
                                # num_classes=n_classes,
                                global_pool=global_pool,
                                mem_type=mem_type,
                                add_mem_num=add_mem_num,
                                add_coordinate_mem=False if mem_config is None else (mem_config.CoM and mem_config.inner_mem),
                                coordinate_configs=mem_config,
                                )
        try:
            hidden_size = self.net.head.in_features
        except AttributeError:
            fc_ind = True
            hidden_size = self.net.fc.in_features
        # if ckpt is None and not use_pretrained:
        #     torch.save(self.net.state_dict(), f'{model_name}.pth')
        
        # Load checkpoint weights
        if ckpt is not None:
            print(f"loading from {ckpt}")
            self.net.load_state_dict(torch.load(ckpt), strict=False)
            
        # self.net.head = torch.nn.Linear(self.net.embed_dim, n_classes)
        

        
        fc_ind = False
        
        if mem_config is not None and mem_config.CoM:
            self.visit_penalty_weight = mem_config.visit_penalty_weight
            self.mem_number = mem_config.mem_number
            self.aug_type = mem_config.aug_type
            self.contrastive_type = mem_config.contrastive_type
            self.contrastive_weight = mem_config.contrastive_weight
            print("using CoM")
            # if mem_config.inner
            if mem_config.outer_mem:
                print("using outer mem")
                self.net.head = RecurrentCoordinateMemLayer(hidden_size=hidden_size, output_shape=n_classes,
                                                neighbor_number=mem_config.neighbor_number, 
                                                memory_size=mem_config.mem_number,
                                                aug_type=mem_config.aug_type,
                                                temp=mem_config.mem_temp, restrict_method=mem_config.restrict_method, 
                                                spread_method=mem_config.spread_method,
                                                contrastive_type=mem_config.contrastive_type,
                                                contrastive_temp=mem_config.contrastive_temp,
                                                sum_type=mem_config.sum_type, 
                                                max_recurrent_step=mem_config.max_recurrent_step,
                                                weighted_penalty=mem_config.weighted_penalty,
                                                key_value_mem=mem_config.key_value_mem,)
            else:
                try:
                    self.net.head = torch.nn.Linear(self.net.head.in_features, n_classes)
                except AttributeError:
                    fc_ind = True
                    self.net.fc = torch.nn.Linear(self.net.fc.in_features, n_classes)
        elif mem_config is not None and mem_config.HMA:
            print("using HMA")
            attention_config = {"model_num_heads":mem_config.num_heads, 
                                'viz_att_maps': False, 
                                'model_hidden_dropout_prob': 0.1, 
                                'model_layer_norm_eps': 1e-12,  
                                'null_KV':mem_config.null_KV,
                                'topk':mem_config.topk,
                                'norm_after_topk':mem_config.norm_after_topk,
                                'softmax_temp':mem_config.softmax_temp}
            memory_config = {"num_global_memory":mem_config.global_memory_number,
                            "local_mem_size":mem_config.local_mem_size,
                            "gaussain_mem":mem_config.gaussain_mem,}
            model_config = {'label_hidden':mem_config.label_hidden,
                            "dim_in":hidden_size, 
                            "dim_out":hidden_size,
                            "sep_label":False,
                            "encoder_momentum":mem_config.encoder_momentum*(1-self.tune_head),
                            'abd_augmentation':mem_config.abd_augmentation,
                            "logits_resource":mem_config.logits_resource,
                            'aba_ablation':mem_config.aba_ablation,
                            'abd_ablation':mem_config.abd_ablation,}
            self.net = HMA(encoder_model=self.net, config=[attention_config, memory_config, model_config], num_classes=n_classes)
        else:
            try:
                self.net.head = torch.nn.Linear(self.net.head.in_features, n_classes)
            except AttributeError:
                fc_ind = True
                self.net.fc = torch.nn.Linear(self.net.fc.in_features, n_classes)




        # Freeze transformer layers if linear probing
        if self.tune_head:
            if mem_config is not None and mem_config.HMA or (mem_config.CoM and mem_config.outer_mem):
                print("head augmentation")
                for name, param in self.net.encoder.named_parameters():
                    if "head" not in name:
                        if mem_config is not None and mem_config.CoM and mem_config.inner_mem and 'mem' in name:
                            pass
                        else:
                            param.requires_grad = False
                    if fc_ind:
                        if "fc." in name:
                            param.requires_grad = True
                    if self.tune_cls and 'cls_token' in name:
                        print("tuning cls_token")
                        param.requires_grad = True
                    if self.mem_type is not None and 'added_mem' in name:
                        print(f"tuning learnable memory tokens {name}")
                        param.requires_grad = True
            else:
                print("vanilla head")
                for name, param in self.net.named_parameters():
                    if "head" not in name:
                        if mem_config is not None and mem_config.CoM and mem_config.inner_mem and 'mem' in name:
                            pass
                        else:
                            param.requires_grad = False
                    if fc_ind:
                        if "fc." in name:
                            param.requires_grad = True
                    if self.tune_cls and 'cls_token' in name:
                        print("tuning cls token")
                        param.requires_grad = True
                    if self.mem_type is not None and 'added_mem' in name:
                        print(f"tuning learnable memory tokens {name}")
                        param.requires_grad = True
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(f"tuning {name}")
        # Define metrics
        self.train_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes, task="multiclass", top_k=5
                ),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes, task="multiclass", top_k=5
                ),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                "acc_top5": Accuracy(
                    num_classes=self.n_classes, task="multiclass", top_k=5
                ),
                "stats": StatScores(
                    task="multiclass", average=None, num_classes=self.n_classes
                ),
            }
        )

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        # Define regularizers
        self.mixup = Mixup(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            prob=self.mix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=self.n_classes,
        )

        # Change to channel last memory format
        # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        if self.channels_last:
            print("Using channel last memory format")
            self = self.to(memory_format=torch.channels_last)

    def forward(self, x):
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)

        return self.net(x)

    def shared_step(self, batch, mode="train"):
        x, y = batch

        if mode == "train":
            # Only converts targets to one-hot if no label smoothing, mixup or cutmix is set
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, num_classes=self.n_classes).float()

        # Pass through network
        pred = self(x)
        if isinstance(pred, torch.Tensor):
            loss = self.loss_fn(pred, y)
            # Get accuracy
            metrics = getattr(self, f"{mode}_metrics")(pred, y.argmax(1))
        else:
            if self.contrastive_type == 'mem':
                contrastive_loss = pred[2]
            elif self.contrastive_type == 'coordinate':
                contrastive_loss = pred[1]
            else:
                contrastive_loss = 0
            if self.contrastive_type is not None:
                self.log(f"{self.contrastive_type}_contra_loss", contrastive_loss, on_epoch=True)

            label_loss = self.loss_fn(pred[0], y)
            loss = self.contrastive_weight*contrastive_loss + label_loss
            
            if self.mem_type == 'recurrent':
                visit_penalty = pred[2]
                self.log(f"visit penalty", visit_penalty, on_epoch=True)
                loss = loss + visit_penalty*self.visit_penalty_weight
            # Get accuracy
            metrics = getattr(self, f"{mode}_metrics")(pred[0], y.argmax(1))

        if self.mem_config is not None and self.mem_config.CoM:
            if self.mem_config.inner_mem:
                for i in self.mem_config.mem_layers:
                    self.log(f"train_visit_entropy_layer{i}", self.net.blocks[i].inner_mem_layer.get_entropy(), on_epoch=True)
                    self.log(f"train_visit_all_time_layer{i}", self.net.blocks[i].inner_mem_layer.train_count_table.sum().float(), on_epoch=True)

            if self.mem_config.outer_mem:
                train_entropy = self.net.head.get_entropy()
                self.log(f"train_visit_entropy", train_entropy, on_epoch=True)
                self.log(f"train_visit_all_time", self.net.head.train_count_table.sum().float(), on_epoch=True)
        # Log
        self.log(f"{mode}_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"{mode}_{k.lower()}", v, on_epoch=True)

        if mode == "test":
            return metrics["stats"]
        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs: List[torch.Tensor]):
        """Save per-class accuracies to csv"""
        # Aggregate all batch stats
        combined_stats = torch.sum(torch.stack(outputs, dim=-1), dim=-1)

        # Calculate accuracy per class
        per_class_acc = []
        for tp, _, _, _, sup in combined_stats:
            acc = tp / sup
            per_class_acc.append((acc.item(), sup.item()))

        # Save to csv
        df = pd.DataFrame(per_class_acc, columns=["acc", "n"])
        df.to_csv("per-class-acc-test.csv")
        print("Saved per-class results in per-class-acc-test.csv")
        return combined_stats

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
