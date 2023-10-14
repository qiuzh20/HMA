import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import DataModule
from model_timm import TimmClassificationModel
from src.pl_utils import MyLightningArgumentParser, init_logger
import time
import torch

import os
from os import listdir

import argparse

from src.HMA.model import HMA
from src.coordinatedmemory.coordinateMemory import RecurrentCoordinateMemLayer

model_class = TimmClassificationModel
dm_class = DataModule

torch.set_float32_matmul_precision('medium')

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

parser.add_argument("--name", type=str, default="exp")
parser.add_argument("--logger_type", type=str, help="Name of logger", default="tensorboard", choices=["csv", "tensorboard"])
parser.add_argument("--save_path",type=str,help="Save path of outputs", default="./output/")
parser.add_argument('--store_ckpt', action='store_true')
parser.add_argument("--sub_file", type=str, default=None)
parser.add_argument('--enable_progress_bar', action='store_true')

# Wandb args
parser.add_argument("--project", type=str, help="Name of wandb project", default="default")

# training config
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--val_check_interval", type=int, default=30)
parser.add_argument("--log_every_n_steps", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--scheduler", type=str, default="cosine")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.0)

# data config
parser.add_argument("--dataset", type=str, default="dtd")
parser.add_argument("--size", type=int, default=224)

# base model config
parser.add_argument("--model_name", type=str, default="vit_base_patch32_224.augreg_in21k")
parser.add_argument("--ckpt", type=str, default="./pretrained_ckpt/vit_base_patch32_224.augreg_in21k.pth")
parser.add_argument("--checkpoint", type=str, help="path to model checkpoint", default=None)
parser.add_argument("--default_root_dir", type=str, default="./output")
parser.add_argument('--tune_head', action='store_true')
parser.add_argument('--tune_cls', action='store_false')
parser.add_argument('--use_pretrained', action='store_false')

# add soft-prompts follow [Fine-tuning Image Transformers using Learnable Memory](https://arxiv.org/abs/2203.15243)
parser.add_argument("--mem_type", type=str, default=None, choices=('prop', 'standard', None))
parser.add_argument("--add_mem_num", type=int, default=5)

# HMA configs
parser.add_argument('--HMA', action='store_true')
if (parser.parse_known_args()[0].HMA):
    HMA.register_args(parser)

args = parser.parse_args()


args.save_path = args.save_path + '/' +  args.dataset + f"/{time.strftime('%m%d')}" + f'/tune_head{args.tune_head}' + f'/tune_cls{args.tune_cls}'
if args.sub_file is not None:
    args.save_path = args.save_path + f'/{args.sub_file}'

if args.HMA:
    args.save_path =  args.save_path + f"/HMA"
    if args.aba_ablation:
        args.name = args.name + f"_aba_abla"
    if args.abd_ablation:
        args.name = args.name + f"_abd_abla"
    args.name = args.name + f"_label_hidden_{args.label_hidden}_abd_{args.abd_augmentation}_Rmem_{args.local_mem_size}_Smem{args.global_memory_number}_gaussain_{args.gaussain_mem}_topk_{args.topk}_logits{args.logits_resource}_norm_{args.norm_after_topk}_softmax_temp{args.softmax_temp}_mom{args.encoder_momentum}_null{args.null_KV}"
    if args.mem_type != None:
        args.name = args.name +  args.mem_type + f'_{args.add_mem_num}'
elif args.mem_type is not None:
    args.save_path =  args.save_path + f"/LM"
    args.name = args.name + args.mem_type + f'_{args.add_mem_num}'
else:
    args.save_path = args.save_path + '/baseline'

args.name = args.name + f'_lr{args.lr}_seed{args.seed}'

pl.seed_everything(args.seed)

# Setup trainer
logger = init_logger(args)


checkpoint_callback = ModelCheckpoint(
    # dirpath=args["checkpoint"],
    filename="best-{epoch}-{val_acc:.4f}",
    monitor="val_acc",
    mode="max",
    save_last=False,
    save_top_k=1,
)

if args.checkpoint is not None:
    print("using ckpt from args")
    model = model_class.load_from_checkpoint(args.__dict__, weights=None)
    dm = dm_class.load_from_checkpoint(args.__dict__)
else:
    dm = dm_class(**args.__dict__)
    args.n_classes = dm.num_classes  # Get the number of classes for the dataset
    model = model_class(mem_config=args, **args.__dict__) #["model"])

trainer = pl.Trainer(max_steps=args.max_steps,
                     accelerator='gpu',
                     precision='32',
                    val_check_interval=args.val_check_interval,
                    log_every_n_steps=args.log_every_n_steps,
                    logger=logger, 
                    callbacks=[checkpoint_callback], 
                    check_val_every_n_epoch=None, 
                    enable_progress_bar=args.enable_progress_bar)

begin_time = time.time()

# Train
trainer.tune(model, dm)
trainer.fit(model, dm)


end_time = time.time()

output = trainer.test(model, datamodule=dm, ckpt_path='best')


# save results
print(output)
with open(args.save_path+'/'+args.name+f'/test_{output[0]["test_acc"]}.txt', 'a') as f:
    f.write(str(output))
    f.write(f'\n use {end_time-begin_time}s')

with open(args.save_path + f'/all_test_results.txt', 'a') as f:
    f.write(f"{args.name}: {output[0]['test_acc']}\n")
    f.write(f'{output[0]}\n')
    f.write('\n')

if not args.store_ckpt:
    for file_name in listdir(args.save_path+'/'+args.name + '/version_0/checkpoints'):
        if file_name.endswith('.ckpt'):
            os.remove(args.save_path+'/'+args.name + '/version_0/checkpoints/' + file_name)