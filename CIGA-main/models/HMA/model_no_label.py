from torchvision.models.resnet import resnet18
import torch
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import numpy as np
from models.HMA.MAB_modules import MHSA
import copy
import torch.nn as nn

from tqdm import tqdm
# from ..npt.model.npt_modules import MHSA
# from ..npt.configs import build_parser
from torchvision.models.resnet import ResNet

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

class lgma(torch.nn.Module):
    def __init__(self, encoder_model:torch.nn.Module, config, num_classes):
        super().__init__()
        self.attention_config, self.memory_config, self.model_config = Dict2Class(config[0]), Dict2Class(config[1]), Dict2Class(config[2]) 

        self.dim_feature = self.model_config.dim_feature
        self.dim_hidden = self.model_config.dim_hidden
        self.dim_out = self.model_config.dim_out
                
        # (a) encode feature, option:
        # 1. simple encode
        # 2. add momentum encoder like MoCo
        self.encoder = encoder_model
        self.encoder_momentum = self.model_config.encoder_momentum
        if self.encoder_momentum:
            self.momen_encoder = copy.deepcopy(encoder_model)
            
        # (b) local memory augmentation
        # create the queue
        self.local_mem_size = self.memory_config.local_mem_size
        if self.local_mem_size:
            # ref MOCO: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
            # distributed is not implemented
            self.register_buffer("queue", torch.randn(self.local_mem_size, self.dim_feature))
            self.queue = nn.functional.normalize(self.queue, dim=1)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.local_read_block = MHSA(self.dim_feature, self.dim_hidden, self.dim_out, self.attention_config)

        
        # (c) aba augmentation, option:
        # 1. only use attention between datapoints
        # 2. add global memory
        self.abd_augmentation = self.model_config.abd_augmentation
        if self.abd_augmentation:
            if self.local_mem_size:
                # after local memory augmentaion, the output is in the shape of (bs, dim_out+label_hidden)
                self.ABD_block = MHSA(self.dim_out, self.dim_hidden, self.dim_out, self.attention_config)
            else:
                self.ABD_block = MHSA(self.dim_feature, self.dim_hidden, self.dim_out, self.attention_config)
            self.num_global_memory = self.memory_config.num_global_memory
            if self.num_global_memory:
                self.global_memories = torch.nn.Parameter(torch.zeros(self.num_global_memory * (num_classes+2), self.dim_feature))
        
        # predict head
        self.num_classes = num_classes
        
        if not self.local_mem_size and not self.abd_augmentation:
            self.fc = torch.nn.Linear(self.dim_feature, self.num_classes)
        else:
            self.fc = torch.nn.Linear(self.dim_out, self.num_classes)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update local memory encoder
        """
        for param_encoder, param_momen_encoder in zip(self.encoder.parameters(), self.momen_encoder.parameters()):
            param_momen_encoder.data = param_momen_encoder.data * self.encoder_momentum + param_encoder.data * (1. - self.encoder_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.local_mem_size % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, x, label=None, update_memory=True):
        """
        Args:
            x: inputs
            label (torch.long, optional): update memory buffer during training. Defaults to None.
            update_memory (bool, optional): select whether update memory. Defaults to True.
        Returns:
            logits
        """
        # (a) extract feature & embedding unknown tokens
        c1 = self.encoder(x)
        # directly output
        if not self.local_mem_size and not self.abd_augmentation:
            logits = self.fc(nn.functional.relu(c1))
            return logits
        ori_bs = c1.shape[0]
        
        # (b) local memory augmentation
        if self.local_mem_size: 
            c1 = c1.unsqueeze(1) # bs, 1, d
            local_memory = self.queue.unsqueeze(1).repeat((c1.shape[0], 1, 1)) # bs, mem_size, d
            local_memory_c = torch.cat([c1, local_memory], dim=1) # bs, mem_size+1, d
            c1 = self.local_read_block(Q=c1, KV=local_memory_c) # bs, 1, d
            c1 = c1.squeeze(1) # bs, d
            # update local memory queue
            if label is not None:
                with torch.no_grad():
                    if self.encoder_momentum:
                        # update encoder
                        self._momentum_update_key_encoder()
                        mem_feature = self.momen_encoder(x)
                        self._dequeue_and_enqueue(mem_feature)
                    else:
                        self._dequeue_and_enqueue(c1.detach())
        
        # (c) attention between data points augmentation
        if self.abd_augmentation:
            if self.num_global_memory:
                global_memory_c = torch.cat([c1, self.global_memories], dim=0) # bs + global_mem_size, d
                c1 = c1.unsqueeze(0) # 1, bs, d
                c1 = self.ABD_block(Q=c1, KV=global_memory_c) # 1, bs, d
                c1 = c1.squeeze(0) # bs, d
            else:
                c1 = c1.unsqueeze(0) # 1, bs, d
                c1 = self.ABD_block(c1) # 1, bs, d
                c1 = c1.squeeze(0) # bs, d
        
        # predict with different features
        logits = self.fc(c1)
        return logits
    

if __name__ == "__main__":
    class Dict2Class(object):
        def __init__(self, my_dict):
            for key in my_dict:
                setattr(self, key, my_dict[key])

    attention_config = {"model_ablate_rff":False, "topk_retrival":False,  "dim_hidden":128, "num_aba_layer":4, 
                   'model_mix_heads': True, "model_num_heads":8, 'model_sep_res_embed': True, 
                   'model_rff_depth': 1, 'model_pre_layer_norm': True, 'viz_att_maps': False, "model_att_block_layer_norm": True, 
                   'model_att_score_norm': 'softmax','model_hidden_dropout_prob': 0.1, 'model_layer_norm_eps': 1e-12,  
                   'model_hidden_dropout_prob': 0.1, 'model_att_score_dropout_prob': 0.1}

    memory_config = {"global_memory":True, "local_memory": True, "num_global_memory":100, "do_mask":True}
    model_config = {"num_classes":10, "dim_hidden":128}

    attention_config = Dict2Class(attention_config)
    memory_config = Dict2Class(memory_config)
    model_config = Dict2Class(model_config)

    cnn_block = resnet18(num_classes=model_config.dim_hidden)
    model = lgma(cnn_block, [attention_config, memory_config, model_config]).cuda()

    local_memory = torch.rand(32, 10, 3, 32, 32).cuda()

    image = torch.rand(32, 3, 32, 32).cuda()
    logits = model(image, local_memory=local_memory)
    print(logits.shape)

    gmem = torch.rand(100, 128).cuda()

    print(model.global_memories.device)
    model.update_global_memory(gmem)
    print(model.global_memories.device)

    logits = model(image, local_memory=local_memory)
    print(logits.shape)
