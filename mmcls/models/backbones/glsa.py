# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn

from torch.nn import Sequential as Seq
from .vig_model import Grapher, act_layer,GrapherLabel

from mmcv.cnn.bricks import DropPath
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,constant_init)
from easydict import EasyDict as edict
import numpy as np
from ..builder import BACKBONES
from .base_backbone import BaseBackbone

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from timm.models.layers import DropPath

import clip  
from clip import tokenize

norm_cfg=dict(type='SyncBN', requires_grad=True)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 20, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        _, norm1 = build_norm_layer(norm_cfg, hidden_features, postfix=1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            norm1
        )
        self.act = act_layer(act)
        _, norm2 = build_norm_layer(norm_cfg, out_features, postfix=1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            norm2
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  

class Stem(nn.Module):

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        _, norm1 = build_norm_layer(norm_cfg, out_dim // 2, postfix=1)
        _, norm2 = build_norm_layer(norm_cfg, out_dim, postfix=1)
        _, norm3 = build_norm_layer(norm_cfg, out_dim, postfix=1)
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            norm1,
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            norm2,
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            norm3
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        _, norm = build_norm_layer(norm_cfg, out_dim, postfix=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            norm
        )

    def forward(self, x):
        x = self.conv(x)
        return x

@BACKBONES.register_module()
class GLSANet(BaseBackbone):
    arch_settings = edict({
        't':{"k" : 9,  
            'conv' : 'mr',  
            'act' : 'gelu' , 
            'norm' : 'batch',  
            'bias' : True , 
            'dropout' : 0.0  ,
            'use_dilation' : True , 
            'epsilon' : 0.2 , 
            'use_stochastic' : False , 
            'blocks' : [2, 2, 6, 2]  ,
            'channels' : [48, 96, 240, 384] , 
            'emb_dims' : 1024 , 
     },
        's': {"k": 9,  
              'conv': 'mr',  
              'act': 'gelu',  
              'norm': 'batch',  
              'bias': True,  
              'dropout': 0.0,  
              'use_dilation': True,  
              'epsilon': 0.2,  
              'use_stochastic': False,  
              'blocks': [2, 2, 6, 2], 
              'channels': [80, 160, 400, 640],  
              'emb_dims': 1024,  
              },
    })
    def __init__(self,
                 choice='s',
                 k=9,
                 k_label_gcn=9,
                 use_multi_group=True, 
                 backbone_multi_group=True,
                 num_group=2,
                 drop_path=0.0,
                 n_classes=1000,
                 out_indices=(3,),
                 size=576,
                 num_gcn=1,
                 pretrain_path=None,
                 init_cfg=None,
                 use_clip_label=False   
                 ):
        super(GLSANet, self).__init__(init_cfg)
        opt=self.arch_settings[choice]
        k = k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = drop_path

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]                                        
        max_dilation = 49 // max(num_knn)

        # Label Embeddings
        print("--------------------do you use clip--------------")
        print(use_clip_label)
        self.use_clip_label = use_clip_label  
        if self.use_clip_label:
           
            clip_model, _ = clip.load('./checkpoint/RN50.pt', device='cpu', jit=False) ##ViT-B-16.p  # 加载 CLIP 模型
            clip_model.eval()  
            
            n_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]
            prompts = [f"a photo of a {name}" for name in n_classes]
            tokenized = clip.tokenize(prompts).to('cpu')

            with torch.no_grad():
                clip_feats = clip_model.encode_text(tokenized).float()  # [n_classes, 512]

            
            clip_feats = clip_feats.cuda()
            self.register_buffer("clip_label_embed", clip_feats) 
            self.label_proj = nn.Sequential(
                nn.Linear(clip_feats.shape[-1], 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, channels[0])  
            del clip_model  
        else:
            self.label_input = torch.Tensor(np.arange(n_classes)).view(1, -1).long()
            self.label_lt = torch.nn.Embedding(n_classes,channels[0], padding_idx=None)
            
        
        self.layer_index = [np.sum(blocks[:i + 1]) + i - 1 for i in range(0,len(blocks))]

        self.out_indices = [np.sum(blocks[:i+1])+i-1 for i in out_indices]

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], size // 4, size // 4))
        HW = size // 4 * size // 4

        self.backbone = nn.ModuleList([])
        self.gcn_label = nn.ModuleList([])
        self.ffn_label = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i <len(blocks) - 1:
                self.gcn_label += [
                    Seq(GrapherLabel(channels[i], k_label_gcn, 1,
                                     'mr',
                                     act, norm,
                                     bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                     relative_pos=False, num_nodes=n_classes,
                                     use_multi_group=use_multi_group,
                                     num_group=num_group
                                     ),
                        )]
                self.ffn_label+= [
                        Seq(
                            nn.Linear(channels[i], channels[i + 1]),

                            )]
            else:
                self.gcn_label += Seq(nn.ModuleList(GrapherLabel(channels[i], k_label_gcn, 1,
                                                                 'mr',
                                                                 act, norm,
                                                                 bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                                                 relative_pos=False, num_nodes=n_classes,
                                                                 use_multi_group=use_multi_group,
                                                                 num_group=num_group
                                                                 )
                                                for _ in range(num_gcn)))

            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True,use_multi_group=backbone_multi_group,num_group=num_group),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.gcn_label = Seq(*self.gcn_label)
        self.ffn_label = Seq(*self.ffn_label)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    def init_weights(self):
        super(GLSANet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            return

    def forward(self, inputs):

        if self.use_clip_label:
            label_emb = self.label_proj(self.clip_label_embed)
            init_label_embeddings = label_emb.unsqueeze(0).repeat(inputs.size(0), 1, 1)
        else:
            const_label_input = self.label_input.repeat(inputs.size(0), 1).to(inputs.device)
            init_label_embeddings = self.label_lt(const_label_input)
        
        oral_label=init_label_embeddings
        x = self.stem(inputs) + self.pos_embed
        j=0
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in self.layer_index:
                for k in range(len(self.gcn_label[j])):
                    init_label_embeddings,edge_index=self.gcn_label[j][k](init_label_embeddings,x)
                if j<3:
                    init_label_embeddings=self.ffn_label[j](init_label_embeddings)
                j+=1
        features_gap=torch.flatten(self.gap(x), 1)
        outs=[init_label_embeddings,features_gap,edge_index,oral_label]

        return tuple(outs)


