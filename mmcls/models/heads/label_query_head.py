import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from ..builder import HEADS
from .cls_head import ClsHead
from ..losses.label_smooth_loss import LabelSmoothLoss
import pickle  
import numpy as np


@HEADS.register_module()
class LabelQueryHead(ClsHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 softmax=False,
                 double_loss=True,
                 use_align_loss=True,   
                 align_loss_weight=1,   
                 save_align_vis=False,   
                 align_vis_dir='work_dirs/clip_align_heatmap',  
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args, **kwargs):
        super().__init__(init_cfg=init_cfg, *args, **kwargs)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.softmax = softmax
        self.double_loss = double_loss

        self.use_align_loss = use_align_loss
        self.align_loss_weight = align_loss_weight
        self.save_align_vis = save_align_vis
        self.align_vis_dir = align_vis_dir
        self.count=1

        self.bce_loss = LabelSmoothLoss(label_smooth_val=0.1, mode='multi_label', reduction='mean')

        self.fc1 = nn.Linear(self.in_channels, self.num_classes)
        self.fc2 = nn.Linear(self.in_channels, self.num_classes)
        
   
        self.use_logit_bias = False  
        self.logit_bias_weight=0.5 
        
        self.use_sim_proj = True
        self.sim_bias_weight=0  
        self.sim_loss_weight=4
        
        
        if self.use_sim_proj:
            self.sim_proj = nn.Linear(self.in_channels, 80)

        if self.use_logit_bias:
            freq_path = os.path.join(os.path.dirname(__file__), '../../../data/class_freq.pkl')
            with open(freq_path, 'rb') as f:
                freq_data = pickle.load(f)
            class_freq = torch.tensor(freq_data['class_freq'], dtype=torch.float32)
            class_freq[class_freq == 0] = 1.0
            logit_bias = torch.log(class_freq + 1e-6)
            self.register_buffer('logit_bias', logit_bias)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def get_score(self, x):
        x0 = x[0]
        x1 = x[1]
        output1 = self.fc1(x0)
        diag_mask = torch.eye(output1.size(1)).unsqueeze(0).repeat(output1.size(0), 1, 1).to(output1.device)
        output1 = (output1 * diag_mask).sum(-1)
        output3 = self.fc2(x1)
        cls_score = output1 + output3
        
        if self.use_logit_bias:
            cls_score = cls_score - self.logit_bias[None, :]*self.logit_bias_weight
     
        if self.use_sim_proj:
            final_label = x[0]      
            original_label = x[3]   
            final_proj = self.sim_proj(final_label)   
            sim_logits = torch.sum(F.normalize(original_label, dim=-1) * F.normalize(final_proj, dim=-1), dim=-1)  
            mask = (sim_logits >= 0.4) & (sim_logits <= 0.6)
            bias_score = torch.where(mask, sim_logits * self.sim_bias_weight, torch.zeros_like(sim_logits))
            cls_score = cls_score + bias_score
      
        if self.save_align_vis:
            return cls_score,final_proj,output1,output3,sim_logits,x[0],x[1]
        else:
            return cls_score,final_proj
    

    def simple_test(self, x, softmax=False, post_process=True,**kwargs):
        if self.save_align_vis:
            cls_score,_,output1,output3,sim,label_emb,image_feat= self.get_score(x)
        else:
            cls_score,_=self.get_score(x)
         
        if self.softmax:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        return self.post_process(pred) if post_process else pred

    def forward_train(self, x, gt_label, **kwargs):
        cls_score,final_proj = self.get_score(x)
        _gt_label = torch.abs(gt_label)
        if self.softmax:
            gt_label = gt_label.view(-1).long()

        losses = self.loss(cls_score, gt_label, **kwargs)
        bce_loss = self.bce_loss(cls_score, gt_label, avg_factor=len(cls_score), **kwargs)

        loss_dict = {
            #'bce_loss': bce_loss,
            'asy_loss': losses['loss'] * 10.0 if self.double_loss else losses['loss']
        }
        
        if self.use_align_loss: 
            image_feat = x[1]       
            label_emb = x[0]         

            norm_img_feat = F.normalize(image_feat, dim=-1)            
            norm_label_emb = F.normalize(label_emb, dim=-1)            
            sim_matrix = torch.einsum('bc,bkc->bk', norm_img_feat, norm_label_emb)  
            gt_mask = (gt_label > 0).float()  
        
            pos_sim = sim_matrix * gt_mask  
            align_loss = (1 - pos_sim).sum() / (gt_mask.sum() + 1e-6)
            loss_dict['align_loss'] = align_loss * self.align_loss_weight
            
            
        if self.use_sim_proj and self.sim_loss_weight > 0:
            original_label = x[3]   
            final_proj = final_proj

            sim_logits = F.cosine_similarity(original_label, final_proj, dim=-1).clamp(min=0.0, max=1.0)
            with torch.no_grad():
                mask = (gt_label > 0).float()  
            pos_weight=1
            neg_weight=2
            contrastive_loss = (pos_weight *(1 - sim_logits) * mask + neg_weight *sim_logits * (1 - mask)).sum() / (mask.numel() + 1e-6)
            loss_dict['sim_contrastive_loss'] = contrastive_loss * self.sim_loss_weight
            
        return loss_dict