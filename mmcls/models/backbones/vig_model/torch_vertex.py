import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph,DenseDilatedKnnGraphDifferentiable
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init)

norm_cfg=dict(type='SyncBN', requires_grad=True)

class GraphAtten(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True,alpha=0.1):
        super(GraphAtten, self).__init__()
        self.in_channels=in_channels
        self.leakyrelu=nn.LeakyReLU(alpha)
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        self.a = nn.Conv2d(in_channels * 2, 1, 1, bias=bias)
    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        e=self.a(torch.cat([x_i, x_j], dim=1)).squeeze()
        atten=F.softmax(e,-1)
        x_j=(atten.unsqueeze(-1)*x_j.permute(0,2,3,1)).sum(2).transpose(1,2).unsqueeze(-1)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)
class MRConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.in_channels=in_channels
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        # if edge_index is not None:
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        # else:
        #     x_j=y
        B,C,N,K=x_j.shape
        x_j=x_j.reshape(B*C//self.in_channels,self.in_channels,N,K)
        x=x.reshape(B*C//self.in_channels,self.in_channels,N,K)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)
class MRLabel2d(nn.Module):

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)

class EdgeConv2d(nn.Module):
   
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.in_channels=in_channels
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
  
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value

class EdgeConv2dDiff(nn.Module):
   
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2dDiff, self).__init__()
        self.in_channels=in_channels
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x_i,x_j):
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value

class GraphSAGE(nn.Module):
  
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
  
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'gat':
            self.gconv = GraphAtten(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)

class DyGraphConv2dMultiGroup(GraphConv2d):
   
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1,num_head=2):
        super(DyGraphConv2dMultiGroup, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.num_head=num_head
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.fuse_weight = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x_flat = x.reshape(B, C, -1, 1).contiguous()

        dim_per_head = C // self.num_head
        if y is not None:
            y_heads = y.reshape(B * self.num_head, dim_per_head, -1, 1).contiguous()
        else:
            y_heads = None
        x_heads = x_flat.reshape(B * self.num_head, dim_per_head, -1, 1).contiguous()

        edge_index = self.dilated_knn_graph(x_heads, y_heads, relative_pos)
        out_heads = super(DyGraphConv2dMultiGroup, self).forward(x_heads, edge_index, y_heads)
        out_heads = out_heads.reshape(B, -1, H, W).contiguous()  

        if getattr(self, 'use_full_head', True):  
            edge_index_full = self.dilated_knn_graph(x_flat, y, relative_pos)
            out_full = super(DyGraphConv2dMultiGroup, self).forward(x_flat, edge_index_full, y)
            out_full = out_full.reshape(B, -1, H, W).contiguous()  

            if not hasattr(self, 'fuse_weight'):
                self.fuse_weight = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32), requires_grad=True)
            norm_weight = torch.softmax(self.fuse_weight, dim=0)
            out = norm_weight[0] * out_heads + （1-norm_weight[0]） * out_full
            return out, edge_index_full

        else:
            return out_heads, edge_index



class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()

        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous(),edge_index
class DyGraphLabel(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphLabel, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.out_channels=out_channels
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x,y):
        B, C,_ ,_= x.shape
        y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y)
        x= super(DyGraphLabel, self).forward(x, edge_index, y)
        x = x.reshape(B, self.out_channels,-1,1).contiguous()
        return x,edge_index

class DyGraphLabelMultiGroup(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1,num_head=2,bit_graph=True):
        super(DyGraphLabelMultiGroup, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.num_head=num_head
        self.out_channels=out_channels
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon) 
    def forward(self, x,y=None):
        B, C,N,_ = x.shape
        dim_per_head = C // self.num_head
        if y is not None:
            y = y.reshape(B*self.num_head, dim_per_head, -1, 1).contiguous()
        x = x.reshape(B*self.num_head, dim_per_head, -1, 1).contiguous()###拆分
        edge_index = self.dilated_knn_graph(x, y)  ###核心
        x = super(DyGraphLabelMultiGroup, self).forward(x, edge_index, y)
        x = x.reshape(B, self.out_channels,-1,1).contiguous()
        return x,edge_index[0]
    
class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0,
                 relative_pos=False,use_multi_group=False,num_group=2):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        _, norm1 = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            norm1
        )
        if use_multi_group:
            self.graph_conv = DyGraphConv2dMultiGroup(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r,num_head=num_group)
        else:
            self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        _, norm2 = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            norm2
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x,_ = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
class FFNLabel(nn.Module):
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

    def forward(self, x,y=None):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        x = x.transpose(2, 1).squeeze(-1)
        return x  
class GrapherLabel(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False,
                 epsilon=0.0, r=1, n=196,
                 drop_path=0.0, relative_pos=False,
                 num_nodes=80,
                 use_multi_group=False,num_group=2):
        super(GrapherLabel, self).__init__()
        self.channels = in_channels
        _, norm1 = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            norm1
        )
        _, norm2 = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            norm2
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if not use_multi_group:
            self.graph_conv = DyGraphLabel(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                            act, norm, bias, stochastic, epsilon, r)
        else:
            self.graph_conv = DyGraphLabelMultiGroup(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                        act, norm, bias, stochastic, epsilon, r,num_head=num_group)
        self.ffn=FFNLabel(in_channels, in_channels * 4, act=act, drop_path=drop_path)

    def forward(self,x,features):
        B, C, H, W = features.shape
        features=features.reshape(B, C, -1).contiguous()
        x=x.transpose(2, 1).unsqueeze(-1)
        _tmp = x
        x = self.fc1(x)
        x,edge_index = self.graph_conv(x,features)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        x=self.ffn(x)
        return x,edge_index

