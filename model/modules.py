"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import mod_resnet
from model import cbam
import math
import torch.nn.init as init


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1) # B indim H/16 W/16
        x = self.block1(x)         # B outdim H/16 W/16
        r = self.attention(x)      # B outdim H/16 W/16
        x = self.block2(x + r)     # B outdim H/16 W/16

        return x


# Single object version, used only in static image pretraining
# This will be loaded and modified into the multiple objects version later (in stage 1/2/3)
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1) # no, 5, h, w

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16) # 1/16, 512

        return x
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)

class Get_Sr_Ratio(nn.Module):
    def __init__(self, input_channels, N):
        super(Get_Sr_Ratio, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(2)
        self.linear1 = nn.Linear(2 * 2 * input_channels, 128)
        self.linear2 = nn.Linear(128, N)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the adaptive pool
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.softmax(x, 1)
    
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Dynamic_conv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.sbn = nn.SyncBatchNorm(out_planes)

        self.weight_2 = nn.Parameter(torch.Tensor(out_planes, in_planes, 2, 2), requires_grad=True)
        self.bias_2 = nn.Parameter(torch.Tensor(out_planes), requires_grad=True)
        self.weight_4 = nn.Parameter(torch.Tensor(out_planes, in_planes, 4, 4), requires_grad=True)
        self.bias_4 = nn.Parameter(torch.Tensor(out_planes), requires_grad=True)
        # 只取[1, 2, 4]下采样倍率 ---0407
        # self.weight_8 = nn.Parameter(torch.Tensor(out_planes, in_planes, 8, 8), requires_grad=True)
        # self.bias_8 = nn.Parameter(torch.Tensor(out_planes), requires_grad=True)
        # self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size =  2, stride = 2)
        # self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size =  4, stride = 4)
        # self.conv8 = nn.Conv2d(in_planes, out_planes, kernel_size =  8, stride = 8)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_4, a=math.sqrt(5))
        # init.kaiming_uniform_(self.weight_8, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight_2[0, 0].numel())
        nn.init.uniform_(self.bias_2, -bound, bound)
        bound = 1 / math.sqrt(self.weight_4[0, 0].numel())
        nn.init.uniform_(self.bias_4, -bound, bound)
        # bound = 1 / math.sqrt(self.weight_8[0, 0].numel())
        # nn.init.uniform_(self.bias_8, -bound, bound)


    def forward(self, x, ratio_sr):
        # ratio_sr = self.attention(x)      
        ratio_index  =  torch.argmax(ratio_sr, dim=1)[0]
        softmax_attention = ratio_sr[0][ratio_index]
        # print(ratio_index)
        # ratio_index = 1
        if ratio_index == 0:
            # return x*softmax_attention, ratio_index
            return x, ratio_index
        elif ratio_index == 1:
            aggregate_weight = softmax_attention * self.weight_2
            aggregate_bias = softmax_attention * self.bias_2
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=2)
            return self.sbn(output), ratio_index
        
        elif ratio_index == 2:
            aggregate_weight = softmax_attention * self.weight_4
            aggregate_bias = softmax_attention * self.bias_4
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=4)
            return self.sbn(output), ratio_index
        
        # 只取[1, 2, 4]下采样倍率 ---0407
        # elif ratio_index == 3:
        #     aggregate_weight = softmax_attention * self.weight_8
        #     aggregate_bias = softmax_attention * self.bias_8
        #     output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=8)
        #     return self.sbn(output),  ratio_index

        
# 全局注意力
class FASA(nn.Module):

    def __init__(self, dim: int, kernel_size: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_head = num_heads
        self.dim_head = dim // num_heads
        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        # self.kv = nn.Conv2d(self.dim_head, self.dim_head * 2, 1, 1, 0)
        self.kv = nn.Conv2d(self.dim_head *4 , self.dim_head * 2 * 4, 1, 1, 0) # 每4个heads为1组---0407
        # self.pool = Dynamic_conv2d(32, 32)
        self.pool = Dynamic_conv2d(128, 128) # 16个heads分4个group, channels = 32 * 4  ---0407
        # self.attention = Get_Sr_Ratio(32, 4)
        self.attention = Get_Sr_Ratio(128, 3) # 只取[1, 2, 4]下采样倍率 ---0407
        
        self.window_size = window_size
        self.scalor = self.dim_head ** -0.5
        # local branch
        self.local_mixer1 = nn.Sequential(
                get_dwconv(dim, 3, True),
                nn.SyncBatchNorm(dim),
                nn.ReLU(inplace = True)
            )
        self.local_mixer2 = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.SyncBatchNorm(dim)
            )
        self.global_mixer = nn.Conv2d(dim, dim, 1, 1, 0)

    def refined_downsample(self, dim, window_size, kernel_size):
        if window_size==1:
            return nn.Identity()
        for i in range(4):
            if 2**i == window_size:
                break
        block = nn.Sequential()
        for num in range(i):
            block.add_module('conv{}'.format(num), nn.Conv2d(dim, dim, kernel_size, 2, kernel_size//2, groups=dim))
            block.add_module('bn{}'.format(num), nn.SyncBatchNorm(dim))
            if num != i-1:
                block.add_module('linear{}'.format(num), nn.Conv2d(dim, dim, 1, 1, 0))
        return block

    # def forward(self, x: torch.Tensor, sr_ratio: torch.Tensor, sr_index: torch.Tensor):
    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        q_local = self.q(x)
        tensor_list_head = []
        tensor_list_batch = []
        # local
        local_feat = self.local_mixer1(q_local) 
        local_feat = self.local_mixer2(local_feat)
        # global
        for i in range (b):    
            x_sample = x[i].reshape(1, c, h, w)
            # q4ratio = q_local[i].reshape(-1, self.dim_head, h, w).contiguous()  # D C/D H W
            # ratio = self.attention(q4ratio) # D 4 每个head对应4个下采样比率的权重
            
            q4ratio = q_local[i].reshape(-1, self.dim_head * 4, h, w).contiguous()  # D/4 C/D*4 H W   --- 0407
            ratio = self.attention(q4ratio) # D//4 3 每组head对应3个下采样比率的权重 --- 0407
            
            # for j in range(self.num_head): # 遍历每个头
            for j in range(self.num_head // 4): # 遍历每个组 --- 0407
                # head_x = x_sample[-1, self.dim_head * j: self.dim_head * (j + 1)].reshape(1, -1, h, w) # 用于第j个头计算的x
                group_x = x_sample[-1, self.dim_head * j * 4 : self.dim_head * (j + 1) * 4].reshape(1, -1, h, w) # 用于第j个组计算的x --- 0407
                # pool_x, index = self.pool(head_x, ratio[j].reshape(-1, 4)) # 下采样后的x 对应的下采样比例
                pool_x, index = self.pool(group_x, ratio[j].reshape(-1, 3)) # 下采样后的x 对应的下采样比例 --- 0407
                _, _, h_down, w_down = pool_x.shape
                # k, v = self.kv(pool_x).reshape(1, 2, -1, self.dim_head, h_down * w_down).permute(1, 0, 2, 4, 3).contiguous()
                k, v = self.kv(pool_x).reshape(1, 2, -1, self.dim_head*4, h_down * w_down).permute(1, 0, 2, 4, 3).contiguous() # ---0407
                # q4attn = q_local[i, self.dim_head * j: self.dim_head * (j + 1)].reshape(1, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()
                q4attn = q_local[i, self.dim_head * j * 4: self.dim_head * (j + 1) * 4 ].reshape(1, -1, self.dim_head * 4, h * w).transpose(-1, -2).contiguous() # --- 0407
                # 第 j 个头注意力的计算
                # 第 j 个组注意力的计算 --- 0407
                attn = torch.softmax(q4attn @ k.transpose(-1, -2), -1)
                global_feat_j = attn @ v  # (b m (h w) d)
                # global_feat_j = global_feat_j.transpose(-1, -2).reshape(1, 32, h, w)
                global_feat_j = global_feat_j.transpose(-1, -2).reshape(1, 32 * 4, h, w) # ---0407
                tensor_list_head.append(global_feat_j)
            global_feat_head = torch.cat(tensor_list_head, dim=1)
            global_feat_head = self.global_mixer(global_feat_head)
            tensor_list_head.clear()  # 每次计算batch=1后需要清空存放各个head计算得到注意力的列表
            tensor_list_batch.append(global_feat_head)
        global_feat = torch.cat(tensor_list_batch, dim=0)
        # 验证不要 global 的fps
        # q = q_local.reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous() #(b m (h w) d)
        # k, v = self.kv(self.pool(x)).reshape(b, 2, -1, self.dim_head, H*W).permute(1, 0, 2, 4, 3).contiguous() #(b m (H W) d)
        # attn = torch.softmax(self.scalor * q @ k.transpose(-1, -2), -1)
        # global_feat = attn @ v #(b m (h w) d)
        # global_feat = global_feat.transpose(-1, -2).reshape(b, c, h, w)
        # global_feat = self.global_mixer(global_feat)
        
        # local_feat = self.local_mixer1(q_local)   # 验证不要local的fps
        # local_feat = self.local_mixer2(local_feat) # 验证不要local的fps

        return local_feat * global_feat
        # return global_feat  # 验证不要local的fps
        # return local_feat  # 验证不要global的fps
        
    
class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels,  out_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x) #(b c h w)
        return x 


