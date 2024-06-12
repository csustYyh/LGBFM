"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *

from model import cbam # 每层解码加入 cbam

# compress = ResBlock(1024, 512) # 验证不要DFAM 要 CAM 的fps
# compress.cuda()
# key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1) # 验证不要DFAM 要 CAM 的fps
# key_comp.cuda()
        

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuser = FeatureFusionBlock(1024+512, 512) # local fusion
        self.cbam1 = cbam.CBAM(512) # 解码时加cbam
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.cbam2 = cbam.CBAM(256) # 解码时加cbam
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4
        self.cbam3 = cbam.CBAM(256) # 解码时加cbam

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

        
        
    def forward(self, memory_read, f16, f8, f4):
        x = self.fuser(f16, memory_read)   # B, 512, H/16, W/16
        
        # x = compress(torch.cat([memory_read, key_comp(f16)], dim=1)) # 验证不要DFAM 要 CAM 的fps
        x = self.cbam1(x) # 解码时加cbam
        x = self.up_16_8(f8, x)  # B, 256, H/8, W/8
        x = self.cbam2(x) # 解码时加cbam
        x = self.up_8_4(f4, x)   # B, 256, H/4, W/4
        x = self.cbam3(x) # 解码时加cbam
        x = self.pred(F.relu(x)) # B, 1, H/4, W/4
 
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) # B, 1, H, W
        
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def get_affinity(self, mk, qk):
        # 预训练时：stage_0
        # mk: memory_key (B, 64, T, H/16, W/16) T = 1 or 2
        # qk: query_key  (B, 64, H/16, W/16)
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)  # (B, 64, T*H*W)
        qk = qk.flatten(start_dim=2)  # (B, 64, H*W)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 

        return affinity

    def readout(self, affinity, mv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        # mem_out = torch.cat([mem, qv], dim=1) # B, 1024, H/16, W/16
        

        return mem

# 全局注意力
class FATBlock(nn.Module):

    def __init__(self, dim: int, out_dim: int, kernel_size: int, num_heads: int, window_size: int, 
                 mlp_ratio: float):
        super().__init__()
        self.dim = dim

        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = FASA(dim, kernel_size, num_heads, window_size)
        self.drop_path = nn.Identity()
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.ffn = ConvFFN(dim, mlp_hidden_dim, out_dim) # FeedForward

    # def forward(self, x: torch.Tensor, sr_ratio:  torch.Tensor, sr_index: torch.Tensor):
    def forward(self, x: torch.Tensor):
        # x = x + self.drop_path(self.attn(self.norm1(x), sr_ratio, sr_index))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    
    
class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO() 
        else:
            self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        # self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        # self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()
        
        # self.pool = Dynamic_conv2d(512, 512)
        # Projection from f16 feature space to key space
        # before it, empose global attention
        self.key_proj1 = KeyProjection(1024, keydim=512)
        self.global_attn1 = FATBlock(512, 512, 9, 16, 2, 4)
        self.global_attn2 = FATBlock(512, 512, 9, 16, 2, 4)
        # self.global_attn3 = FATBlock(512, 512, 9, 16, 2, 4)
        # self.global_attn4 = FATBlock(512, 512, 9, 16, 2, 4)
        self.key_proj2 = KeyProjection(512, keydim=64)

        # self.get_sr_ratio = Get_Sr_Ratio(512, 4)
        

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        # input: b*t*c*h*w 
        # 预训练时：(stage_0) 8 * 3 * 3 * 384 * 384
        b, t = frame.shape[:2]

        # input: b*t, c, h, w  output: f16:1/16, 1024  f8:1/8, 512  f4:1/4, 256
        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        # 投影到 key 特征空间 1/16, 64
        # k16 = self.key_proj(f16)
        # 压缩一下 f16 的特征通道，用以后续解码 1/16, 512
        # f16_thin = self.key_comp(f16)
        
        # 先投影到dim = 512
        k16_512 = self.key_proj1(f16)  
        # sr_ratio = self.get_sr_ratio(k16_512) 
        # sr_ratio_index = torch.argmax(sr_ratio, dim=1)
        # # print(sr_ratio_index)
        k16_attn1 = self.global_attn1(k16_512)
        k16_attn2 = self.global_attn2(k16_attn1)
        # k16_attn1 = self.global_attn1(k16_512)
        # k16_attn2 = self.global_attn2(k16_attn1)
        k16 = self.key_proj2(k16_attn2)

        # B*C*T*H*W
        # 预训练时：(stage_0) 
        # k16:     8 * 64 * 3 * 384/16 * 384/16
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        # 预训练时：(stage_0)
        # f16_thin: 8 * 3 * 512 * 384/16 * 384/16
        # f16:      8 * 3 * 1024 * 384/16 * 384/16
        # f8:       8 * 3 * 512 * 384/8 * 384/8
        # f4:       8 * 3 * 256 * 384/4 * 384/4
        # f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])
        
        return k16, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None): 
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def segment(self, qk16, qf16, qf8, qf4, mk16, mv16, selector=None): 
        # q - query, m - memory
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16)  # B, THW, HW
        
        if self.single_object:
            # 预训练时：stage_0
            # memory.readout(affinity, mv16, qv16): B 1024 H/16 W/16
            # logits: B 1 H W
            # prob:   B 1 H W
            # logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)  # 原代码
            
            logits = self.decoder(self.memory.readout(affinity, mv16), qf16, qf8, qf4) 
            
            prob = torch.sigmoid(logits)
        else:
            # 主训练时：stage_3
            # memory.readout(affinity, mv16, qv16): B 1024 H/16 W/16
            # logits: B 2 H W
            # prob:   B 2 H W

            # logits = torch.cat([
            #     self.decoder(self.memory.readout(affinity, mv16[:,0], qv16), qf8, qf4),
            #     self.decoder(self.memory.readout(affinity, mv16[:,1], qv16), qf8, qf4),
            # ], 1) # 原代码

            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:,0]), qf16, qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:,1]), qf16, qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            # 不确定目标个数 乘以selector[1,0/1] 如果第二个目标不存在，则都为0，否则正常计算
            prob = prob * selector.unsqueeze(2).unsqueeze(2) 


        logits = self.aggregate(prob) # aggregate后，增加了背景维度 （B 2/3 H W） 2→单目标 3→多目标
        prob = F.softmax(logits, dim=1)[:, 1:] # （B 1/2 H W）1→单目标 2→多目标

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


