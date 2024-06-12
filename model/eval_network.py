"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *
from model.network import Decoder, FATBlock


class STCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder() 
        self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        # self.key_proj = KeyProjection(1024, keydim=64) 

        # Compress f16 a bit to use in decoding later on
        # self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.decoder = Decoder()

        # 在将readout送入decoder前,加一个 ASPP 模块 
        # 目的：在保持高分辨率的同时，增加感受野的大小，提高分割精度
        # time: 2023/06/07
        # self.aspp = ASPP(1024)
        # Projection from f16 feature space to key space
        # before it, empose global attention
        self.key_proj1 = KeyProjection(1024, keydim=512)
        self.global_attn1 = FATBlock(512, 512, 9, 16, 2, 4)
        self.global_attn2 = FATBlock(512, 512, 9, 16, 2, 4)
        # self.global_attn3 = FATBlock(512, 512, 9, 16, 2, 4)
        # self.global_attn4 = FATBlock(512, 512, 9, 16, 2, 4)
        self.key_proj2 = KeyProjection(512, keydim=64)

    def encode_value(self, frame, kf16, masks): 
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)  # 1/4, 256       1/8, 512       1/16, 1024
        # k16 = self.key_proj(f16)    # 1024 → 64  1/16, 64
        # f16_thin = self.key_comp(f16) # 1024 → 512 1/16, 512
        # 先投影到dim = 512
        k16_512 = self.key_proj1(f16)   
        k16_attn1 = self.global_attn1(k16_512)
        k16_attn2 = self.global_attn2(k16_attn1)
        
        
        # 再投影到dim = 64 得到key
        k16 = self.key_proj2(k16_attn2)
        
        # 验证不要CAM的fps
        # k16 = self.key_proj2(k16_512)

        return k16, f16, f8, f4

    def segment_with_query(self, mem_bank, qf16, qf8, qf4, qk16): 
        k = mem_bank.num_objects # 不包含背景的目标数

        readout_mem = mem_bank.match_memory(qk16) # qk16(1/16, 64) readout_mem(k, 512, h/16, w/16)
        qf16 = qf16.expand(k, -1, -1, -1)         # k 张frame
        # qv16 = torch.cat([readout_mem, qv16], 1)  # qv16(k, 1024, h/16, w/16)
        # print(torch.max(readout_mem))

        # qf8(1, 512, h/8, w/8) qf4(1, 256, h/4, w/4)
        return torch.sigmoid(self.decoder(readout_mem, qf16, qf8, qf4))
