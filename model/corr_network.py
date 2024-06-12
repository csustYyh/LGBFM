"""
corr_network.py - Correspondence version
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class AttentionMemory(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Mk, Qk): 
        B, CK, H, W = Mk.shape

        Mk = Mk.view(B, CK, H*W) 
        Qk = Qk.view(B, CK, H*W)
        a_sq = Mk.pow(2).sum(1).unsqueeze(2) # B, NE, 1 
        ab = Mk.transpose(1, 2) @ Qk # B, NE, HW
 
        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        
        affinity = F.softmax(affinity, dim=1)

        # B, CK, H, W = Mk.shape

        # Mk = Mk.view(B, CK, H*W) 
        # Mk = torch.transpose(Mk, 1, 2)  # B * HW * CK
 
        # Qk = Qk.view(B, CK, H*W).expand(B, -1, -1) / math.sqrt(CK)  # B * CK * HW
 
        # affinity = torch.bmm(Mk, Qk) # B * HW * HW
        # affinity = F.softmax(affinity, dim=1)


        return affinity

    def readout(self, affinity, mv):
        B, CV, H, W = mv.shape

        mv = mv.flatten(start_dim=2)

        readout = torch.bmm(mv, affinity) # Weighted-sum B, CV, HW
        readout = readout.view(B, CV, H, W)

        return readout

class CorrespondenceNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder() 
        self.key_proj = KeyProjection(1024, keydim=64) 



        self.attn_memory = AttentionMemory()

    def get_query_key(self, frame):
        f16, _, _ = self.key_encoder(frame)
        k16 = self.key_proj(f16)    # 1024 → 64  1/16, 64 
        return k16

    def get_memory_key(self, frame):
        f16, _, _ = self.key_encoder(frame)
        k16 = self.key_proj(f16)    # 1024 → 64  1/16, 64 
        return k16

    def get_W(self, mk16, qk16):
        W = self.attn_memory(mk16, qk16)
        return W

    def transfer(self, W, val):
        return self.attn_memory.readout(W, val)
