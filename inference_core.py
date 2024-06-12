import torch
import random

from inference_memory_bank import MemoryBank
from model.eval_network import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by\



class InferenceCore:
    def __init__(self, prop_net:STCN, images, num_objects, top_k=20, mem_every=5, include_last=False):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        # True dimensions
        t = images.shape[1]       # 视频序列帧数
        h, w = images.shape[-2:]  # 宽 高

        # Pad each side to multiple of 16
        # 对frame进行pad,使其高宽是16的整数倍，返回填充后的frame以及填充的像素个数
        # 如原尺寸为480 * 854，填充之后frame为 480 * 864， pad为(5,5,0,0)
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:] # pad 后的宽高(16 的整数倍)

        self.images = images  # (B, 帧数, 3, pad后的高, pad后的宽) B = 1 
        self.device = 'cuda'

        self.k = num_objects  # 目标数，不包括背景

        # Background included, not always consistent (i.e. sum up to 1)
        # prob 即分割概率特征图, (k+1, 帧数, 1, pad后的高, pad后的宽)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.mem_bank = MemoryBank(k=self.k, top_k=top_k) # 实例化一个记忆池

    def encode_key(self, idx): # 对第 idx 张图像进行 key 编码
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
        # k16(1/16, 64), f16_thin(1/16, 512), f16(1/16, 1024), f8(1/8, 512), f4(1/4, 256)
        return result

    def do_pass(self, key_k, key_v, idx, end_idx):
        self.mem_bank.add_memory(key_k, key_v) # 第一帧的 k/v 加入记忆池
        closest_ti = end_idx # 序列中最后一张图片的序号 编号从 1 开始  即第 N 张

        # Note that we never reach closest_ti, just the frame before it 
        # 左闭右开，从0开始存储，故第1张图片为 images[0] 最后一张图片为 images[closest - 1]
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        # 从第2张图片开始处理 第1张图片(images[0])在 interact() 中已处理
        for ti in this_range:
            # k16(1/16, 64), qf16(1/16, 1024), qf8(1/8, 512), qf4(1/4, 256)
            k16, qf16, qf8, qf4 = self.encode_key(ti)
            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf16, qf8, qf4, k16)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:,ti] = out_mask

            if ti != end:
                # is_mem_frame = ((ti % self.mem_every) == 0) # 判断当前帧是否为记忆帧 N = 5
                is_mem_frame = (random.random() < (1 / 3)) # 验证随机帧
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:,ti].cuda(), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    self.mem_bank.add_memory(prev_key, prev_value, is_temp=not is_mem_frame)

        return closest_ti

    def interact(self, mask, frame_idx, end_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame
        key_k, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(), qf16, self.prob[1:,frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)
