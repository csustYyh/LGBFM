import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='my_saves/3080Ti/val_86.7.pth')
parser.add_argument('--davis_path', default='/datanew/vos-datasets/DAVIS/2017')
parser.add_argument('--output', default='./output')
parser.add_argument('--split', help='val/testdev', default='val')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
# 是否使用前一帧作为记忆帧，默认 false
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true') 
args = parser.parse_args()

davis_path = args.davis_path
out_path = os.path.join(args.output, args.split)

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
else:
    raise NotImplementedError

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    with torch.cuda.amp.autocast(enabled=args.amp): 
        rgb = data['rgb'].cuda()    # (B, 帧数, 3, H, W)  B = batch_size = 1
        msk = data['gt'][0].cuda()  # (no, 帧数, 1, H, W) 其中目标数no不包括背景
        info = data['info']
        name = info['name'][0]      # 视频序列名
        k = len(info['labels'][0])  # 目标数（不包括背景）
        size = info['size_480p']    # [H, W] eg:[tensor([480]), tensor([854])]

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
                        mem_every=args.mem_every, include_last=args.include_last)
        processor.interact(msk[:,0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda') # (帧数, 1, H, W)
        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)  # 恢复图片原始尺寸（模型推理时是 pad 后的尺寸）
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)
        
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8) # (帧数, H, W)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        # Save the results
        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

        del rgb
        del msk
        del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)