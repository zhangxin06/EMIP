from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

from model.EMIP_long.helpers import *
from model.EMIP_short.create_backbone import NeighborConnectionDecoder, DimensionalReduction
print('Space-time Memory Networks: initialized.')

class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.conv1_m = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                 nn.LayerNorm(64),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(64, 128, 3, 1, 1))
        self.conv1_fusion = nn.Sequential(nn.Conv2d(128, 512, 3, 1, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, 128, 3, 1, 1))

    def forward(self, in_f, in_of):
        x = in_f + in_of
        x = self.conv1_fusion(x)
        return x


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()


    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H * W)  # b, emb, HW

        p = torch.bmm(mi, qi)  # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW

        mo = m_out.view(B, D_o, T * H * W)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out.squeeze(dim=2)], dim=1)

        return mem_out, p


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class LTM(nn.Module):
    def __init__(self):
        super(LTM, self).__init__()
        self.KV_M_r4 = KeyValue(128, keydim=128, valdim=128)
        self.KV_Q_r4 = KeyValue(128, keydim=128, valdim=128)

        self.Memory = Memory()
        self.fusion = fusion()
        self.Decoder = NeighborConnectionDecoder(32)
        self.dr1 = DimensionalReduction(256, 32)
        self.dr2 = DimensionalReduction(320, 32)
        self.dr3 = DimensionalReduction(512, 32)

    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0, 1:num_objects + 1, :, 0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, fea, corr):
        # memorize a frame
        num_objects = 1

        r4 = self.fusion(fea[0], corr)
        k4, v4 = self.KV_M_r4(r4)  # num_objects, 128 and 512, H/16, W/16
        k4 = torch.unsqueeze(torch.unsqueeze(k4, dim=2), dim=0)  # 1 1 128 1 44 44
        v4 = torch.unsqueeze(torch.unsqueeze(v4, dim=2), dim=0)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W))
        em[0, 0] = torch.prod(1 - ps, dim=0)  # bg prob
        em[0, 1:num_objects + 1] = ps  # obj prob
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def segment(self, frame_ori, keys, values, num_objects):
        # pad
        frame = torch.unsqueeze(frame_ori[0], dim=0)

        # r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(frame[0])  # 1, dim, H/16, W/16

        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,0:], values[0,0:], k4, v4)

        return m4

    def forward(self, *args, **kwargs):
        if len(args) > 3:  # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)


