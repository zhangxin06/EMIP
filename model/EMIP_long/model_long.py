# torch libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# customized libraries
from model.EMIP_short.model import CoUpdater
from model.EMIP_long.LTM import LTM
from model.EMIP_short.motion.PromptInteract import Injector
from model.EMIP_short.create_backbone import DimensionalReduction
from model.EMIP_short.create_backbone import Network, NeighborConnectionDecoder


class FlowEncoder(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.conv1 = nn.Conv2d(2, c, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(c, c * 2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(c * 2, c, 1, stride=1, padding=0)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        if is_list:
            x = torch.split(x, batch_dim, dim=0)
        return x

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

    def forward(self, motion_s, motion_l):
        x = torch.concat(motion_s, motion_l, dim=1)
        x = self.conv1_fusion(x)
        return x

class Model_long(nn.Module):
    def __init__(self, args=None):
        super(Model_long, self).__init__()

        self.args = args
        self.channel = args['channel']
        self.LTM = LTM()
        self.short_term = CoUpdater(args)
        self.long_dr = DimensionalReduction(256, 128)
        # self.fusion = fusion()

        self.injector1 = Injector()
        self.decoder = NeighborConnectionDecoder(32)
        self.dr1 = DimensionalReduction(128, 32)


    def forward(self, frame0, frame1, index, memory_k, memory_v):
        # memorize
        with torch.no_grad():
            fea_1 = self.short_term.backbone.feat_net(frame0.unsqueeze(dim=0))
            fea_2 = self.short_term.backbone.feat_net(frame1.unsqueeze(dim=0))
            fea_1_gm = self.short_term.GMFlow.backbone(frame0.unsqueeze(dim=0))
            fea_2_gm = self.short_term.GMFlow.backbone(frame1.unsqueeze(dim=0))

            a = self.short_term.injector(fea_1_gm[0], fea_1[0])
            b = self.short_term.injector(fea_2_gm[0], fea_2[0])
            flow_fw, flow_bw, corr = self.short_term.GMFlow([a], [b])

            b_c, hw_1, h_2, W_2 = corr.shape
            corr_bw = corr.view(b_c, h_2, W_2, h_2, W_2)
            corr_bw = corr_bw.view(b_c, h_2, W_2, -1)
            corr_bw = corr_bw.permute(0, 3, 1, 2)
            corr_bw = self.short_term.conv_corr(corr_bw)

            corr = self.short_term.conv_corr(corr)
            fea_new_ori = self.short_term.injector1(fea_1[0], corr)
            fea_new = self.short_term.dr1(fea_new_ori)
            f_2 = self.short_term.dr2(fea_1[1])
            f_3 = self.short_term.dr3(fea_1[2])
            mask = self.short_term.decoder(f_3, f_2, fea_new)
            if index == 0:
                return mask, None, None

            f2_2 = self.short_term.dr2(fea_2[1])
            f2_3 = self.short_term.dr3(fea_2[2])

        prev_key, prev_value = self.LTM(fea_1, corr)
        if index == 1:
            this_keys, this_values = prev_key, prev_value
        else:
            this_keys = torch.cat([memory_k, prev_key], dim=3)
            this_values = torch.cat([memory_v, prev_value], dim=3)

            if this_keys.shape[3] > 5:
                this_keys = this_keys[:, :, :, -5:]
                this_values = this_values[:, :, :, -5:]


        memory = self.LTM(fea_2, this_keys, this_values, 1)
        memory = self.long_dr(memory)

        fea_new_ori_long = self.injector1(fea_2[0], memory)
        fea_new_long = self.dr1(fea_new_ori_long)
        mask_long = self.decoder(f2_3, f2_2, fea_new_long)

        return mask_long, this_keys, this_values

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x