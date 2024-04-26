import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from loss.warp_utils import flow_warp
from loss.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class Objectview(object):
    def __init__(self, d):
        self.__dict__ = d

    def keys(self):
        return self.__dict__.keys()

class unFlowLoss(nn.modules.Module):
    def __init__(self):
        super(unFlowLoss, self).__init__()
        cfg = {"alpha": 10,
               "ssim_sz": 1,
               "occ_from_back": True,
               "type": "unflow",
               "w_l1": 0.15,
               "w_scales": [1.0, 1.0, 1.0, 1.0, 0.0],
               "w_sm_scales": [1.0, 0.0, 0.0, 0.0, 0.0],
               "w_smooth": 50.0,
               "w_ssim": 0.85,
               "w_ternary": 0.0,
               "warp_pad": "border",
               "with_bk": True}
        cfg = Objectview(cfg)
        self.cfg = cfg
        self.ssim_sz = 1

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1, im1_scaled * occu_mask1, self.ssim_sz)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        # loss = [torch.exp(l) for l in loss]
        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg.keys() and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def compute_loss(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)

            if i == 0:
                if self.cfg.occ_from_back:
                    occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
            else:
                occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                           (h, w), mode='nearest')
                occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                           (h, w), mode='nearest')

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)

            if self.cfg.with_bk:
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        warp_loss = sum(pyramid_warp_losses)

        # smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        # total_loss = warp_loss + smooth_loss
        smooth_loss = 0.0
        total_loss = warp_loss
        return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean()

        # total_loss = warp_loss
        # return total_loss, warp_loss, 0.0, pyramid_flows[0].abs().mean()


class unFlowLoss_decay(nn.modules.Module):
    def __init__(self):
        super(unFlowLoss_decay, self).__init__()
        cfg = {"alpha": 10,
               "ssim_sz": 1,
               "occ_from_back": True,
               "type": "unflow",
               "w_l1": 0.15,
               "w_scales": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               "w_sm_scales": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               "w_smooth": 50.0,
               "w_ssim": 0.85,
               "w_ternary": 0.0,
               "warp_pad": "border",
               "with_bk": True,
               "gamma": 0.8}
        cfg = Objectview(cfg)
        self.cfg = cfg
        self.ssim_sz = 1

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1, im1_scaled * occu_mask1, self.ssim_sz)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        # loss = [torch.exp(l) for l in loss]
        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg.keys() and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def compute_loss(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        n_predictions = len(pyramid_flows)
        for i in range(n_predictions):
            i_weight = self.cfg.gamma ** (n_predictions - i - 1)
            self.cfg.w_scales[i] = i_weight
            self.cfg.w_sm_scales[i] = i_weight

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_scales[i] == 0:  # 不计算了
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)

            if i == 0:
                if self.cfg.occ_from_back:
                    occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
            else:
                occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                           (h, w), mode='nearest')
                occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                           (h, w), mode='nearest')

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)

            if self.cfg.with_bk:
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        warp_loss = sum(pyramid_warp_losses)

        # smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        # total_loss = warp_loss + smooth_loss
        smooth_loss = 0.0
        total_loss = warp_loss
        return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean()

        # total_loss = warp_loss
        # return total_loss, warp_loss, 0.0, pyramid_flows[0].abs().mean()