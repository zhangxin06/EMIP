# torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# customized libraries
from model.EMIP_short.create_backbone import Network, NeighborConnectionDecoder, DimensionalReduction
from model.EMIP_short.motion.gmflow.gmflow import GMFlow
from model.EMIP_short.motion.PromptInteract import Injector
from model.EMIP_short.motion.common import LayerNorm2d

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class CoUpdater(nn.Module):
    def __init__(self, args=None):
        super(CoUpdater, self).__init__()

        self.args = args
        self.channel = args['channel']
        self.test_mode = args['test_mode']
        self.corr_levels = args['corr_levels']
        self.corr_radius = args['corr_radius']
        self.hidden_dim = args['hidden_dim']
        self.context_dim = args['context_dim']
        self.iters = args['iters']
        self.inp_size = args['inp_size']

        self.backbone = Network(channel=self.channel, pretrained=None, backbone_name=args['backbone_name'],
                                input_shape=args['in_channel_list'])
        self.decoder = NeighborConnectionDecoder(self.channel)
        self.GMFlow = GMFlow(feature_channels=args['GMFlow']['feature_channels'], args=args)
        self.dr1 = DimensionalReduction(128, self.channel)
        self.dr2 = DimensionalReduction(320, self.channel)
        self.dr2_new = nn.Conv2d(128, 32,kernel_size=3, stride=2, padding=1)
        self.dr3 = DimensionalReduction(512, self.channel)
        self.dr3_new = nn.Sequential(nn.Conv2d(128, 64, 3, 2, 1), nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(64, 32, 3, 2, 1), nn.BatchNorm2d(32),
                                 nn.ReLU(inplace=True))
        self.conv_corr =nn.Sequential(nn.Conv2d(44*44, 968, 3, 1, 1),
                                 nn.BatchNorm2d(968),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(968, 128, 3, 1, 1))

        self.injector = Injector()
        self.injector1 = Injector()
        self.transformer_dim_1 = 64
        self.downscaling1 = nn.Sequential(
            nn.Conv2d(self.transformer_dim_1, self.transformer_dim_1 * 2, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim_1 * 2),
            nn.GELU(),
        )
        self.transformer_dim_4 = 512
        self.upscaling4 = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim_4, self.transformer_dim_4 // 2, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim_4 // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.transformer_dim_4 // 2, self.transformer_dim_4 // 4, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.upscaling3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU(),
        )

    def forward(self, image1, image2):
        fea_1 = self.backbone.feat_net(image1)
        fea_2 = self.backbone.feat_net(image2)
        fea_1_gm = self.GMFlow.backbone(image1)
        fea_2_gm = self.GMFlow.backbone(image2)

        a = self.injector(fea_1_gm[0], fea_1[0])
        b = self.injector(fea_2_gm[0], fea_2[0])
        flow_fw, flow_bw, corr = self.GMFlow([a], [b])

        corr = self.conv_corr(corr)
        fea_new = self.injector1(fea_1[0], corr)
        fea_new = self.dr1(fea_new)
        f_2 = self.dr2(fea_1[1])
        f_3 = self.dr3(fea_1[2])
        mask = self.decoder(f_3, f_2, fea_new)
        return mask, flow_fw, flow_bw


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (input_size, input_size),  # (self.image_encoder.img_size, self.image_encoder.img_size)
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


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