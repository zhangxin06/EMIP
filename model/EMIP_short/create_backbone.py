import torch
import torch.nn as nn
from timm.models import create_model
import torch.nn.functional as F


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
        return x


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3):
        zt5_1 = zt5
        zt4_1 = self.conv_upsample1(self.upsample(zt5)) * zt4
        zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) * zt3

        zt4_2 = torch.cat((zt4_1, self.conv_upsample4(self.upsample(zt5_1))), 1)
        zt4_2 = self.conv_concat2(zt4_2)

        zt3_2 = torch.cat((zt3_1, self.conv_upsample5(self.upsample(zt4_2))), 1)
        zt3_2 = self.conv_concat3(zt3_2)

        pc = self.conv4(zt3_2)
        pc = self.conv5(pc)

        res = F.interpolate(pc, scale_factor=8, mode='bilinear')
        return res

class FeatureExtraction(nn.Module):
    def __init__(self, channel=32, pretrained=None, backbone_name='efficientnet_b4', input_shape=None):
        super(FeatureExtraction, self).__init__()
        if input_shape is None:
            input_shape = [128, 320, 512]
        self.pretrained = pretrained

        # ---- create Backbone ----
        self.backbone_name = backbone_name
        print("Creating model: {}".format(self.backbone_name))
        if backbone_name == 'res2net50_26w_4s':
            from lib.Res2Net_v1b import res2net50_v1b_26w_4s
            self.backbone = res2net50_v1b_26w_4s(pretrained=True)
        elif backbone_name == 'efficientnet_b4':
            from lib.EfficientNet import EfficientNet
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        elif backbone_name == 'efficientnet_b1':
            from lib.EfficientNet import EfficientNet
            self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        elif backbone_name == 'pvt_small':
            from lib.pvt import pvt_small
            self.backbone = pvt_small(pretrained=None)
        elif backbone_name == 'pvt_v2_b5':
            from lib.pvt_v2 import pvt_v2_b5
            self.pvtv2_en = pvt_v2_b5(pretrained=None, in_channel_list=input_shape)
        else:
            self.backbone = create_model(self.backbone_name,
                                         pretrained=self.pretrained,
                                         drop_rate=0.0,
                                         drop_path_rate=0.3,
                                         drop_block_rate=None,
                                         )

        if self.pretrained:
            self.load_model(self.pretrained)

    def load_model(self, pretrained):
        pretrained_dict = torch.load(pretrained)['state_dict']
        model_dict = self.state_dict()
        print("Load pretrained parameters from {}".format(pretrained))
        for k, v in pretrained_dict.items():
            if k.startswith('backbone.'):
                k = k.replace('backbone.', 'pvtv2_en.')
            # pdb.set_trace()
            if (k in model_dict):
                print("load:%s" % k)
            else:
                print("jump over:%s" % k)

        pretrained_dict = {k.replace('backbone.', 'pvtv2_en.'): v for k, v in pretrained_dict.items() if (k.replace('backbone.', 'pvtv2_en.') in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        # Feature Extraction
        global x2, x3, x4
        if self.backbone_name == 'res2net50_26w_4s':
            # Feature Extraction
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x1 = self.backbone.layer1(x)
            x2 = self.backbone.layer2(x1)
            x3 = self.backbone.layer3(x2)
            x4 = self.backbone.layer4(x3)

        elif self.backbone_name == 'efficientnet_b4':
            x2 = self.backbone.extract_endpoints['reduction_2']
            x3 = self.backbone.extract_endpoints['reduction_3']
            x4 = self.backbone.extract_endpoints['reduction_4']

        elif self.backbone_name == 'efficientnet_b1':
            x2 = self.backbone.extract_endpoints['reduction_2']
            x3 = self.backbone.extract_endpoints['reduction_3']
            x4 = self.backbone.extract_endpoints['reduction_4']

        elif self.backbone_name == 'pvt_small':
            x1, x2, x3, x4 = self.backbone(x)

        elif self.backbone_name == 'pvt_v2_b5':
            x1, x2, x3, x4 = self.pvtv2_en(x)
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(self.backbone_name))

        return x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, channel=32):
        super(Decoder, self).__init__()
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)

    def forward(self, x):
        x2_rfb, x3_rfb, x4_rfb = x[0],x[1], x[2]
        # Receptive Field Block (enhanced)

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')

        return S_g_pred


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, pretrained=None, backbone_name='efficientnet_b4', input_shape=None):
        super(Network, self).__init__()
        self.channel = channel
        self.pretrained = pretrained
        self.feat_net = FeatureExtraction(channel=channel, pretrained=self.pretrained, backbone_name=backbone_name,
                                             input_shape=input_shape)
        self.decoder = Decoder(self.channel)

    def forward(self, x):
        fmap = self.feat_net(x)
        out = self.decoder(fmap)
        return out


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)
