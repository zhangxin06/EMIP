import torch.nn as nn

from .trident_conv import MultiScaleTridentConv
import torch.nn.functional as F

class DWConv_Adaptor(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_Adaptor, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 **kwargs,
                 ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride,
                                       norm_layer=norm_layer,
                                       )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        # adaptor
        # 第一种 跑了13epoch，最好6epoch, wFm=0.331, Sm-0.650, MAE=0.0147(未加残差)
        # 第一种 跑了49epoch，最好7epoch, wFm=0.327, Sm-0.648, MAE=0.0132(加残差)
        self.dw_dim = 64
        self.ratio = 0.25
        D_hidden_features = int(self.dw_dim * self.ratio)
        self.dwconv64 = nn.Conv2d(64, 64, 3, 1, 1, bias=True, groups=64)
        self.dwconv96 = nn.Conv2d(96, 96, 3, 1, 1, bias=True, groups=96)
        self.dwconv128 = nn.Conv2d(128, 128, 3, 1, 1, bias=True, groups=128)

        self.dwconv_pre = nn.Conv2d(self.dw_dim, D_hidden_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.dwconv = nn.Conv2d(D_hidden_features, D_hidden_features, 3, 1, 1, bias=True, groups=D_hidden_features)
        self.dwconv_post = nn.Conv2d(D_hidden_features, self.dw_dim, kernel_size=3, stride=1, padding=1, bias=False)

        # 第二种
        # self.project_in = nn.Conv2d(self.dw_dim, self.dw_dim * 2, kernel_size=1, bias=False)
        # self.dwconv = nn.Conv2d(self.dw_dim * 2, self.dw_dim * 2, kernel_size=3, stride=1, padding=1,
        #                         groups=self.dw_dim * 2, bias=False)
        # self.project_out = nn.Conv2d(self.dw_dim, self.dw_dim, kernel_size=1, bias=False)

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim,
                                                      kernel_size=3,
                                                      strides=strides,
                                                      paddings=1,
                                                      num_branch=self.num_branch,
                                                      )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x): # (bs,3,352,352)
        x = self.conv1(x) # (bs,64,176,176)
        x = self.norm1(x)
        x = self.relu1(x)

        # x = self.layer1(x)  # 1/2    # (bs,64,176,176)
        # x = self.layer2(x)  # 1/4    # (bs,96,88,88)
        # x = self.layer3(x)  # 1/8 or 1/4   # (bs,128,44,44)

        # zhangxin 加, 可学习adaptor
        # 第一种
        # x = self.dwconv(x)
        # x = self.gelu(x)
        # x = x + F.gelu(self.dwconv(x))

        # 第二种
        # t = x
        # x = self.project_in(x)
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        # x = self.project_out(x)
        # x = t + x

        # 第三种
        x = self.layer1(x)  # 1/2    # (bs,64,176,176)
        # x = x + F.gelu(self.dwconv64(x))
        x = self.layer2(x)  # 1/4    # (bs,96,88,88)
        # x = x + F.gelu(self.dwconv96(x))
        x = self.layer3(x)  # 1/8 or 1/4   # (bs,128,44,44)  10log到此为止
        # x = x + F.gelu(self.dwconv128(x)) # 11.log

        x = self.conv2(x) # (bs,128,44,44)

        if self.num_branch > 1: # 不进
            out = self.trident_conv([x] * self.num_branch)  # high to low res
        else:
            out = [x]

        return out
