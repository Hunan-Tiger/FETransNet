import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath
from functools import partial
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import cv2
import numpy as np

from Shunted import ShuntedTransformer


class SE(nn.Module):
    def __init__(self, dim=192, resolution=56):
        super(SE, self).__init__()
        self.dim = dim
        self.gap = nn.AvgPool2d(resolution, stride=1)
        self.fc1 = nn.Linear(dim, dim // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // 4, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.gap(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.permute(0, 3, 1, 2).contiguous()



# -----------------------------------------------------------------------------------------
def _make_nConv(in_channels, out_channels, nb_Conv):  # 上采样不使用res
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels))
    return nn.Sequential(*layers)


def _make_layer(in_ch, out_ch, block_num, stride=1):  # 下采样使用res
    # 当维度增加时，对shortcut进行option B的处理
    shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
        nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
        nn.BatchNorm2d(out_ch)
    )
    layers = []
    layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace = True原地操作
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super(ConvBatchNorm, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, no_down=False):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.no_down = no_down
        self.nConvs = _make_layer(in_channels, out_channels, nb_Conv, stride=1)

    def forward(self, x):
        out = x
        if self.no_down is False:
            out = self.maxpool(x)
        return self.nConvs(out)


class FocalModulation(nn.Module):
    def __init__(self, dim, proj_drop=0., focal_level=3, focal_window=3,
                 focal_factor=2, use_postln=False):

        super(FocalModulation, self).__init__()
        self.dim = dim
        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, dim + (self.focal_level + 1), bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window  # 9 11 13
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),  # 深度卷积
                    nn.GELU(),
                )
            )

    def forward(self, x_all):
        B, nH, nW, C = x_all.shape
        x_all = self.f(x_all)
        x_all = x_all.permute(0, 3, 1, 2).contiguous()
        ctx, gates = torch.split(x_all, (C, self.focal_level + 1), 1)
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        x_out = ctx_all + ctx_global * gates[:, self.focal_level:]  # 1 192 56 56
        return x_out


class FocalModulationBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=3, focal_window=9, focal_factor=2,
                 use_layerscale=False, layerscale_value=1e-4, Fusion_HW=56):
        super(FocalModulationBlock, self).__init__()
        self.dim = dim
        self.Fusion_HW = Fusion_HW
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_layerscale = use_layerscale

        self.norm1 = norm_layer(dim[0] + dim[1])
        self.dims = dim[0] + dim[1]
        self.modulation = FocalModulation(
            dim=self.dims, focal_window=self.focal_window,
            focal_level=self.focal_level, focal_factor=self.focal_factor, proj_drop=drop)

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dims, out_channels=dim[0], kernel_size=1,
                      stride=1, padding=0, groups=1, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )  # 56 56 64+128 -> 224 224 64
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dims, out_channels=dim[1], kernel_size=1,
                      stride=1, padding=0, groups=1, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.proj1 = nn.Linear(dim[0], dim[0])
        self.proj2 = nn.Linear(dim[1], dim[1])
        
        self.se = SE(dim=dim[0] + dim[1], resolution=Fusion_HW)

    def forward(self, x1, x2, e1, e2):  # x1: 1 64 56 56; e1: 1 64 224 224
        B, _, _, _ = x1.shape
        x_all = torch.cat([x1,x2],dim=1)
        x_all = x_all+self.se(x_all)
        x_all = x_all.permute(0, 2, 3, 1).contiguous()
        x_all = self.norm1(x_all)

        x_out = self.modulation(x_all)  # 1 64+128 56 56
        xo1 = self.down1(x_out)  # 1 64 224 224
        xo2 = self.down2(x_out)  # 1 128 112 112
        e1 = (e1 * xo1 + e1).permute(0, 2, 3, 1).contiguous()
        e2 = (e2 * xo2 + e2).permute(0, 2, 3, 1).contiguous()

        out1 = self.proj1(e1).permute(0, 3, 1, 2).contiguous()
        out2 = self.proj2(e2).permute(0, 3, 1, 2).contiguous()

        return out1, out2


class HWTranslation(nn.Module):  # 通道不变，尺寸统一
    def __init__(self, patchsize, in_channels):
        super(HWTranslation, self).__init__()
        patch_size = _pair(patchsize)
        self.patch = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=patch_size,
                               stride=patch_size)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch(x)
        return x


class Focal(nn.Module):
    def __init__(self, focal_level=3, focal_window=9, focal_factor=2, channel_num=[64, 128],
                 patchSize=[4, 2], fusion_hw=56, depths=2, drop_rate=0., drop_path_rate=0.1):
        super(Focal, self).__init__()
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.embeddings_1 = HWTranslation(self.patchSize_1, in_channels=channel_num[0])
        self.embeddings_2 = HWTranslation(self.patchSize_2, in_channels=channel_num[1])

        self.Inter = FocalModulationBlock(focal_level=self.focal_level, focal_window=self.focal_window,
                                          focal_factor=self.focal_factor, dim=channel_num,
                                          drop=drop_rate, drop_path=drop_path_rate, Fusion_HW=fusion_hw)

    def forward(self, en1, en2):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        x1, x2 = self.Inter(emb1, emb2, en1, en2)
        return x1, x2


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, no_up=False):
        super(UpBlock, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv)
        self.no_up = no_up

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            x = torch.cat([skip_x, x], dim=1)
        x = self.nConvs(x)
        if self.no_up is False:
            x = self.up(x)
        return x


class focal_shunted(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        super(focal_shunted, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64
        self.down1 = DownBlock(n_channels, in_channels, nb_Conv=2, no_down=True)  # 1 64 224 224
        self.down2 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)  # 1 128 112 112
        self.down3 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)  # 1 256 56 56
        self.down4 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)  # 1 512 28 28
        self.down5 = DownBlock(in_channels * 8, in_channels * 16, nb_Conv=2)  # 1 1024 14 14

        self.InterCNN = Focal(channel_num=[in_channels, in_channels * 2], fusion_hw=56)
        self.InterShunted = ShuntedTransformer(img_size=56, in_chans=[256, 512], embed_dims=[256, 512],
                                               num_heads=[8, 16], mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None,
                                               drop_rate=0.,
                                               attn_drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                               sr_ratios=[8, 4])

        self.up5 = UpBlock(in_channels * 16, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlock(in_channels * 8 * 2, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 4 * 2, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 2 * 2, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2, no_up=True)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, img):
        if img.size()[1] == 1:
            x = img.repeat(1,3,1,1)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x1d, x2d = self.InterCNN(x1, x2)
        x3d, x4d = self.InterShunted(x3, x4, x2d)

        x5 = self.up5(x5)  # 1 512 28 28
        x4 = self.up4(x5, x4d)  # 1 256 56 56
        x3 = self.up3(x4, x3d)  # 1 128 112 112
        x2 = self.up2(x3, x2d)  # 1 64 224 224
        x1 = self.up1(x2, x1d)  # 1 64 224 224
        logits = self.outc(x1)

        return logits


if __name__ == "__main__":
    t = torch.randn(1, 1, 224, 224)
    f = focal_shunted()
    out = f(t)
    print(out.shape)