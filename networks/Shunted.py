import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class CNNLinear(nn.Module):
    def __init__(self, in_channel, out_channel, kel_size=1, stride=1):
        super(CNNLinear, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kel_size, stride=stride, bias=False)

    def forward(self, x):
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp_shunted(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio == 8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio == 4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio == 2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1,
                                         groups=dim // 2)  # 每个输出卷积核，只与输入的对应的通道进行卷积
            self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1,
                                         groups=dim // 2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]  # B head N C
            k2, v2 = kv2[0], kv2[1]
            attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)

            attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)

            x = torch.cat([x1, x2], dim=-1)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                transpose(1, 2).view(B, C, H, W)).view(B,C,N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_shunted(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):  # 使用新策略，x=LN(PE(LN(x)))
    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.norm_0 = nn.LayerNorm(in_chans)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # b h*w c
        x = self.norm_0(x)
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ShuntedTransformer(nn.Module):  # Small
    def __init__(self, img_size=56, in_chans=[256, 512], embed_dims=[256, 512],
                 num_heads=[8, 16], mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 sr_ratios=[8, 4]):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, 0.2, 2)]
        self.cnn2_3 = CNNLinear(128, 256, kel_size=2, stride=2)
        self.cnn3_4 = CNNLinear(256, 512, kel_size=2, stride=2)

        self.patch_embed_1 = OverlapPatchEmbed(img_size=img_size,
                                          patch_size=3,
                                          stride=1,
                                          in_chans=in_chans[0] * 2,
                                          embed_dim=embed_dims[0])
        self.patch_embed_2 = OverlapPatchEmbed(img_size=img_size // 2,
                                          patch_size=3,
                                          stride=1,
                                          in_chans=in_chans[1] * 2,
                                          embed_dim=embed_dims[1])

        self.block_1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[j], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                                 for j in range(2)])
        self.block_2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[j], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
                                 for j in range(2)])
        self.norm_0 = norm_layer(embed_dims[0])
        self.norm_1 = norm_layer(embed_dims[1])
        
        self.se1 = SE(dim=embed_dims[0]*2, resolution=img_size)
        self.se2 = SE(dim=embed_dims[1]*2, resolution=img_size//2)

    def forward(self, x3, x4, cnnx=None):# cnnx= 1 128 112 112
        B = x3.shape[0]
        #-----
        if cnnx is not None:
            cnnx = self.cnn2_3(cnnx)  # 1 256 56 56
            fu1 = torch.cat([x3, cnnx], dim=1)
        else:
            fu1 = torch.cat([x3, x3], dim=1)
        fu1 = fu1+self.se1(fu1)
        res1 = x3
        fu1,H,W= self.patch_embed_1(fu1)
        for blk in self.block_1:
            fu1 = blk(fu1,H,W)
        fu1 = self.norm_0(fu1)
        fu1 = fu1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3 = fu1+res1
        #------
        fu2 = self.cnn3_4(x3)
        fu2 = torch.cat([fu2, x4], dim=1)
        fu2 = fu2+self.se2(fu2)
        res2 = x4
        fu2, H, W = self.patch_embed_2(fu2)
        for blk in self.block_2:
            fu2 = blk(fu2, H, W)
        fu2 = self.norm_1(fu2)
        fu2 = fu2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x4 = fu2 + res2
        return x3, x4
    
    

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