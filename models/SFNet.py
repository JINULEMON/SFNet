import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from torchstat import stat
import ptflops
import torchvision.models as models
from ptflops import get_model_complexity_info


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,patch_size=2, in_chans=16, out_chans=32):
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.norm = nn.LayerNorm(out_chans)
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=patch_size+1, stride=patch_size, padding=1)  # 将in_chans转化为embed_dim

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # x的shape为[B,C,H*W]再转置为[B,H*W,C]
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # x进行reshape为[B,H,W,C]在permute为[B,C,H,W]
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # act_layer=nn.GELU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # 通道由in_features变为hidden_features
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # 通道由hidden_features变为out_features
        x = self.drop(x)
        return x

class LocalAgg(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)  # 采用1*1卷积，即point-wise卷积
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


'''
class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, head_dim=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # shape为[B,num_heads,H*W,C//num_heads]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)  #x的size为[B,N,C]
        x = self.proj_drop(x)
        return x
'''


class InvertedDepthWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 1x1 pointwise conv
        layers.append(nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride,
                                groups=in_channel, padding=1))
        layers.extend([
            nn.BatchNorm2d(in_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


''''       
class MEW(nn.Module):
    def __init__(self, dim, bias=False, a=16, b=16, c_h=16, c_w=16):
        super().__init__()

        self.register_buffer("dim", torch.as_tensor(dim))
        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))
        self.register_buffer("c_h", torch.as_tensor(c_h))
        self.register_buffer("c_w", torch.as_tensor(c_w))
'''
class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, head_dim=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio1=4):
        super().__init__()

        self.c_weight = nn.Parameter(torch.Tensor(2, dim//2, 12, 12))
        nn.init.ones_(self.c_weight)

        self.wg_c = InvertedDepthWiseConv2d(dim//2, dim//2)

        self.sr1 = sr_ratio1
        if self.sr1 > 1:
            # self.sampler = nn.AvgPool2d(1, sr_ratio)  #二维平均池化:步长为R，将特征下采样sr_ratio倍
            self.sampler1 = nn.MaxPool2d(sr_ratio1)  # 二维平均池化:步长为R，将特征下采样sr_ratio倍
            kernel_size = sr_ratio1
            self.LocalProp1 = nn.ConvTranspose2d(dim // 2, dim // 2, kernel_size, stride=sr_ratio1,
                                                 groups=dim // 2)  # 转置卷积，将特征上采样sr_ratio倍
            self.norm1 = nn.BatchNorm2d(dim // 2)
        else:
            self.sampler1 = nn.Identity()
            self.LocalProp1 = nn.Identity()
            self.norm1 = nn.Identity()

        self.attn = nn.Conv2d(dim // 2, dim // 2, 5, padding=2, groups=dim // 2)
        self.conv1 = nn.Conv2d(dim // 2, dim // 2, 1)  # 采用1*1卷积，即point-wise卷积
        self.dilation1 = nn.Conv2d(dim // 6, dim // 6, 5, dilation=1, padding=2, groups=dim // 6)
        self.dilation2 = nn.Conv2d(dim // 6, dim // 6, 5, dilation=2, padding=4, groups=dim // 6)
        self.dilation3 = nn.Conv2d(dim // 6, dim // 6, 5, dilation=3, padding=6, groups=dim // 6)
        self.conv2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.act=nn.SiLU()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = self.sampler1(x1)
        B, c, a, b = x1.size()

        # ----- c convlution -----#
        x1 = torch.fft.rfft2(x1, dim=(2, 3), norm='ortho')
        c_weight = self.c_weight
        c_weight = self.wg_c(F.interpolate(c_weight, size=x1.shape[2:4],
                                           mode='bilinear', align_corners=True)).permute(1, 2, 3, 0)
        c_weight = torch.view_as_complex(c_weight.contiguous())
        x1 = x1 * c_weight
        x1 = torch.fft.irfft2(x1, s=(a, b), dim=(2, 3), norm='ortho')
        x1 = self.LocalProp1(x1)
        x2 = self.conv1(self.attn(x2))
        x2_1, x2_2, x2_3 = torch.chunk(x2, 3, dim=1)
        x2_1 = self.dilation1(x2_1)
        x2_2 = self.dilation1(x2_2)
        x2_3 = self.dilation1(x2_3)
        x2 = torch.cat([x2_1, x2_2, x2_3], dim=1)
        x2 = self.conv2(x2)
        x = torch.cat([x1,x2],dim=1)
        x = self.act(x)

        return x

class SelfAttn(nn.Module):
    def __init__(self, dim, head_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,sr_ratio1=1):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        #self.norm1 = norm_layer(dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = GlobalSparseAttn(
            dim,
            head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio1=sr_ratio1)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm2 = norm_layer(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        #x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        #x = x.flatten(2).transpose(1, 2)  # x的shape为[B,N,H*W]再转置为[B,H*W,N]
        x = x + self.drop_path(self.attn(self.norm1(x)))  # attn为GlobalSparseAttn
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = x.transpose(1, 2).reshape(B, N, H, W)  # x的shape转置为[B,N,H*W]再reshape为[B,N,H,W]
        return x

class LGLBlock(nn.Module):
    def __init__(self, dim, head_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio1=4.):
        super().__init__()

        self.SelfAttn = SelfAttn(dim, head_dim, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                                 norm_layer,sr_ratio1)

    def forward(self, x):

        x = self.SelfAttn(x)
        return x

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=2):
        super(segmenthead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out

class EdgeVit(nn.Module):
    def __init__(self, in_chans=3,
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=None, num_classes=19,sr_ratio1=4):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.num_classes = num_classes
        self.patch_embed1 = PatchEmbed(patch_size=2, in_chans=3, out_chans=24)
        self.LocalAgg1 = LGLBlock(
            dim=24, head_dim=4, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
            norm_layer=norm_layer,sr_ratio1=4)

        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=24, out_chans=48)

        self.LocalAgg2 = LGLBlock(
            dim=48, head_dim=4, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
            norm_layer=norm_layer,sr_ratio1=2)
        self.LocalAgg2_1 = LGLBlock(
            dim=48, head_dim=4, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
            norm_layer=norm_layer, sr_ratio1=2)

        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=48, out_chans=72)
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.block1=LGLBlock(
            dim=72, head_dim=4, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU, norm_layer=norm_layer,sr_ratio1=1)
        self.block1_1 = LGLBlock(
            dim=72, head_dim=4, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
            norm_layer=norm_layer,sr_ratio1=1)
        self.block1_1_1 = LGLBlock(
            dim=72, head_dim=4, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
            norm_layer=norm_layer,sr_ratio1=1)





    def forward(self, _input):
        x = _input[:,0:3,:,:]
        loc = _input[:,3,:,:].unsqueeze(1)
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        feat4 = self.patch_embed1(x)
        detail_4 = self.LocalAgg1(feat4)
        feat8 = self.patch_embed2(detail_4)
        detail_8 = self.LocalAgg2(feat8)
        detail_8 = self.LocalAgg2_1(detail_8)
        seman_16 = self.patch_embed3(detail_8)
        seman_16 = self.block1(seman_16)
        seman_16 = self.block1_1(seman_16)
        seman_16 = self.block1_1_1(seman_16)



        return detail_4,detail_8,seman_16



def edgevit_xxs(pretrained=False, **kwargs):
    model = EdgeVit(head_dim=4, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=3)
    return model

from einops import rearrange
from torch.nn import *
from mmcv.cnn import build_activation_layer, build_norm_layer
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
import numpy as np
import torch
from torch.nn import Module, ModuleList, Upsample
from mmcv.cnn import ConvModule
from torch.nn import Sequential, Conv2d, UpsamplingBilinear2d
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class CFPModule1(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)  # 利用3*3卷积将通道数由nIn减少为nIn//4
        self.conv1x1_2 = Conv(nIn, nIn, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 4, bn_acti=True)  # dilation=9,深度可分离卷积
        self.dconv_1_1 = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 4, bn_acti=True)  # dialtion=1
        self.dconv_2_1 = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 4, bn_acti=True)  # dialtion=3
        self.dconv_3_1 = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 4, bn_acti=True)  # dialtion=5

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)  # 利用3*3卷积将通道数由nIn减少为nIn//4

        o1_1 = self.dconv_1_1(inp)
        o2_1 = self.dconv_2_1(inp)
        o3_1 = self.dconv_3_1(inp)
        o4_1 = self.dconv_4_1(inp)

        output = torch.cat([o1_1, o2_1, o3_1, o4_1], 1)
        output = self.conv1x1_2(output)
        return output


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=64, embed_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=64, embed_dim=128):
        super().__init__()

        '''
        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())
        '''
        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        # x = x.flatten(2).transpose(1, 2)  #X的size为[B,N,H,W]flatten为[B,N,H*W]再transpose为[B,H*W,N]
        return x

class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, normalize=True,
                 gaussian=False):
        super(FilterHigh, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        self.filter_low = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
        # count_include_pad - 如果等于True，计算平均池化时，将包括padding填充的0
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):

        img = self.filter_low(img) - img
        return img

class Decoder(Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, dims, dim, class_num=3):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels = dims[0], dims[1], dims[2]
        embedding_dim = dim

        #self.CFP_1 = CFPModule1(32, d=8)


        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

        self.filter = FilterHigh(recursions=1, stride=1, kernel_size=5, include_pad=True,
                                 gaussian=False, normalize=False)
        self.linear_predhigh = segmenthead(48, 16, 3)
        self.linear_predlow = segmenthead(48, 16, 3)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, inputs):
        c1, c2, c3 = inputs
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c3.shape



        _c3 = self.linear_c3(c3)
        #_c3 = self.CFP_1(_c3)
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.linear_c2(c2)
        #_c2 = self.CFP_1(_c2)
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = self.linear_c1(c1)


        L2 = self.linear_fuse2(torch.cat([_c3, _c2], dim=1))
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))

        #        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        xhigh = self.filter(_c)
        xhigh = self.linear_predhigh(xhigh)   #边界部分
        xlow = self.linear_predlow(_c)
        attnhigh = self.sigmoid1(xhigh)

        x = xlow + xlow * attnhigh   #边界注意力增强部分
        return xhigh, x


'''
class Decoder(Module):
    def __init__(self, class_num=3):
        super(Decoder, self).__init__()
        self.num_classes = class_num
        self.CFP_1 = CFPModule1(64, d=8)
        self.ps = UpsamplingBilinear2d(scale_factor=2)
        self.linear_fuse4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.linear_fuse3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.linear_fuse2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.linear_fuse1 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        self.linear_pred = nn.Conv2d(4, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        _c4 = self.CFP_1(c4)
        __c4 = self.linear_fuse4(torch.cat([c4, _c4], dim=1))
        __c4 = self.conv4(__c4)
        _c3 = self.ps(__c4)
        __c3 = self.linear_fuse3(torch.cat([c3, _c3], dim=1))
        __c3 = self.conv3(__c3)
        _c2 = self.ps(__c3)
        __c2 = self.linear_fuse2(torch.cat([c2, _c2], dim=1))
        __c2 = self.conv2(__c2)
        _c1 = self.ps(__c2)
        __c1 = self.linear_fuse1(torch.cat([c1, _c1], dim=1))
        c0 = self.linear_pred(__c1)
        return c0
'''

class ssa_PLD(nn.Module):
    def __init__(self, class_num=3, **kwargs):
        super(ssa_PLD, self).__init__()
        self.class_num = class_num
        ######################################load_weight
        self.backbone = edgevit_xxs()
        #####################################
        self.decode_head = Decoder(dims=[24, 48, 72], dim=48, class_num=class_num)
        #self.decode_head = Decoder(class_num=class_num)

    def forward(self, x):
        features = self.backbone(x)

        featureshigh, featureslow = self.decode_head(features)
        # up = UpsamplingBilinear2d(scale_factor=4)
        # features = up(features)
        return {
            # 'coarse_masks': [x_extra_p, x_extra_d],
            'pred_edges': [featureshigh],
            'pred_masks': [featureslow],
            # 'pred_locs': [loc]
        }

    '''
    def _init_weights(self):
        pretrained_dict = torch.load('/mnt/DATA-1/DATA-2/Feilong/scformer/models/ssa/ckpt_S.pth')
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("successfully loaded!!!!")
    '''

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d,nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


if __name__ == '__main__':
    model = ssa_PLD(3)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    #stat(model, (4, 384, 384))
    input = torch.rand(1, 4, 384, 384)
    #stat(model, (4, 640, 480))

    flops, params = get_model_complexity_info(model, (4, 640, 480), as_strings=True,
                                              print_per_layer_stat=True)  # (3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)

    outs = model(input)
    outs = outs['pred_masks']

    #print(outs.shape)

    for feature_map in outs:
        print(feature_map.shape)

