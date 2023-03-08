import spconv
import copy
import time
import torch
import numpy as np
from torch import nn
from numpy import log
from pdb import set_trace as bp
from core.models.registry import BBONES2D, HEAD, BBONES3D, MODELS
from configs.system_config import *
from core import box_np_ops
from core.gaussian import gaussian_radius, draw_heatmap_gaussian

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn

import torch
from torch import nn
from torch.nn import functional as F

import pdb

# ===================
#     RGA Module
# ===================

class RGA_Module(nn.Module):
	def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
		cha_ratio=16, spa_ratio=16, down_ratio=16):
		super(RGA_Module, self).__init__()

		self.in_channel = in_channel
		self.in_spatial = in_spatial
		
		self.use_spatial = use_spatial
		self.use_channel = use_channel

		print ('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

		self.inter_channel = in_channel // cha_ratio
		self.inter_spatial = in_spatial // spa_ratio
		
		# Embedding functions for original features
		if self.use_spatial:
			self.gx_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		if self.use_channel:
			self.gx_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		
		# Embedding functions for relation features
		if self.use_spatial:
			self.gg_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		if self.use_channel:
			self.gg_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel*2, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		
		# Networks for learning attention weights
		if self.use_spatial:
			num_channel_s = 1 + self.inter_spatial
			self.W_spatial = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_s//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)
		if self.use_channel:	
			num_channel_c = 1 + self.inter_channel
			self.W_channel = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_c//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_c//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)

		# Embedding functions for modeling relations
		if self.use_spatial:
			self.theta_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
			self.phi_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		if self.use_channel:
			self.theta_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
			self.phi_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
				
	def forward(self, x):
		b, c, h, w = x.size()
		
		if self.use_spatial:
			# spatial attention
			theta_xs = self.theta_spatial(x)	
			phi_xs = self.phi_spatial(x)
			theta_xs = theta_xs.view(b, self.inter_channel, -1)
			theta_xs = theta_xs.permute(0, 2, 1)
			phi_xs = phi_xs.view(b, self.inter_channel, -1)
			Gs = torch.matmul(theta_xs, phi_xs)
			Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w)
			Gs_out = Gs.view(b, h*w, h, w)
			Gs_joint = torch.cat((Gs_in, Gs_out), 1)
			Gs_joint = self.gg_spatial(Gs_joint)
		
			g_xs = self.gx_spatial(x)
			g_xs = torch.mean(g_xs, dim=1, keepdim=True)
			ys = torch.cat((g_xs, Gs_joint), 1)

			W_ys = self.W_spatial(ys)
			if not self.use_channel:
				out = F.sigmoid(W_ys.expand_as(x)) * x
				return out
			else:
				x = F.sigmoid(W_ys.expand_as(x)) * x

		if self.use_channel:
			# channel attention
			xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
			theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
			phi_xc = self.phi_channel(xc).squeeze(-1)
			Gc = torch.matmul(theta_xc, phi_xc)
			Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
			Gc_out = Gc.unsqueeze(-1)
			Gc_joint = torch.cat((Gc_in, Gc_out), 1)
			Gc_joint = self.gg_channel(Gc_joint)

			g_xc = self.gx_channel(xc)
			g_xc = torch.mean(g_xc, dim=1, keepdim=True)
			yc = torch.cat((g_xc, Gc_joint), 1)

			W_yc = self.W_channel(yc).transpose(1, 2)
			out = F.sigmoid(W_yc) * x

			return out

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# from core.visualization_openpcdet import draw_scenes
class BasisBlock(nn.Module):
    """
    BasisBlock for input to ResNet
    """

    def __init__(self, n_input_channels):
        super(BasisBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


#################
# Residual Unit #
#################


class ResidualUnit(nn.Module):
    def __init__(self, n_input, n_output, downsample=False):
        """
        Residual Unit consisting of two convolutional layers and an identity mapping
        :param n_input: number of input channels
        :param n_output: number of output channels
        :param downsample: downsample the output by a factor of 2
        """
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # down-sampling: use stride two for convolutional kernel and create 1x1 kernel for down-sampling of input
        self.downsample = None
        if downsample:
            self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                            nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True))
        else:
            self.identity_channels = nn.Conv2d(n_input, n_output, kernel_size=(1, 1), bias=False)

    def forward(self, x):

        # store input for skip-connection
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # downsample input to match output dimensions
        if self.downsample is not None:
            identity = self.downsample(identity)
        else:
            identity = self.identity_channels(identity)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x


##################
# Residual Block #
##################


class ResidualBlock(nn.Module):
    """
        Residual Block containing specified number of residual layers
        """

    def __init__(self, n_input, n_output, n_res_units):
        super(ResidualBlock, self).__init__()

        # use down-sampling only in the first residual layer of the block
        first_unit = True

        # specific channel numbers
        if n_res_units == 3:
            inputs = [n_input, n_output//4, n_output//4]
            outputs = [n_output//4, n_output//4, n_output]
        else:
            inputs = [n_input, n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output]
            outputs = [n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output, n_output]

        # create residual units
        units = []
        for unit_id in range(n_res_units):
            if first_unit:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id], downsample=True))
                first_unit = False
            else:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id]))
        self.res_block = nn.Sequential(*units)

    def forward(self, x):

        x = self.res_block(x)

        return x


#############
# FPN Block #
#############


class FPNBlock(nn.Module):
    """
        Block for Feature Pyramid Network including up-sampling and concatenation of feature maps
        """

    def __init__(self, bottom_up_channels, top_down_channels, fused_channels):
        super(FPNBlock, self).__init__()
        # reduce number of top-down channels to 196
        intermediate_channels = 196
        if top_down_channels > 196:
            self.channel_conv_td = nn.Conv2d(top_down_channels, intermediate_channels, kernel_size=(1, 1),
                                             stride=(1, 1), bias=False)
        else:
            self.channel_conv_td = None

        # change number of bottom-up channels to 128
        self.channel_conv_bu = nn.Conv2d(bottom_up_channels, fused_channels, kernel_size=(1, 1),
                                         stride=(1, 1), bias=False)

        # transposed convolution on top-down feature maps
        if fused_channels == 128:
            out_pad = (1, 1)
        else:
            out_pad = (1, 1)
        if self.channel_conv_td is not None:
            self.deconv = nn.ConvTranspose2d(intermediate_channels, fused_channels, kernel_size=(3, 3), padding=(1, 1),
                                             stride=2, output_padding=out_pad)
        else:
            self.deconv = nn.ConvTranspose2d(top_down_channels, fused_channels, kernel_size=(3, 3), padding=(1, 1),
                                             stride=2, output_padding=out_pad)

    def forward(self, x_td, x_bu):

        # apply 1x1 convolutional to obtain required number of channels if needed
        if self.channel_conv_td is not None:
            x_td = self.channel_conv_td(x_td)

        # up-sample top-down feature maps
        x_td = self.deconv(x_td)

        # apply 1x1 convolutional to obtain required number of channels
        x_bu = self.channel_conv_bu(x_bu)

        # perform element-wise addition
        try:
            x = x_td.add(x_bu)
        except:
            print(x_td.shape," ", x_bu.shape)
            print("Checking shape of this")
            exit()

        return x


####################
# Detection Header #
####################

class DetectionHeader(nn.Module):

    def __init__(self, n_input, n_output):
        super(DetectionHeader, self).__init__()
        basic_block = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
        self.conv1 = basic_block
        self.conv2 = copy.deepcopy(basic_block)
        self.conv3 = copy.deepcopy(basic_block)
        self.conv4 = copy.deepcopy(basic_block)
        self.classification = nn.Conv2d(n_output, 1, kernel_size=(3, 3), padding=(1, 1))
        self.regression = nn.Conv2d(n_output, 8, kernel_size=(3, 3), padding=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        class_output = self.sigmoid(self.classification(x))
        regression_output = self.regression(x)

        return class_output, regression_output

def rotate_nms_cc(dets, scores, thresh, nms_pre=None):
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 2:4],
                                                     dets[:, 4])

    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)

    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)
    # print(dets_corners.shape, order.shape, standup_iou.shape)
    return rotate_non_max_suppression_cpu(dets_corners, order, standup_iou,
                                          thresh)

def from_2d_box_to_3d(dd_box):
    ddd_box = torch.zeros((dd_box.shape[0], 7)).float()
    ddd_box[:,0] = dd_box[:, 0]
    ddd_box[:,1] = dd_box[:, 1]
    ddd_box[:,3] = dd_box[:, 2]
    ddd_box[:,4] = dd_box[:, 3]
    ddd_box[:,6] = dd_box[:, 4]

    return ddd_box

def my_rotate_nms(box_preds_class, top_scores_class, top_labels_class, iou_thres):
    order = top_scores_class.sort(descending=True)[1]
    box_preds_class = box_preds_class[order]
    top_scores_class = top_scores_class[order]
    top_labels_class = top_labels_class[order]
    box_preds_class_3d = from_2d_box_to_3d(box_preds_class)
    iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(box_preds_class_3d, box_preds_class_3d)
    iou_matrix.fill_diagonal_(0.0)
    total_box = box_preds_class.shape[0]
    check_ = torch.zeros(total_box)
    selected_index = []
    for i in range(total_box):
        if check_[i] == 0:
            selected_index.append(i)
            mask = iou_matrix[i] > iou_thres
            check_[mask] = 1

    box_preds_class = box_preds_class[selected_index]
    top_scores_class = top_scores_class[selected_index]
    top_labels_class = top_labels_class[selected_index]
    return box_preds_class, top_scores_class, top_labels_class

@MODELS.register_module()
class base2d(nn.Module):

    def __init__(self, config_box):

        super(base2d, self).__init__()
        # Take configuration
        config = config_box["model_config"]
        data_config = config_box["data_config"]
        grid_size = config_box["grid_size"]
        try:
            self.attention = config.MODEL.attention
        except:
            self.attention = ""

        # Process data

        self.output_shape = [30, 800, 800]

        # self.middle_layer = BBONES3D.get(config.MODEL.BBONES3D.name)(config.MODEL.num_input_features, self.output_shape) #TODO H: assign doesn't match with config
        self.num_classes = config["num_classes"]

        self.name_to_id = data_config["CLASS_ENCODE"]
        self.id_to_name = dict([(v, k) for k, v in self.name_to_id.items()])

        self.detection_range = data_config["POINT_CLOUD_RANGE"]
        self.post_center_range = copy.copy(self.detection_range)

        self.out_size_factor = 4
        self.feature_x = int(np.ceil(grid_size[0] / self.out_size_factor))
        self.feature_y = int(np.ceil(grid_size[1] / self.out_size_factor))

        self.pre_define_vector = torch.zeros((self.feature_y, self.feature_x, 2))
        for i in range(self.feature_y):
            self.pre_define_vector[i,:,0] = torch.arange(0, self.feature_x, 1)

        for i in range(self.feature_x):
            self.pre_define_vector[:,i,1] = torch.arange(0, self.feature_y, 1)

        self.pre_define_vector = self.pre_define_vector.cuda()
        self.voxel_size = data_config["VOXEL_GENERATOR"]["VOXEL_SIZE"]
        self.gaussian_overlap = 0.1
        self.min_radius = 2
        self.max_objs = 600

        self.basis_block = BasisBlock(n_input_channels=30)
        block = {'C': MBConv, 'T': Transformer}

        self.se1 = nn.Identity() 
        self.se2 = nn.Identity() 
        self.se3 = nn.Identity() 
        self.se4 = nn.Identity() 

        self.res_block_1 = ResidualBlock(n_input=32, n_output=96, n_res_units=3)
        if self.attention == "se":
            self.se1 = SELayer(96)

        if self.attention == "cbam":
            self.se1 = CBAM(96)

        if self.attention == "rotate":
            self.se1 = TripletAttention()
        
        if self.attention == "rga":
            self.se1 = nn.Identity()

        if self.attention == "":
            self.se1 = nn.Identity()

        self.res_block_2 = ResidualBlock(n_input=96, n_output=196, n_res_units=6)
        if self.attention == "se":
            self.se2 = SELayer(196)

        if self.attention == "cbam":
            self.se2 = CBAM(196)
        
        if self.attention == "rotate":
            self.se2 = TripletAttention()

        if self.attention == "rga":
            #self.se2 = RGA_Module(196, 200 * 200)
            self.se2 = nn.Identity()

        if self.attention == "":

            self.se2 = nn.Identity()

        self.res_block_3 = ResidualBlock(n_input=196, n_output=256, n_res_units=6)
        if self.attention == "se":
            self.se3 = SELayer(256)

        if self.attention == "cbam":
            self.se3 = CBAM(256)

        if self.attention == "rotate":
            self.se3 = TripletAttention()
        
        if self.attention == "rga":
            self.se3 = RGA_Module(256, 100* 100)
            #self.se3 = nn.Identity()

        if self.attention == "":
            self.se3 = nn.Identity()
        
        # if self.attention == "coatnet":
        #     self.se3 = nn.Identity()
        #     self.res_block_3 = self._make_layer(block["T"],inp=196,oup=256, depth=2, image_size=(800//8, 800//8))

        self.res_block_4 = ResidualBlock(n_input=256, n_output=384, n_res_units=3)
        if self.attention == "se":
            self.se4 = SELayer(384)

        if self.attention == "cbam":
            self.se4 = CBAM(384)

        if self.attention == "rotate":
            self.se4 = TripletAttention()

        if self.attention == "":
            self.se4 = nn.Identity(
            )
            
        if self.attention == "rga":
            self.se4 = RGA_Module(384, 50 * 50)
            #self.se4 = nn.Identity()
        
        if self.attention == "coatnet":
            self.se4 = nn.Identity()
            self.res_block_4 = self._make_layer(block["T"],inp=256,oup=384, depth=4, image_size=(800//16, 800//16))
        

        # FPN blocks
        self.fpn_block_1 = FPNBlock(top_down_channels=384, bottom_up_channels=256, fused_channels=128)
        self.fpn_block_2 = FPNBlock(top_down_channels=128, bottom_up_channels=196, fused_channels=96)

        # Detection Header
#        self.header = DetectionHeader(n_input=96, n_output=96)
        # self.BEVNet1 = BBONES2D.get(config.MODEL.BBONES2D.name)(config.MODEL.BBONES2D)
        self.rpn1 = HEAD.get(config.MODEL.HEAD.name)(config.MODEL.HEAD)
        
        # self.iou_aware = IOU_AWARE(config.MODEL.HEAD.num_input_features, 1)

        self.training = False

        self._nms_score_thresholds = [0.2, 0.2, 0.2]
        self._use_rotate_nms = True
        self._nms_pre_max_sizes = [1000, 1000, 1000]
        self._nms_post_max_sizes = [1200, 1200, 1200]
        self._nms_iou_thresholds = [0.01, 0.01, 0.01]
        

        self.loc_loss = config_box["loc_loss"]
        self.cls_loss = config_box["cls_loss"]
        self.sparse_shape = [30, 800, 800]

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def forward(self, data):
        
        batch_size = data["batch_size"]

        if "voxel" in data:
            voxel_features = data['voxel']
        else:
            voxel_features = torch.ones((len(data['coordinates']), 1)).float().cuda()
            coors = data['coordinates']
            coors = coors.int()
            x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
            voxel_features = x.dense().squeeze(1)

        x_b = self.basis_block(voxel_features)
        x_1 = self.res_block_1(x_b)
        #bp()
        x_1 = self.se1(x_1)

        x_2 = self.res_block_2(x_1)
        x_2 = self.se2(x_2)

        x_3 = self.res_block_3(x_2)
        x_3 = self.se3(x_3)

        x_4 = self.res_block_4(x_3)
        x_4 = self.se4(x_4)

        x_34 = self.fpn_block_1(x_4, x_3)
        spatial_features = self.fpn_block_2(x_34, x_2)
        
        pred = self.rpn1(spatial_features)
        if self.training == False:
            rr = self.predict1(data, pred)
            return rr

        cls_preds = pred["cls_preds"].contiguous()
        box_preds = pred["box_preds"].contiguous()
        device = cls_preds.get_device()

        B, _, X, Y, _ = box_preds.shape
        heatmap_label = torch.zeros(cls_preds.shape).to(device)
        heatmap_ious_label = torch.zeros(cls_preds.shape).to(device)

        anno_box = torch.zeros((B, self.max_objs, 6)).to(device)
        anno_box_xydxdy = torch.zeros((B, self.max_objs, 4)).to(device)
        index_matrix = torch.zeros((B, self.max_objs), dtype=torch.int64).to(device)
        masks_obj = torch.zeros((B, self.max_objs), dtype=torch.uint8).to(device)
        masks_class = torch.zeros((B, self.max_objs), dtype=torch.uint8).to(device)

        total_length_x = int(((self.detection_range[3] - self.detection_range[0]) / self.voxel_size[0]) / self.out_size_factor)
        total_length_y = int(((self.detection_range[4] - self.detection_range[1]) / self.voxel_size[1]) / self.out_size_factor)

        # print(">>>> Start <<<<")
        for batch_id in range(len(data["gt_boxes"])):
            
            single_gt_box = data["gt_boxes"][batch_id, :, :]
            filter_gt = np.sum(single_gt_box, axis=1)
            single_gt_box = single_gt_box[filter_gt != 0]
            single_gt_class = torch.from_numpy(single_gt_box[:, -1])
            single_gt_bbox = torch.from_numpy(single_gt_box[:,:-1])
            obj_id = 0

            for id_map in range(self.num_classes):
                index_cls = np.where(single_gt_class == (id_map + 1))[0]
                for idxx in index_cls:
                    bbox = single_gt_bbox[idxx]
                    x, y, z = bbox[0], bbox[1], bbox[2]
                    dx, dy = bbox[3], bbox[4]

                    dx = dx / self.voxel_size[0] / self.out_size_factor
                    dy = dy / self.voxel_size[1] / self.out_size_factor

                    coor_x = ((x - self.detection_range[0]) / self.voxel_size[0]) / self.out_size_factor
                    coor_y = ((y - self.detection_range[1]) / self.voxel_size[1]) / self.out_size_factor

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)
                    
                    radius = gaussian_radius((dy, dx), min_overlap=self.gaussian_overlap)
                    radius = max(self.min_radius, int(radius))
                    draw_gaussian = draw_heatmap_gaussian
                    draw_gaussian(heatmap_label[batch_id, 0, :, :, id_map], center_int, radius)

                    xx, yy = center_int[0], center_int[1]

                    obj_index = yy * Y + xx
                    index_matrix[batch_id, obj_id] = obj_index
                    masks_obj[batch_id, obj_id] = 1
                    masks_class[batch_id, obj_id] = id_map

                    rx, ry, rz = bbox[3], bbox[4], bbox[5]
                    rot = bbox[6]
                    anno_box[batch_id, obj_id, 0] = coor_x - xx
                    anno_box[batch_id, obj_id, 1] = coor_y - yy

                    anno_box[batch_id, obj_id, 2] = log(rx)
                    anno_box[batch_id, obj_id, 3] = log(ry)

                    anno_box[batch_id, obj_id, 4] = torch.sin(2 * rot)
                    anno_box[batch_id, obj_id, 5] = torch.cos(2 * rot)


                    #anno_box_xydxdy[batch_id, obj_id, 0] = bbox[0]
                    #anno_box_xydxdy[batch_id, obj_id, 1] = bbox[1]

                    #anno_box_xydxdy[batch_id, obj_id, 2] = bbox[3]
                    #anno_box_xydxdy[batch_id, obj_id, 3] = bbox[4]
                    obj_id += 1

        box_preds = box_preds.view(box_preds.size(0), -1, box_preds.size(-1)).contiguous()
        box_preds = self._gather_feat(box_preds, index_matrix)
        masks_obj = masks_obj.unsqueeze(2).expand_as(anno_box).float()

        loss_dict = {}
        loss_heat_map_1 = self.cls_loss(cls_preds[:, :, :, :, 0].clone(), heatmap_label[:, :, :, :, 0])
        loss_heat_map_2 = self.cls_loss(cls_preds[:, :, :, :, 1].clone(), heatmap_label[:, :, :, :, 1])
        loss_heat_map_3 = self.cls_loss(cls_preds[:, :, :, :, 2].clone(), heatmap_label[:, :, :, :, 2])

        cls_loss = loss_heat_map_1.sum() + loss_heat_map_2.sum() + loss_heat_map_3.sum()
        loc_loss = self.loc_loss(box_preds, anno_box, masks_obj).sum()

        loss = cls_loss + 2 * loc_loss
        loss_dict["loss"] = loss
        loss_dict["_cls_loss"] = cls_loss
        loss_dict["_loc_loss"] = loc_loss
        loss_dict["_iou_loss"] = 0

        return loss_dict


    def predict(self, example, preds_dict):
        batch_size = len(example[list(example)[0]])

        batch_box_preds = preds_dict["box_preds"]
   
        batch_cls_preds = preds_dict["cls_preds"]

        num_class_with_bg = self.num_classes

        predictions_dicts = []

        if time_check == True:
            now_for = time.time()

        nms_func = rotate_nms_gpu

        device = batch_box_preds.get_device()
        self.pre_define_vector = self.pre_define_vector.to(device)
        idd = -1

        for box_preds, cls_preds in zip(batch_box_preds, batch_cls_preds):
            idd += 1
            box_preds = box_preds.float()
            box_preds[0, :, :, 0] += self.pre_define_vector[:, :, 0]
            box_preds[0, :, :, 1] += self.pre_define_vector[:, :, 1]

            box_preds[0, :, :, 0] = box_preds[0, :, :, 0] * self.out_size_factor * self.voxel_size[0] + \
                                    self.detection_range[0]
            box_preds[0, :, :, 1] = box_preds[0, :, :, 1] * self.out_size_factor * self.voxel_size[1] + \
                                    self.detection_range[1]
            box_preds[0, :, :, 2] = torch.exp(box_preds[0, :, :, 2])
            box_preds[0, :, :, 3] = torch.exp(box_preds[0, :, :, 3])
            box_preds[0, :, :, 4] = torch.atan2(box_preds[0, :, :, 4], box_preds[0, :, :, 5])/2
            box_preds = box_preds[:, :, :, :-1]
            box_preds = box_preds.view(-1, 5)
            cls_preds = cls_preds.float()
            cls_preds = cls_preds = cls_preds.view(-1, 3)
            total_scores = torch.sigmoid(cls_preds)

            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            nnn = time.time()
            if self._nms_score_thresholds[0] > 0.0:
                top_scores_keep = top_scores >= self._nms_score_thresholds[0]
                top_scores = top_scores.masked_select(top_scores_keep)

            if time_check == True:
                print("threshold time:", time.time() - nnn)

            if top_scores.shape[0] != 0:
                box_preds = box_preds[top_scores_keep]
                top_labels = top_labels[top_scores_keep]

                boxes_for_nms = box_preds

                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=1200,
                    post_max_size=120,
                    iou_threshold=0.1,
                )

                if time_check == True:
                    print("Time of roate nms gpu:", time.time() - now)
            else:
                selected = []

            if selected is not None:
                selected_boxes = box_preds[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            else:
                selected_boxes, selected_labels, selected_scores = None, None, None

            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores

                if "kernel" not in example:

                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "predict_heatmap": batch_cls_preds[idd]
                    }


            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                        torch.zeros([0, box_preds.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                    "scores":
                        torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
                if "kernel" in example:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "predict_heatmap": batch_cls_preds[idd]
                    }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def predict1(self, example, preds_dict):

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        num_class_with_bg = self.num_classes

        predictions_dicts = []

        nms_func = rotate_nms_gpu
        idd = -1
        for box_preds, cls_preds in zip(batch_box_preds, batch_cls_preds):
            
            idd += 1
            box_preds = box_preds.float()
            box_preds[0, :, :, 0] += self.pre_define_vector[:, :, 0]
            box_preds[0, :, :, 1] += self.pre_define_vector[:, :, 1]

            box_preds[0, :, :, 0] = box_preds[0, :, :, 0] * self.out_size_factor * self.voxel_size[0] + \
                                    self.detection_range[0]
            box_preds[0, :, :, 1] = box_preds[0, :, :, 1] * self.out_size_factor * self.voxel_size[1] + \
                                    self.detection_range[1]
            box_preds[0, :, :, 2] = torch.exp(box_preds[0, :, :, 2])
            box_preds[0, :, :, 3] = torch.exp(box_preds[0, :, :, 3])
            box_preds[0, :, :, 4] = torch.atan2(box_preds[0, :, :, 4], box_preds[0, :, :, 5]) /2
            
            box_preds = box_preds[:, :, :, :-1]
            box_preds = box_preds.view(-1, 5)
            cls_preds = cls_preds.float()
            
            total_scores = torch.sigmoid(cls_preds)
            total_scores = total_scores.view(-1,3).float()


            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            # nnn = time.time()

            all_box_prediction = []
            all_score_prediction = []
            all_label_prediction = []
            
            for classid, (score_threshold, nms_pre, nms_post, iou_thres) in enumerate(
                    zip(self._nms_score_thresholds, self._nms_pre_max_sizes, self._nms_post_max_sizes,
                        self._nms_iou_thresholds)):
                
                labels_keep = (top_labels == classid)
                top_scores_class = top_scores.masked_select(labels_keep)
                top_labels_class = top_labels.masked_select(labels_keep)

                box_preds_class = box_preds[labels_keep]

                top_scores_keep = top_scores_class >= score_threshold
                top_scores_class = top_scores_class.masked_select(top_scores_keep)
                top_labels_class = top_labels_class.masked_select(top_scores_keep)

                box_preds_class = box_preds_class[top_scores_keep]

                if classid == 0:
                    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 4.5
                    box_preds_class = box_preds_class[area_filter]
                    top_scores_class = top_scores_class.masked_select(area_filter)
                    top_labels_class = top_labels_class.masked_select(area_filter)

                    # area_filter_big = (box_preds_class[:,2] * box_preds_class[:,3]) > 10
                    # area_filter_big = area_filter_big.float()
                    # area_filter_big *= 0.1
                    # top_scores_class = top_scores_class + area_filter_big

                if classid == 2:
                    area_filter = (box_preds_class[:, 2] * box_preds_class[:, 3]) > 1
                    box_preds_class = box_preds_class[area_filter]
                    top_scores_class = top_scores_class.masked_select(area_filter)
                    top_labels_class = top_labels_class.masked_select(area_filter)

                if box_preds_class.shape[0] != 0:
                    selected = rotate_nms_cc(
                        box_preds_class.float().cpu().numpy(),
                        top_scores_class.float().cpu().numpy(),
                        iou_thres
                    )
                else:
                    selected = []

                if len(selected) > 0:
                    selected_boxes = box_preds_class[selected]
                    selected_top_scores_class = top_scores_class[selected]
                    selected_top_labels_class = top_labels_class[selected]
                else:
                    selected_boxes = box_preds_class
                    selected_top_scores_class = top_scores_class
                    selected_top_labels_class = top_labels_class

                all_box_prediction.append(selected_boxes)
                all_score_prediction.append(selected_top_scores_class)
                all_label_prediction.append(selected_top_labels_class)



            all_box_prediction = torch.cat(all_box_prediction, dim=0)
            all_score_prediction = torch.cat(all_score_prediction, dim=0)
            all_label_prediction = torch.cat(all_label_prediction, dim=0)
            

            if len(all_box_prediction) > 0:
                if "kernel" not in example:

                    predictions_dict = {
                        "box3d_lidar": all_box_prediction,
                        "scores": all_score_prediction,
                        "label_preds": all_label_prediction,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": all_box_prediction,
                        "scores": all_score_prediction,
                        "label_preds": all_label_prediction,
                        "predict_heatmap": batch_cls_preds[idd]
                    }

            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                        torch.zeros([0, box_preds.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                    "scores":
                        torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

