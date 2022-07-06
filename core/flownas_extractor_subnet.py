import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random
import collections
import math
from modules.static_layers import MobileInvertedResidualBlock, MBInvertedConvLayer, ShortcutLayer
from modules.nn_utils import make_divisible, int2list
# from .modules.nn_base import MyNetwork
from modules.attentive_nas_static_model import AttentiveNasStaticModel

class BasicEncoder(nn.Module):
    def __init__(self, arch_info, resolution, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.first_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.output_dim = 128
        self.block_out_channels = [64, 64, 96, 96, 128, 128]
        self.block_stride = [1, 1, 2, 1, 2, 1]
        self.features = torch.nn.ModuleList()

        # 6 block
        self.resolution = resolution
        self.width_list = arch_info['width']
        self.depth_list = arch_info['depth']
        self.ks_list = arch_info['kernel_size']
        self.expand_ratio_list = arch_info['expand_ratio']
        self.stride_list = arch_info['stride']
        self.cfg_candidates = arch_info

        self.block_group_info = []
        blocks = []
        _block_index = 0
        feature_dim = 64
        for stage_id in range(len(self.stride_list)):
            width = self.width_list[stage_id]
            n_block = self.depth_list[stage_id]
            ks = self.ks_list[stage_id]
            expand_ratio_list = self.expand_ratio_list[stage_id]
            # print(width, stage_id)
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = self.stride_list[stage_id] if i == 0 else 1
                # if min(expand_ratio_list) >= 4:
                #     expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                mobile_inverted_conv = MBInvertedConvLayer(in_channels=feature_dim,
                                                           out_channels=output_channel,
                                                           kernel_size=ks,
                                                           stride=stride,
                                                           expand_ratio=expand_ratio_list,
                                                           norm=norm_fn)

                shortcut = ShortcutLayer(feature_dim, output_channel, reduction=stride)
                blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks = nn.ModuleList(blocks)

        self.last_conv = nn.Conv2d(feature_dim, output_dim, 1, 1, 0, bias=False)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        bn_param = (0., 1e-5)
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]
        # self.zero_residual_block_bn_weights()

    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, MobileInvertedResidualBlock):
                    if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and m.shortcut is not None:
                        if m.mobile_inverted_conv.point_linear.bn.bn.weight is not None:
                            m.mobile_inverted_conv.point_linear.bn.bn.weight.zero_()


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.first_conv(x)
        x = self.norm1(x)
        x = self.relu1(x)

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.last_conv(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        _str += self.last_conv.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': AttentiveNasStaticModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'last_conv': self.last_conv.config,
        }

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                            nn.SyncBatchNorm) or isinstance(
                    m, nn.InstanceNorm2d):
                if momentum is not None:
                    m.momentum = float(momentum)
                else:
                    m.momentum = None
                m.eps = float(eps)
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                            nn.SyncBatchNorm) or isinstance(
                    m, nn.InstanceNorm2d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def compute_active_subnet_flops(self):

        def count_conv(c_in, c_out, size_out, groups, k):
            kernel_ops = k ** 2
            output_elements = c_out * size_out
            ops = c_in * output_elements * kernel_ops / groups
            return ops

        def count_linear(c_in, c_out):
            return c_in * c_out

        total_ops = 0

        c_in = 3
        # size_out = 468 // self.first_conv.stride
        size_out = self.resolution[0] * self.resolution[1]
        size_out = size_out // self.first_conv.stride[0]
        size_out = size_out // self.first_conv.stride[0]

        c_out = self.in_planes

        total_ops += count_conv(c_in, c_out, size_out, 1, 3)
        c_in = c_out

        # mb blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                block = self.blocks[idx]
                c_middle = make_divisible(round(c_in * block.mobile_inverted_conv.active_expand_ratio), 8)
                # 1*1 conv
                if block.mobile_inverted_conv.inverted_bottleneck is not None:
                    total_ops += count_conv(c_in, c_middle, size_out, 1, 1)
                # dw conv
                stride = 1 if idx > active_idx[0] else block.mobile_inverted_conv.stride
                if size_out % stride == 0:
                    size_out = size_out // stride
                else:
                    size_out = (size_out + 1) // stride
                total_ops += count_conv(c_middle, c_middle, size_out, c_middle,
                                        block.mobile_inverted_conv.active_kernel_size)
                # 1*1 conv
                c_out = block.mobile_inverted_conv.active_out_channel
                total_ops += count_conv(c_middle, c_out, size_out, 1, 1)
                # se
                if block.mobile_inverted_conv.use_se:
                    num_mid = make_divisible(c_middle // block.mobile_inverted_conv.depth_conv.se.reduction, divisor=8)
                    total_ops += count_conv(c_middle, num_mid, 1, 1, 1) * 2
                if block.shortcut and c_in != c_out:
                    total_ops += count_conv(c_in, c_out, size_out, 1, 1)
                c_in = c_out

        # c_out = self.last_conv.out_channel
        c_out = self.output_dim
        total_ops += count_conv(c_in, c_out, size_out, 1, 1)

        return total_ops / 1e6

    def load_weights_from_pretrained_models(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        assert isinstance(checkpoint, dict)
        pretrained_state_dicts = checkpoint['state_dict']
        for k, v in self.state_dict().items():
            name = 'module.' + k if not k.startswith('module') else k
            v.copy_(pretrained_state_dicts[name])
