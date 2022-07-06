import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random
import collections
import math
from modules.dynamic_layers import DynamicMBConvLayer, DynamicConvBnActLayer, DynamicShortcutLayer, DynamicPointLayer
from modules.static_layers import MobileInvertedResidualBlock
from modules.nn_utils import make_divisible, int2list
# from .modules.nn_base import MyNetwork
from modules.attentive_nas_static_model import AttentiveNasStaticModel
from utils.utils import  ChannelPool


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

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
        self.depth_list =  arch_info['depth']
        self.ks_list = arch_info['kernel_size']
        self.expand_ratio_list = arch_info['expand_ratio']
        self.stride_list = arch_info['stride']
        self.cfg_candidates = arch_info

        self.block_group_info = []
        blocks = []
        _block_index = 0
        feature_dim = [64]
        for stage_id in range(len(self.stride_list)):
            width = self.width_list[stage_id]
            n_block = max(self.depth_list[stage_id])
            ks = self.ks_list[stage_id]
            expand_ratio_list = self.expand_ratio_list[stage_id]

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block
            
            output_channel = width
            for i in range(n_block):
                stride = self.stride_list[stage_id] if i == 0 else 1
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim, 
                    out_channel_list=output_channel, 
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list, 
                    stride=stride,
                    norm=norm_fn
                )
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=stride)
                blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks = nn.ModuleList(blocks)

        self.last_conv = DynamicPointLayer(
            in_channel_list=feature_dim, out_channel_list=output_dim, bias=False)

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
        self.zero_residual_block_bn_weights()
        self.pool = ChannelPool(1)

    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, MobileInvertedResidualBlock):
                    if isinstance(m.mobile_inverted_conv, DynamicMBConvLayer) and m.shortcut is not None:
                        if m.mobile_inverted_conv.point_linear.bn.bn.weight is not None:
                            m.mobile_inverted_conv.point_linear.bn.bn.weight.zero_()

    # def _make_layer(self, dim, stride=1):
    #     layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
    #     layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
    #     layers = (layer1, layer2)
        
    #     self.in_planes = dim
    #     return nn.Sequential(*layers)


    # def forward(self, x, architecture):

    #     # if input is list, combine batch dimension
    #     is_list = isinstance(x, tuple) or isinstance(x, list)
    #     if is_list:
    #         batch_dim = x[0].shape[0]
    #         x = torch.cat(x, dim=0)
    #     if len(architecture) != len(self.features):
    #         print(architecture)
    #     x = self.conv1(x)
    #     x = self.norm1(x)
    #     x = self.relu1(x)

    #     # x = self.layer1(x)
    #     # x = self.layer2(x)
    #     # x = self.layer3(x)
    #     for archs, arch_id in zip(self.features, architecture):
    #         x = archs(x, arch_id)

    #     x = self.conv2(x)

    #     if self.training and self.dropout is not None:
    #         x = self.dropout(x)

    #     if is_list:
    #         x = torch.split(x, [batch_dim, batch_dim], dim=0)

    #     return x
    def forward(self, x, return_feats=False):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.first_conv(x)
        x = self.norm1(x)
        x = self.relu1(x)
        feats = []
        feats.append(x)
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)
            if  stage_id+1 < len(self.stride_list) and self.stride_list[stage_id+1]!=1:
                feats.append(x)
                # print(x.shape, stage_id, 'stage')
        feats.append(x)
        x = self.last_conv(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        feats.append(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        if return_feats:
            return tuple([x, feats])
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
            'name': AttentiveNasDynamicModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'last_conv': self.last_conv.config,
        }

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.InstanceNorm2d):
                if momentum is not None:
                    m.momentum = float(momentum)
                else:
                    m.momentum = None
                m.eps = float(eps)
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.InstanceNorm2d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def set_active_subnet(self, width=None, depth=None, kernel_size=None, expand_ratio=None, **kwargs):
        assert len(depth) == len(kernel_size) == len(expand_ratio) == len(width)
        # first conv
        # self.first_conv.active_out_channel = width[0]

        for stage_id, (c, k, e, d) in enumerate(zip(width, kernel_size, expand_ratio, depth)):
            start_idx, end_idx = min(self.block_group_info[stage_id]), max(self.block_group_info[stage_id])
            for block_id in range(start_idx, start_idx+d):
                block = self.blocks[block_id]
                #block output channels
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c

                #dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k

                #dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e

        #IRBlocks repated times
        for i, d in enumerate(depth):
            self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        #last conv
        # self.last_conv.active_out_channel = width[-1]

    def get_active_subnet_settings(self):
        width, depth, kernel_size, expand_ratio= [], [], [],  []

        #first conv
        # width.append(self.first_conv.active_out_channel)
        for stage_id in range(len(self.block_group_info)):
            start_idx = min(self.block_group_info[stage_id])
            block = self.blocks[start_idx]  #first block
            width.append(block.mobile_inverted_conv.active_out_channel)
            kernel_size.append(block.mobile_inverted_conv.active_kernel_size)
            expand_ratio.append(block.mobile_inverted_conv.active_expand_ratio)
            depth.append(self.runtime_depth[stage_id])

        return {
            'width': width,
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'depth': depth,
        }

    def sample_min_subnet(self):
        return self._sample_active_subnet(min_net=True)


    def sample_max_subnet(self):
        return self._sample_active_subnet(max_net=True)


    def sample_active_subnet(self, compute_flops=False, min_net=False, max_net=False):
        cfg = self._sample_active_subnet(
            min_net, max_net
        ) 
        if compute_flops:
            cfg['flops'] = self.compute_active_subnet_flops()
        return cfg


    def sample_active_subnet_within_range(self, targeted_min_flops, targeted_max_flops):
        while True:
            cfg = self._sample_active_subnet() 
            cfg['flops'] = self.compute_active_subnet_flops()
            if cfg['flops'] >= targeted_min_flops and cfg['flops'] <= targeted_max_flops:
                return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False):

        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))

        cfg = {}
        # sample a resolution
        # cfg['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = []
            for vv in self.cfg_candidates[k]:
                cfg[k].append(sample_cfg(int2list(vv), min_net, max_net))

        self.set_active_subnet(
            cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg


    def mutate_and_reset(self, cfg, prob=0.1, keep_resolution=False):
        cfg = copy.deepcopy(cfg)
        pick_another = lambda x, candidates: x if len(candidates) == 1 else random.choice([v for v in candidates if v != x])
        # sample a resolution
        r = random.random()
        # sample channels, depth, kernel_size, expand_ratio
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            for _i, _v in enumerate(cfg[k]):
                r = random.random()
                if r < prob:
                    cfg[k][_i] = pick_another(cfg[k][_i], int2list(self.cfg_candidates[k][_i]))

        self.set_active_subnet(
            cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg


    def crossover_and_reset(self, cfg1, cfg2, p=0.5):
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if random.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        cfg = {}
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = _cross_helper(cfg1[k], cfg2[k], p)

        self.set_active_subnet(
            cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg


    def get_active_subnet(self, preserve_weight=True):
        with torch.no_grad():

            blocks = []
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                        self.blocks[idx].shortcut.get_active_subnet(input_channel, preserve_weight) if self.blocks[idx].shortcut is not None else None
                    ))
                    input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks += stage_blocks

            _subnet = AttentiveNasStaticModel(blocks)
            _subnet.set_bn_param(**self.get_bn_param())
            return _subnet


    def get_active_net_config(self):
        raise NotImplementedError


    def compute_active_subnet_flops(self):

        def count_conv(c_in, c_out, size_out, groups, k):
            kernel_ops = k**2
            output_elements = c_out * size_out
            ops = c_in * output_elements * kernel_ops / groups
            return ops

        def count_linear(c_in, c_out):
            return c_in * c_out

        total_ops = 0

        c_in = 3
        # size_out = 468 // self.first_conv.stride
        size_out = self.resolution[0]*self.resolution[1]
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
                    size_out = (size_out +1) // stride
                total_ops += count_conv(c_middle, c_middle, size_out, c_middle, block.mobile_inverted_conv.active_kernel_size)
                # 1*1 conv
                c_out = block.mobile_inverted_conv.active_out_channel
                total_ops += count_conv(c_middle, c_out, size_out, 1, 1)
                #se
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



class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
