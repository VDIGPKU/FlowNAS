# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# adapted from OFA: https://github.com/mit-han-lab/once-for-all


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

from .static_layers import set_layer_from_config, MBInvertedConvLayer, ConvBnActLayer, ShortcutLayer, LinearLayer, MobileInvertedResidualBlock, IdentityLayer
from .nn_utils import  make_divisible
from .nn_base import MyNetwork

class AttentiveNasStaticModel(MyNetwork):

    def __init__(self, first_conv, blocks, last_conv):
        super(AttentiveNasStaticModel, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.last_conv = last_conv

    def forward(self, x):
        # resize input to target resolution first
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        #_str += self.last_conv.module_str + '\n'
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
            #'last_conv': self.last_conv.config,
        }


    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.InstanceNorm2d):
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()



