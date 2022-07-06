# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import atexit
import os
import random
import copy

def round_flops(flops, step):
    return int(round(flops / step) * step)


def sample_helper(m):
    return random.choices(m)[0]

class ArchSampler():
    def __init__(self, arch_info, discretize_step=None, model=None):
        super(ArchSampler, self).__init__()

        self.arch_info = arch_info
        self.discretize_step = discretize_step
        self.model = model.module

    def sample_archs_according_to_flops(self, n_samples=1, max_trials=100, return_flops=True, return_trials=False):
        archs = []
        arch = {}
        target_flops = 0
        # print(self.arch_info)
        for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
            arch[k] = []
            for idx in range(len(self.arch_info[k])):
                arch[k].append(sample_helper(self.arch_info[k][idx]))
        if self.model:
            self.model.set_active_subnet(**arch)
            flops = self.model.compute_active_subnet_flops()
            if return_flops:
                arch['flops'] = flops
            target_flops = round_flops(flops, self.discretize_step)
        else:
            raise NotImplementedError

        archs.append(arch)

        while len(archs) < n_samples - 1:
            for _trial in range(max_trials+1):
                arch = {}
                for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
                    arch[k] = []
                    for idx in range(len(self.arch_info[k])):
                        arch[k].append(sample_helper(self.arch_info[k][idx]))
                if self.model:
                    self.model.set_active_subnet(**arch)
                    flops = self.model.compute_active_subnet_flops()
                    if return_flops:
                        arch['flops'] = flops
                    if round_flops(flops, self.discretize_step) == target_flops:
                        break
                else:
                    raise NotImplementedError
            #accepte the sample anyway
            archs.append(arch)
        return archs