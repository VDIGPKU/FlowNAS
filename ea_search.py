import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

# from attentive_nas_raft_supernet import RAFT
# from attentive_nas_raft_supernet_w_estimator import RAFT_SuperNet
from flownas_supernet import FlowNAS as RAFT_SuperNet
from flownas_subnet import FlowNASSubnet as subRAFT
# from attentive_nas_raft_supernet import RAFT
# from attentive_nas_raft_subnet import RAFT as subRAFT
from arch_sampler import ArchSampler, sample_helper
# from operations import *
from raft_teacher import RAFT

import logging
import evaluate_supernet

from utils.utils import InputPadder, forward_interpolate
sys.setrecursionlimit(10000)
import functools
import random
import torch.utils.data as data
from tqdm import tqdm
from random import choice
from utils.utils import  ChannelPool
from einops import rearrange, reduce

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

# choice = lambda x: x[np.random.randint(len(x)-1)] if isinstance(
#     x, tuple) else choice(tuple(x))

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 1
VAL_FREQ = 10000
ARCH_INFO = {
    'width': [ [56, 64],
               [64, 72],
               [88, 96],
               [96, 104, 112],
               [112, 120, 128],
               [128, 136]],

    'depth':  [ [1,2],
                 [1,2,3],
                 [1,2,3],
                 [1,2,3],
                 [2,3,4],
                 [1,2],],

    'kernel_size':  [ [3,5],
                      [3,5],
                      [3,5],
                      [3,5],
                      [3,5],
                      [3,5],],

    'expand_ratio' : [[1],
                      [1,2,4],
                      [4,5,6],
                      [4,5,6],
                      [6],
                      [6]],

    'stride' : [1, 1, 2, 1, 2, 1]
}


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

def max_pooling_layer(x, type='max'):
    return reduce(x, 'b c h w -> b 1 h w', type)
def dist(flow_preds, flow_gt, padder=None):
    n_predictions = len(flow_preds)
    dist_sum=0
    # for i in range(n_predictions):
    #     # print(flow_gt[i].shape, flow_preds[i].shape)
    #     flow_gt[i] = max_pooling_layer(flow_gt[i])
    #     flow_preds[i] = max_pooling_layer(flow_preds[i])
    #     print(flow_gt[i].shape, flow_preds[i].shape, i, 'predictn')
    #     epe = torch.sum((flow_preds[i] - flow_gt[i]) ** 2, dim=0).sqrt()
    #     dist_sum += epe

    flow_gt = max_pooling_layer(flow_gt[-1])
    flow_preds = max_pooling_layer(flow_preds[-1])
    flow_gt = padder.unpad(flow_gt).cpu()
    flow_preds = padder.unpad(flow_preds).cpu()

    epe = torch.sum((flow_preds - flow_gt) ** 2, dim=0).sqrt()
    epe = epe.view(-1).numpy()
    dist_sum += epe

    return dist_sum
@no_grad_wrapper
def get_cand_err(model, teacher=None, architecture=None, dataset_type=None, iters=12, split='val'):
    print(architecture)
    model.module.set_active_subnet(**architecture)
    model.eval()
    if 'kitti' in dataset_type:
        output = validate_kitti(model.module, teacher.module, split=split, iters=iters)
    else:
        if 'clean' in dataset_type:
            output = validate_sintel_multi(model, teacher, iters=iters, dataset_type='clean', split=split)
        elif 'final' in dataset_type:
            output = validate_sintel_multi(model, teacher, iters=iters, dataset_type='final', split=split)
        else:
            print('no such dataset {}'.format(dataset_type))
            raise NotImplementedError
    return output

@torch.no_grad()
def validate_sintel_multi(model, teacher, iters=32, architecture=None, split='val', dataset_type='clean', num_batch=8):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    teacher.eval()
    results = {}

    dtypes = []
    if dataset_type == 'clean':
        dtypes.append('clean')

    if 'final' in dataset_type:
        dtypes.append('final')
    
    count = 0


    for dstype in dtypes:
        val_dataset = datasets.MyMpiSintel(split=split, dstype=dstype, flownet_split=True)
        epe_list = []
        feat_dist_list = []

        # print(len(val_dataset))
        epe_list = []
        val_loader = data.DataLoader(val_dataset, batch_size=num_batch, pin_memory=False, shuffle=False, num_workers=4, drop_last=False)
        # for val_id in tqdm(range(len(val_dataset))):
        num_iter = len(val_dataset)//num_batch
        for image1, image2, flow_gt, _ in tqdm(val_loader):
                
            image1 = image1.cuda()
            image2 = image2.cuda()
            flow_gt = flow_gt.cuda()
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            count += 1
            if count<num_iter:
                flow_low, flow_pr, feats_f, feats_c = model(image1, image2, iters=iters, test_mode=True, return_feats=True)
                flow_low_g, flow_pr_g, feats_f_g, feats_c_g = teacher(image1, image2, iters=iters, test_mode=True, return_feats=True)
                # epe = validate_sintel_multi_gpu_forward(model, image1, image2, flow_gt, iters, inner_loop=inner_loop)
            else:
                flow_low, flow_pr, feats_f, feats_c = model.module(image1, image2, iters=iters, test_mode=True, return_feats=True)
                flow_low_g, flow_pr_g, feats_f_g, feats_c_g = teacher.module(image1, image2, iters=iters, test_mode=True, return_feats=True)
                # epe = validate_sintel_multi_gpu_forward(model.module, image1, image2, flow_gt, iters, inner_loop=inner_loop)
            feat_dist_f = dist(feats_f, feats_f_g, padder)
            feat_dist_c = dist(feats_c, feats_c_g, padder)
            feat_dist = feat_dist_f + feat_dist_c
            feat_dist_list.append(feat_dist/image1.size(0))
            
            flow = padder.unpad(flow_pr)

            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            # epe = epe.view(-1).numpy()

            epe_list.append(epe.view(-1).cpu().numpy())


        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))

        results[dstype] = epe
        results['feat_dist'] = np.mean(feat_dist_list)

    return results

@torch.no_grad()
def validate_kitti(model, teacher, iters=24, split='val'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()

    val_dataset = datasets.MyKITTI(split=split)

    out_list, epe_list = [], []
    feat_dist_list = []

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr, feats_f, feats_c = model(image1, image2, iters=iters, test_mode=True, return_feats=True)

        with torch.no_grad():
            flow_low_g, flow_pr_g, feats_f_g, feats_c_g = teacher(image1, image2, iters=iters, test_mode=True, return_feats=True)
        feat_dist_f = dist(feats_f, feats_f_g, padder)
        feat_dist_c = dist(feats_c, feats_c_g, padder)
        feat_dist = feat_dist_f + feat_dist_c
        feat_dist_list.append(feat_dist)
        
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()

        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    feat_dist = np.mean(np.concatenate(feat_dist_list))

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti_f1': f1, 'feat_dist': feat_dist}
    # return {'kitti-epe': epe}

def cand2arch(cand):
    arch = {}
    cnt = 0
    for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
        arch[k] = []
        for idx in range(len(ARCH_INFO[k])):
            # print(idx+cnt, k)
            arch[k].append(cand[idx + cnt])
        cnt += len(ARCH_INFO[k])
    return arch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1000./1000.

def cal_param(cand):
    arch = cand2arch(cand)
    arch['stride'] = [1, 1, 2, 1, 2, 1]
    model = subRAFT(args, arch, args.image_size)
    param = count_parameters(model)
    del model
    return param


def get_random_cand_with_constrain(params):
    # arch = {}
    # for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
    #     arch[k] = []
    #     for idx in range(len(ARCH_INFO[k])):
    #         arch[k].append(sample_helper(ARCH_INFO[k][idx]))
    # return tuple(arch)
    cand = []
    for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
        for idx in range(len(ARCH_INFO[k])):
            cand.append(sample_helper(ARCH_INFO[k][idx]))
    return tuple(cand)


class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.topk = args.topk
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit

        self.name = args.name

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], self.topk: []}
        self.epoch = 0
        self.candidates = []
        # self.nr_state = len(PRIMITIVES_MB)

        self.model = torch.nn.DataParallel(RAFT_SuperNet(args, ARCH_INFO, args.image_size))
        # self.model.load_weights_from_pretrained_models(args.model)
        # self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus)
        self.model.load_state_dict(torch.load(args.model))
        self.model.cuda()
        self.model.eval()

        self.teacher = torch.nn.DataParallel(RAFT(args))
        self.teacher.load_state_dict(torch.load(args.teacher_restore_ckpt), strict=False)
        self.teacher.cuda()
        self.teacher.eval()

        self.arch_sampler = ArchSampler(arch_info=ARCH_INFO, discretize_step=25.0, model=self.model)
        times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.val_set = args.dataset
        model_path = os.path.split(args.model)[0]
        
        # self.logfile = os.path.join('ea_log/ea_kd_{}_{}.log'.format(self.val_set, times))
        self.logfile = os.path.join('ea_log/ea_{}_{}_{}.log'.format(self.name, self.val_set, times))

        self.not_use_diss = args.not_use_diss
        self.spilt = args.split
        self.iters = args.iters

    def is_legal(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        params = cal_param(cand)
        info['params'] = params
        if params>5.0:
            info['visited'] = True
            return False
        print('evaluation', cand)
        arch = cand2arch(cand)
        print('evaluation', cand, arch)
        output = get_cand_err(self.model, self.teacher, arch, dataset_type=self.val_set, split=self.spilt, iters=self.iters)
        if output:
            if 'kitti' in self.val_set:
                clean = output['kitti_f1']
            else:
                if 'clean' in self.val_set:
                    clean = output['clean']
                else:
                    clean = output['final']
            print(output)
            info['feat_dist'] = output['feat_dist']
            info['epe'] = clean
            if self.not_use_diss:
                info['clean'] = clean
            else:
                info['clean'] = clean + output['feat_dist']
            info['visited'] = True
            print(info)
            return clean
        return False

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        for tt in t:
            print(tt, key(tt))
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                # print(cand)
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
            for cand in cands:
                yield cand

    def get_random(self, num):
        print('random select ........')


        def get_random_cand():
            # arch = {}
            # for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
            #     arch[k] = []
            #     for idx in range(len(ARCH_INFO[k])):
            #         arch[k].append(sample_helper(ARCH_INFO[k][idx]))
            # return tuple(arch)
            cand = []
            for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
                for idx in range(len(ARCH_INFO[k])):
                    cand.append(sample_helper(ARCH_INFO[k][idx]))
            return tuple(cand)

        cand_iter = self.stack_random_cand(get_random_cand)

        while len(self.candidates) < num:
            cand = next(cand_iter)
            # print('random', cand)
            acc = self.is_legal(cand)
            if not acc:
                continue

            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
            logging.info('random {}/{}, arch:{}, {}, params:{}, error:{}, dist:{}, epe:{}'.format(len(self.candidates), num, cand, cand2arch(cand), self.vis_dict[cand]['params'], self.vis_dict[cand]['clean'], self.vis_dict[cand]['feat_dist'], self.vis_dict[cand]['epe']))

        print('random_num = {}'.format(len(self.candidates)))
        print('random', cand_iter)

    def val_one_arch(self, arch):
        # print(arch)
        # output = get_cand_err(self.model, arch, self.val_set)
        # output = get_cand_err(self.model, self.teacher, arch, dataset_type=self.val_set, iters=self.iters)
        output = get_cand_err(self.model, self.teacher, arch, dataset_type=self.val_set, split=self.spilt, iters=self.iters)
        if output:
            print(output)
        # print(clean)


    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            print(len(cand))
            cnt = 0
            for key in ['width', 'kernel_size', 'depth', 'expand_ratio']:
                for idx in range(len(ARCH_INFO[key])):
                    if np.random.random_sample() < m_prob:
                        cand[idx + cnt] = sample_helper(ARCH_INFO[key][idx])
                cnt += len(ARCH_INFO[key])

            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)

            print('mutation',  cand)

            acc = self.is_legal(cand)
            if not acc:
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))
            logging.info('mutation {}/{}, arch:{}, {}, param:{}, error:{}, dist:{}, epe:{}'.format(len(res), mutation_num, cand, cand2arch(cand), self.vis_dict[cand]['params'], self.vis_dict[cand]['clean'], self.vis_dict[cand]['feat_dist'], self.vis_dict[cand]['epe']))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            print(p1, p2)
            cand = []
            for i in range(len(p1)):
                rand = np.random.randint(2)
                if rand:
                    cand.append(p2[i])
                else:
                    cand.append(p1[i])
            cand = tuple(cand)
           
            return cand

        cand_iter = self.stack_random_cand(random_func)
        print('crossover', cand_iter)

        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)

            acc = self.is_legal(cand)

            if not acc:
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))
            logging.info('crossover {}/{}, arch:{}, {}, params:{}, error:{}, dist:{}, epe:{}'.format(len(res), crossover_num, cand, cand2arch(cand), self.vis_dict[cand]['params'],self.vis_dict[cand]['clean'], self.vis_dict[cand]['feat_dist'], self.vis_dict[cand]['epe']))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):

        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        print('logghghhhhhhhhhh')
        logging.basicConfig(filename=self.logfile, level=logging.INFO)
        print(self.logfile)
        logging.info(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))
            # record = open('ea32.txt', 'a+')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['clean'])
            self.update_top_k(
                self.candidates, k=self.topk, key=lambda x: self.vis_dict[x]['clean'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.topk])))
            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.topk])))
            for i, cand in enumerate(self.keep_top_k[self.topk]):
                print('No.{} {} error: {}'.format(
                    i + 1, cand, self.vis_dict[cand]['clean']))
                logging.info('No.{}, arch:{}, {}, params:{}, error:{}, dist:{}, epe:{}'.format(i + 1, cand, cand2arch(cand), self.vis_dict[cand]['params'],self.vis_dict[cand]['clean'], self.vis_dict[cand]['feat_dist'], self.vis_dict[cand]['epe']))

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        print('result------')
        info = self.vis_dict
        cands = sorted([cand for cand in info if 'clean' in info[cand]],
                       key=lambda cand: info[cand]['clean'])
        for cand in cands:
            print(cand, info[cand]['clean'])
        opt = cands[-1]
        logging.info('result----- {}'.format(opt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation", type=str)
    parser.add_argument('--split', default='val', help="dataset split", type=str)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--teacher_restore_ckpt', help="restore checkpoint", default='./models/raft-chairs.pth')
    parser.add_argument('--iters', type=int, default=12)

    parser.add_argument('--name', type=str, default='multi')

    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])

    parser.add_argument('--not_use_diss', default=False, action='store_true',
                        help='only use position-wise attention')


    args = parser.parse_args()

    if not os.path.isdir('ea_log'):
        os.mkdir('ea_log')

    with torch.no_grad():
        searcher = EvolutionSearcher(args)
        searcher.search()

