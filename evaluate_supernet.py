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

# from raft import RAFT
from flownas_supernet import FlowNAS

from utils.utils import InputPadder, forward_interpolate
from tqdm import tqdm
import torch.utils.data as data

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

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        # for test_id in range(len(test_dataset)):
        for test_id in tqdm(range(len(test_dataset))):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in tqdm(range(len(test_dataset))):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, architecture=None, is_val=False):
    """ Peform validation using the Sintel (train) split """
    print(architecture)
    model.eval()
    results = {}
    # for dstype in ['albedo','clean', 'final']:
    for dstype in ['clean', 'final']:
        if not is_val:
            val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        else:
            val_dataset = datasets.MyMpiSintel(split='validation', dstype=dstype, flownet_split=True)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            if architecture is None:
                flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            else:
                flow_low, flow_pr = model(image1, image2, iters=iters, architecture=architecture,test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()
            # print(flow.shape)
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        # print(epe_all.shape)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_multi_gpu_forward(model, image1, image2, flow_gt, iters=32, architecture=None):
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    if architecture is None:
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
    else:
        flow_low, flow_pr = model(image1, image2, iters=iters, architecture=architecture,test_mode=True)
    flow = padder.unpad(flow_pr)

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()

    return epe

@torch.no_grad()
def validate_sintel_multi_gpu(model, iters=32, architecture=None, num_batch=8):
    """ Peform validation using the Sintel (train) split """
    print(architecture)
    model.eval()
    results = {}
    # for dstype in ['clean', 'final']:
    count = 0
    # max_count = 10
    for dstype in ['clean']:
        # val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        val_dataset = datasets.MyMpiSintel(split='validation', dstype=dstype)

        # print(len(val_dataset))
        epe_list = []
        val_loader = data.DataLoader(val_dataset, batch_size=num_batch, 
        pin_memory=False, shuffle=False, num_workers=4, drop_last=False)
        # for val_id in tqdm(range(len(val_dataset))):
        num_iter = len(val_dataset)//num_batch
        for image1, image2, flow_gt, _ in tqdm(val_loader):
                
            image1 = image1.cuda()
            image2 = image2.cuda()
            flow_gt = flow_gt.cuda()

            count += 1
            if count<num_iter:
                epe = validate_sintel_multi_gpu_forward(model, image1, image2, flow_gt, iters, architecture)
            else:
                epe = validate_sintel_multi_gpu_forward(model.module, image1, image2, flow_gt, iters, architecture)

            epe_list.append(epe.view(-1).cpu().numpy())


        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))

        results[dstype] = epe

    return results

@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    # val_dataset = datasets.MyKITTI(split='validation')

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
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

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    args = parser.parse_args()


    # model = torch.nn.DataParallel(RAFT(args))
    model = torch.nn.DataParallel(FlowNAS(args, ARCH_INFO, args.image_size), device_ids=args.gpus)
    model.load_state_dict(torch.load(args.model),strict=True)

    model.cuda()
    model.eval()
    architecture = dict()
    # FlowNAS-RAFT-S
    architecture['sintel'] = {
        'width': [64, 72, 96, 104, 120, 136], 
        'kernel_size': [3, 3, 5, 3, 5, 5], 
        'depth': [2, 2, 2, 1, 2, 1], 
        'expand_ratio': [1, 2, 5, 5, 6, 6],
        'stride': [1, 1, 2, 1, 2, 1]
    }

    # FlowNAS-RAFT-K
    architecture['kitti'] = {
        'width': [64, 72, 88, 104, 120, 136], 
        'kernel_size': [3, 5, 3, 5, 5, 5], 
        'depth': [2, 1, 1, 2, 2, 1], 
        'expand_ratio': [1, 4, 6, 5, 6, 6],
        'stride': [1, 1, 2, 1, 2, 1]
    }

    model.module.set_active_subnet(**architecture[args.dataset])

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)
            create_sintel_submission(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)
            create_kitti_submission(model.module)


