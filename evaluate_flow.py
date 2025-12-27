import copy
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

import RAFT.datasets as datasets

from RAFT.utils import flow_viz
from RAFT.utils import frame_utils

from RAFT.raft import RAFT
from RAFT.utils.utils import InputPadder, forward_interpolate


FLOW_MAX = -1


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        # image2 = image1

        _, flow_pr = model(image1, image1, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_dsec(model, iters=24):
    """ Perform validation using the DSEC_RAFT dataset """
    model.eval()
    val_dataset = datasets.DSECRAFT(split='train')

    out_list, epe_list, angle_list = [], [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        # Direction (angular) error
        flow_flat = flow.view(2, -1)[:, val].numpy()
        flow_gt_flat = flow_gt.view(2, -1)[:, val].numpy()
        dot = np.sum(flow_flat * flow_gt_flat, axis=0)
        norm_pred = np.linalg.norm(flow_flat, axis=0)
        norm_gt = np.linalg.norm(flow_gt_flat, axis=0)
        cos_angle = dot / (norm_pred * norm_gt + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        angle_list.append(angle)

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    angle_list = np.concatenate(angle_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    mean_angle = np.mean(angle_list)

    print("Validation DSEC: EPE=%f, F1=%f, Angle=%f" % (epe, f1, mean_angle))
    return {'dsec-epe': epe, 'dsec-f1': f1, 'dsec-angle': mean_angle}


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list, angle_list = [], [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = copy.deepcopy(image1)

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()
        if FLOW_MAX > 0:
            flow_np = flow.numpy()
            abs_max = np.abs(flow_np).max()
            if abs_max > 0:
                ratio = (FLOW_MAX * 0.8) / abs_max
                flow = torch.from_numpy(flow_np * ratio)

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        # Direction (angular) error
        flow_flat = flow.view(2, -1)[:, val].numpy()
        flow_gt_flat = flow_gt.view(2, -1)[:, val].numpy()
        dot = np.sum(flow_flat * flow_gt_flat, axis=0)
        norm_pred = np.linalg.norm(flow_flat, axis=0)
        norm_gt = np.linalg.norm(flow_gt_flat, axis=0)
        cos_angle = dot / (norm_pred * norm_gt + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        angle_list.append(angle)

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    angle_list = np.concatenate(angle_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    mean_angle = np.mean(angle_list)

    print("Validation KITTI: EPE=%f, F1=%f, Angle=%f" % (epe, f1, mean_angle))
    return {'kitti-epe': epe, 'kitti-f1': f1, 'kitti-angle': mean_angle}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()
    
    if args.dataset == 'KITTI':
        if args.phase == 'test':
             create_kitti_submission(model.module)
        else:
             with torch.no_grad():
                validate_kitti(model.module, split='training')
    elif args.dataset == 'DSEC_RAFT':
        dsec_root = getattr(settings, 'dsec_root', args.dataset_root)
        with torch.no_grad():
            validate_dsec(model.module, root=dsec_root, split=args.phase)
