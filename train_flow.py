from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import evaluate_flow
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from RAFT.raft import RAFT
# import evaluate
import RAFT.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    # Angular error
    flow_pred = flow_preds[-1]
    dot = torch.sum(flow_pred * flow_gt, dim=1)
    mag_pred = torch.sum(flow_pred**2, dim=1).sqrt()
    cos = dot / (mag_pred * mag + 1e-8)
    cos = torch.clamp(cos, -1.0, 1.0)
    angle = torch.acos(cos) * 180.0 / np.pi
    angle = angle.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        'angle': angle.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, log_dir=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log_dir = log_dir
        if self.log_dir:
            self.log_file = os.path.join(self.log_dir, 'training_log.csv')
            if not os.path.exists(self.log_file):
                 with open(self.log_file, 'w') as f:
                     # Will write header later when we know the keys
                     pass
        else:
             self.log_file = None

    def _print_training_status(self):
        sorted_keys = sorted(self.running_loss.keys())
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted_keys]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        if self.total_steps + 1 <= SUM_FREQ:
             header_str = f"Step, LR, {', '.join(sorted_keys)}"
             print(header_str)
             if self.log_file:
                 with open(self.log_file, 'a') as f:
                     if os.stat(self.log_file).st_size == 0:
                         f.write(header_str + '\n')

        print(training_str + metrics_str)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                log_line = f"{self.total_steps+1}, {self.scheduler.get_last_lr()[0]:.7f}, {', '.join(['{:.4f}'.format(x) for x in metrics_data])}"
                f.write(log_line + '\n')

        for k in self.running_loss:
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        pass

    def close(self):
        pass


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    out_dir = args.out_dir

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt, weights_only=False), strict=False)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, log_dir=out_dir)


    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            image2 = image1
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = os.path.join(out_dir, '{:06d}.pth'.format(total_steps))
                torch.save(model.state_dict(), PATH)

                model.eval()
                if args.validation is not None:
                    for val_dataset in args.validation:
                        if val_dataset == 'KITTI':
                            evaluate_flow.validate_kitti(model.module)
                        elif val_dataset == 'DSEC_RAFT':
                            evaluate_flow.validate_dsec(model.module, root=args.dataset_root, split='test')

                model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = os.path.join(out_dir, 'final.pth')
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--dataset', type=str, default='KITTI', help='dataset for training')
    parser.add_argument('--dataset_root', type=str, default='datasets/kitti', help='dataset for training')
    parser.add_argument('--out_dir', type=str, default='checkpoints/tmp', help='directory to save checkpoints and summaries')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[375, 1242])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    train(args)