# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import os
import math
import random
from glob import glob
import os.path as osp

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)



class DSECRAFT(FlowDataset):
    def __init__(self, root='data/RAFT_flow_dataset', split='train'):
        super(DSECRAFT, self).__init__()
        image_dir = os.path.join(root, split, 'conditioning_images')
        flow_dir = os.path.join(root, split, 'optical_flow')
        image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if '-1.jpg' in file]
        flow_list = [self.img2flow_name(flow_dir, file) for file in os.listdir(image_dir)]
        self.image_list, self.flow_list = [], []
        for im, flow in zip(image_list, flow_list):
            if flow is not None:
                self.image_list += [im]
                self.flow_list += [os.path.join(flow_dir, flow)]

    def img2flow_name(self, flow_dir, name):
        name_prefix, index = name.split('-')[0], name.split('-')[1]
        index = index.split('.')[0]
        flow_name = name_prefix + '-{}-1-flow_up.npy'.format(index)
        if os.path.exists(os.path.join(flow_dir, flow_name)):
            return flow_name
        else:
            return None

    def __getitem__(self, index):
        img1 = Image.open(self.image_list[index])
        img2 = Image.open(self.image_list[index])
        flow = np.load(self.flow_list[index]).squeeze().transpose(1,2,0).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        valid = np.ones_like(flow)[:,:,0]
        if self.augmentor is not None:
            # if self.sparse:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
        valid = torch.from_numpy(valid).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        return img1, img2, flow, valid



class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='dataset/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))




def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """


    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
    if args.dataset == 'KITTI':
        train_dataset = KITTI(aug_params, split='training', root=args.dataset_root)
    elif args.dataset == 'DSEC_RAFT':
        train_dataset = DSECRAFT(root=args.dataset_root, split='train')
    else:
        raise NotImplementedError

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


if __name__ == '__main__':
    dataset = DSECRAFT(split='train')
    print(len(dataset))
    for i in range(len(dataset)):
        img1, img2, flow, valid = dataset[i]
        a = 1
        # print(img1.shape, img2.shape, flow.shape, valid.shape)