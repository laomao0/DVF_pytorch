import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from core.utils import transforms as tf


class UCF101Test(Dataset):

    def __init__(self, config):
        super(UCF101Test, self).__init__()
        dataset_path = '/DATA/wangshen_data/UCF101/ucf101_extrap_ours'
        all_dirs = sorted(os.listdir(dataset_path))
        self.img_list = []
        for dir in all_dirs:
            if '.' in dir:
                continue
            dir_path = os.path.join(dataset_path, dir)
            img1 = os.path.join(dir_path, 'frame_00.png')
            img2 = os.path.join(dir_path, 'frame_01.png')
            img3 = os.path.join(dir_path, 'frame_02_gt.png')
            self.img_list.append([img1, img2, img3])
        self.img_path = dataset_path
        self.config = config


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, hasmask=False):

        images = []
        imgs = self.img_list[idx]
        for img in imgs:
            img = cv2.imread(img).astype(np.float32)
            images.append(img)

        # norm
        for i, img in enumerate(images):
            images[i] = tf.normalize(images[i], self.config.input_mean,
                                     self.config.input_std)
            images[i] = torch.from_numpy(images[i]).permute(
                2, 0, 1).contiguous().float()

        if self.config.syn_type == 'inter':
            return torch.cat([images[0], images[2]], dim=0)
        elif self.config.syn_type == 'extra':
            return torch.cat([images[0], images[1]], dim=0), images[2]
        else:
            raise ValueError('Unknown syn_type ' + self.syn_type)

