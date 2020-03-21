import os
import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from core.utils import transforms as tf


class VimeoTest(Dataset):

    def __init__(self, config):
        super(VimeoTest, self).__init__()
        self.dataset_path = '/DATA/wangshen_data/vimeo_septuplet/sequences'
        self.img_list = '/DATA/wangshen_data/vimeo_septuplet/sep_testlist.txt'
        self.paths_GT = open(os.path.join(self.img_list)).read().splitlines()
        self.input_list = [2, 3, 4]
        self.img_path = self.dataset_path
        self.config = config


    def __len__(self):
        return len(self.paths_GT)

    def __getitem__(self, idx, hasmask=False):

        key = self.paths_GT[idx]
        if '_' in key:
            name_a, name_b = key.split('_')
        elif '/' in key:
            name_a, name_b = key.split('/')
            key = name_a + '_' + name_b
        else:
            raise ValueError('Error load key')

        #### get the GT image (as the center frame and the 2 around frames)
        images = []
        for v in self.input_list:
            img = cv2.imread(osp.join(self.dataset_path, name_a, name_b, 'im{}.png'.format(v))).astype(np.float32)
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

