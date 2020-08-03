import os
import sys
import numpy as np
import h5py
from scipy.io import loadmat

import torch
from torch.utils.data.dataloader import default_collate

ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']

class ModelNet_pure():
    def __init__(self, data_mat_file):
        self.data_root = data_mat_file

        if not os.path.isfile(self.data_root):
            assert False, 'Not exists .mat file!'

        dataset = loadmat(self.data_root)
        data = torch.FloatTensor(dataset['data'])
        normal = torch.FloatTensor(dataset['normal'])
        label = dataset['label']

        self.start_index = 0
        self.data = data
        self.normal = normal
        self.label = label

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):

        label = self.label[index]
        gt_label = torch.IntTensor(label).long()

        pc = self.data[index].contiguous().t()
        normal = self.normal[index].contiguous().t()

        return [pc, normal, gt_label] 

