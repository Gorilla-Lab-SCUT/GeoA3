import os
import sys
import numpy as np
import h5py
from scipy.io import loadmat

import torch
from torch.utils.data.dataloader import default_collate

ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']


class ModelNet40():
    def __init__(self, data_mat_file='../Data/modelnet10_250instances_1024.mat', attack_label='All', resample_num=-1, is_half_forward=False):
        self.data_root = data_mat_file
        self.attack_label = attack_label
        self.is_half_forward = is_half_forward

        if not os.path.isfile(self.data_root):
            assert False, 'No exists .mat file!'

        dataset = loadmat(self.data_root)
        data = torch.FloatTensor(dataset['data'])
        normal = torch.FloatTensor(dataset['normal'])
        label = dataset['label']

        if resample_num>0:
            tmp_data_set = []
            tmp_normal_set = []
            for j in range(data.size(0)):
                tmp_data, tmp_normal  = self.__farthest_points_normalized(data[j].t(), resample_num, normal[j].t())
                tmp_data_set.append(torch.from_numpy(tmp_data).t().float())
                tmp_normal_set.append(torch.from_numpy(tmp_normal).t().float())
            data = torch.stack(tmp_data_set)
            normal = torch.stack(tmp_normal_set)

        if attack_label in ten_label_names:
            for k, label_name in enumerate(ten_label_names):
                if attack_label == label_name:
                    self.start_index = k*25
                    self.data = data[k*25:(k+1)*25]
                    self.normal = normal[k*25:(k+1)*25]
                    self.label = label[k*25:(k+1)*25]
        elif attack_label == 'All':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        elif attack_label == 'Untarget' or attack_label == 'RandomTarget' or attack_label == 'SingleLabel':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        else:
            assert False

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):

        if (self.attack_label in ten_label_names) or (self.attack_label == 'All'):
            label = self.label[index]

            target_labels = []
            for i in ten_label_indexes:
                if label != i:
                    target_labels.append(i)
            target_labels = torch.IntTensor(np.array(target_labels)).long()
            gt_labels = torch.IntTensor(label).long().expand_as(target_labels)
            assert target_labels.size(0)==9

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(9, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(9, -1, -1)
            if self.is_half_forward:
                return [[pcs[:4,:,:], normals[:4,:,:], gt_labels[:4], target_labels[:4]],[pcs[4:,:,:], normals[4:,:,:], gt_labels[4:], target_labels[4:]]]
            else:
                return [pcs, normals, gt_labels, target_labels]

        elif (self.attack_label == 'Untarget' or self.attack_label == 'RandomTarget' or self.attack_label == 'SingleLabel'):
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)
            return [pcs, normals, gt_labels] 

    def __farthest_points_normalized(self, obj_points, num_points, normal):
        first = np.random.randint(len(obj_points))
        selected = [first]
        dists = np.full(shape = len(obj_points), fill_value = np.inf)

        for _ in range(num_points - 1):
            dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
            selected.append(np.argmax(dists))
        res_points = np.array(obj_points[selected])
        res_normal = np.array(normal[selected])
            
        # normalize the points and faces
        avg = np.average(res_points, axis = 0)
        res_points = res_points - avg[np.newaxis, :]
        dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
        res_points = res_points / dists

        return res_points, res_normal
