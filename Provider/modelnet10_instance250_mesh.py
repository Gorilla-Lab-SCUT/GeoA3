import os
import os.path
from scipy.io import loadmat
import sys
import numpy as np
import struct
import math
import h5py

import torch
from torch.utils.data.dataloader import default_collate


ten_label_indexes = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']

class ModelNet10_250instance_mesh():
    def __init__(self, resume, attack_label='All'):
        if '.mat' not in resume:
            self.data_root = os.path.join(resume, 'modelnet10_250instances.mat')
        else:
            self.data_root = resume
        self.attack_label = attack_label

        if not os.path.exists(self.data_root):
            assert False, 'No ModelNet40 Data!'
        dataset = loadmat(self.data_root)

        #vertices:[250, N(different), 3] ([B,N,(x,y,z)])
        #faces: [250, N(different), 3] ([B, N, (a,b,c)])
        #label: [250]
        vertices = dataset['vertices'][0]
        faces = dataset['faces'][0]
        label = dataset['label'][0]

        if attack_label == 'All' or attack_label == 'Untarget':
            self.vertices = vertices
            self.faces = faces
            self.label = label
            self.start_index = 0
        elif attack_label in ten_label_names:
            for k, label_name in enumerate(ten_label_names):
                if attack_label == label_name:
                    self.vertices = vertices[k*25:(k+1)*25]
                    self.faces = faces[k*25:(k+1)*25]
                    self.label = label[k*25:(k+1)*25]
                    self.start_index = k*25
        else:
            assert False

    def __len__(self):
        return self.vertices.__len__()

    def __getitem__(self, index):
        if (self.attack_label in ten_label_names) or (self.attack_label == 'All'):
            label = self.label[index]
            gt_labels = torch.LongTensor([label]).clone()

            target_labels = []
            for i in ten_label_indexes:
                if label != i:
                    target_labels.append(i)
            target_labels = torch.LongTensor(target_labels)
            assert target_labels.size(0)==9

            #vertice: [N1(aggregate points number), 3(xzy)]
            vertice = torch.FloatTensor(self.vertices[index]).contiguous()
            #face: [N2(faces number), 3(abc)]
            face = torch.FloatTensor(self.faces[index]).contiguous().long()

            return [vertice, face, gt_labels, target_labels]

        elif self.attack_label == 'Untarget':
            label = self.label[index]
            gt_labels = torch.LongTensor([label]).clone()
            #vertices: [N1(aggregate points number), 3(xzy)]
            vertice = torch.FloatTensor(self.vertices[index]).contiguous()
            #faces: [N2(faces number), 3(abc)]
            face = torch.FloatTensor(self.faces[index]).contiguous().long()

            return [vertice, face, gt_labels]

        else:
            assert False, 'Attack label not included.'

if __name__ == '__main__':
    test_dataset = ModelNet10_250instance_mesh(resume='Exps/PointNet/baseline/Id_0_withRotation_resampleData_batchsize_32_wd0.0001/', attack_label='Untarget')
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    itor = iter(test_loader)
    data = itor.next()

