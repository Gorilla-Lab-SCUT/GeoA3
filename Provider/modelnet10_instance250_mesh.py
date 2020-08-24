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


mesh_ten_label_indexes = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]
pc_ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]

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
            for i in mesh_ten_label_indexes:
                if label != i:
                    target_labels.append(i)
            target_labels = torch.LongTensor(target_labels)
            assert target_labels.size(0)==9

            #vertice: [N1(aggregate points number), 3(xzy)]
            vertice = torch.FloatTensor(self.vertices[index]).contiguous()
            vertices = vertice.unsqueeze(0).expand(9, -1, 3)

            #face: [N2(faces number), 3(abc)]
            face = torch.FloatTensor(self.faces[index]).contiguous().long()
            faces = face.unsqueeze(0).expand(9, -1, 3)

            return [vertices[:,:,[1,2,0]], faces, gt_labels, target_labels]

        elif self.attack_label == 'Untarget':
            label = self.label[index]
            label = pc_ten_label_indexes[mesh_ten_label_indexes.index(label)]
            gt_labels = torch.LongTensor([label]).clone()
            #vertices: [N1(aggregate points number), 3(xzy)]
            vertice = torch.FloatTensor(self.vertices[index]).contiguous()
            vertices = vertice.unsqueeze(0).expand(1, -1, 3)
            #faces: [N2(faces number), 3(abc)]
            face = torch.FloatTensor(self.faces[index]).contiguous().long()
            faces = face.unsqueeze(0).expand(1, -1, 3)

            return [vertices[:,:,[1,2,0]], faces, gt_labels]

        else:
            assert False, 'Attack label not included.'

if __name__ == '__main__':
    from pytorch3d.io import load_obj, save_obj
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes

    from Lib.utility import Average_meter, accuracy
    from Model.PointNet import PointNet

    test_dataset = ModelNet10_250instance_mesh(resume='Data/modelnet10_250instances_mesh_PointNet.mat', attack_label='Untarget')
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    net = PointNet(40, npoint=1024).cuda()
    model_path = os.path.join('Pretrained/PointNet/1024/model_best.pth.tar')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    test_acc = Average_meter()

    for i, data in enumerate(test_loader):
        vertice, faces_idx, gt_label = data[0], data[1], data[2]
        gt_target = gt_label.view(-1).cuda()
        src_mesh = Meshes(verts=vertice, faces=faces_idx).cuda()
        pc_ori = sample_points_from_meshes(src_mesh, 1024).permute(0,2,1)
        with torch.no_grad():
            output = net(pc_ori)
        acc = accuracy(output.data, gt_target.data, topk=(1, ))
        test_acc.update(acc[0][0], output.size(0))
        print("Prec@1 {top1.avg:.3f}".format(top1=test_acc))
