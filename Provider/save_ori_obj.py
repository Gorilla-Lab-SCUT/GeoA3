from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import shutil
import gc
import scipy.io as sio
import bisect

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from modelnet40_with_vert import ModelNet40_vert
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.split(BASE_DIR)[0]
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Model'))
sys.path.append(os.path.join(ROOT_DIR, 'Lib'))


parser = argparse.ArgumentParser(description='Saving ori obj mesh')
cfg  = parser.parse_args()
print(cfg)

ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']


fourth_label_indexes = range(40)
fourth_label_names = [
    'night_stand', 'range_hood', 'plant', 'chair', 'tent',
    'curtain', 'piano', 'dresser', 'desk', 'bed',
    'sink',  'laptop', 'flower_pot', 'car', 'stool',
    'vase', 'monitor', 'airplane', 'stairs', 'glass_box',
    'bottle', 'guitar', 'cone',  'toilet', 'bathtub',
    'wardrobe', 'radio',  'person', 'xbox', 'bowl',
    'cup', 'door',  'tv_stand',  'mantel', 'sofa',
    'keyboard', 'bookshelf',  'bench', 'table', 'lamp'
]
convert_from_modelnet40_1024_processed = [17, 24, 9, 37, 36, 20, 29, 13, 3, 22, 30, 5, 8, 31, 7, 12, 19, 21, 35, 39, 11, 33, 16,  0, 27, 6, 2, 26, 1, 10, 34, 18, 14,  38,  4, 23, 32, 15, 25, 28 ]

label_indexes = ten_label_indexes
label_names = ten_label_names

def pc_normalize_torch(point):
    #point:[n,3]
    assert len(point.size()) == 2
    assert point.size(1) == 3
    # normalize the point and face
    with torch.no_grad():
        avg = torch.mean(point.t(), dim = 1)
    normed_point = point - avg[np.newaxis, :]
    with torch.no_grad():
        scale = torch.max(torch.norm(normed_point, dim = 1), dim = 0).values
    normed_point = normed_point / scale[np.newaxis, np.newaxis]
    return normed_point, avg, scale


def main():
    test_dataset = ModelNet40_vert('/data/ModelNet40/', 40, phase='test', regenerate_dataset=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_size = test_dataset.__len__()

    if not os.path.exists(os.path.join('../Vis', 'Ori_Mesh')):
            os.makedirs(os.path.join('../Vis', 'Ori_Mesh'))

    for i, (vert, faces, label) in enumerate(test_loader):
        if convert_from_modelnet40_1024_processed[label[0]] in label_indexes:
            vert = vert.squeeze(0)
            faces = faces.squeeze(0)
            vert, _, _ = pc_normalize_torch(vert)
            trg_mesh = Meshes(verts=[vert], faces=[faces]).cuda()

            file_name = os.path.join('../Vis', 'Ori_Mesh', str(i)+'_'+str(convert_from_modelnet40_1024_processed[label[0]])+'.obj')
            final_verts, final_faces = trg_mesh.get_mesh_verts_faces(0)
            print('Processing ['+str(i)+'/'+str(test_size)+' ] instance')
            save_obj(file_name, final_verts, final_faces)


if __name__ == '__main__':
    main()  
