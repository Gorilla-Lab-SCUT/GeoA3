from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np
import scipy.io as sio
import torch


parser = argparse.ArgumentParser(description='Visualization of original point cloud')
parser.add_argument('--input_mat', default='Data/modelnet10_250instances1024_PointNet.mat', type=str, help='')
cfg  = parser.parse_args()


dataset = sio.loadmat(cfg.input_mat)
pc = torch.FloatTensor(dataset['data'])
normal = torch.FloatTensor(dataset['normal'])
label = dataset['label']

saved_dir = cfg.input_mat.split("/")[1].split(".")[0]
trg_dir = os.path.join("./Data", saved_dir)
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)

for i in range(pc.shape[0]):
    name = str(i)
    saved_pc = pc[i].unsqueeze(0)
    save_normal = normal[i].unsqueeze(0)

    fout = open(os.path.join(trg_dir, name+'.xyz'), 'w')
    for m in range(saved_pc.shape[2]):
        fout.write('%f %f %f %f %f %f\n' % (saved_pc[0, 0, m], saved_pc[0, 1, m], saved_pc[0, 2, m], save_normal[0, 0, m], save_normal[0, 1, m], save_normal[0, 2, m]))
    fout.close()



