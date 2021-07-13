from __future__ import absolute_import, division, print_function

import argparse
import bisect
import gc
import os
import pdb
import pprint
import shutil
import sys
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.split(BASE_DIR)[0]
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Model'))


parser = argparse.ArgumentParser(description='Point Cloud Attacking')
parser.add_argument('--datadir', default='/data/modelnet40_normal_resampled/', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--out_datadir', default='Data', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--arch', default='PointNet', type=str, metavar='ARCH', help='')
parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
parser.add_argument('-outc', '--out_classes', default=10, type=int, metavar='N', help='')
parser.add_argument('-outn', '--max_out_num', default=25, type=int, metavar='N', help='')
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--pre_trn_npoint', default=1024, type=int, metavar='N', help='')
parser.add_argument('--npoint', default=1024, type=int, metavar='N', help='')
parser.add_argument('--is_using_virscan', action='store_true', default=False, help='')
parser.add_argument('--dense_npoints', default=10000, type=int, metavar='N', help='')


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

DATA_PATH = cfg.datadir

if cfg.out_classes == 10:
    label_indexes = ten_label_indexes
    label_names = ten_label_names
elif cfg.out_classes == 40:
    label_indexes = fourth_label_indexes
    label_names = fourth_label_names

def read_off_lines(path):
    with open(path) as file:
        line = file.readline()
        while 'end_header' not in line:
            line = file.readline()
            if 'element vertex' in line:
                points_num = int(line.split()[2])

        points = []
        normal = []
        for _ in range(points_num):
            line = file.readline()
            points.append([float(x) for x in line.split()][:3])
            normal.append([float(x) for x in line.split()][3:])

        points = np.array(points)
        normal = np.array(normal)
    return points, normal

def sample_points(obj, num_points, normal):
    curr_points = []
    curr_normal = []

    areas = np.cross(obj[:, 1] - obj[:, 0], obj[:, 2] - obj[:, 0])
    areas = np.linalg.norm(areas, axis = 1) / 2.0
    prefix_sum = np.cumsum(areas)
    total_area = prefix_sum[-1]

    for _ in range(num_points):
        # pick random triangle based on area
        rand = np.random.uniform(high = total_area)
        if rand >= total_area:
            idx = len(obj) - 1 # can happen due to floating point rounding
        else:
            idx = bisect.bisect_right(prefix_sum, rand)

        # pick random point in triangle
        a, b, c = obj[idx]
        r1 = np.random.random()
        r2 = np.random.random()
        if r1 + r2 >= 1.0:
            r1 = 1 - r1
            r2 = 1 - r2
        r3 = 1-r1-r2
        p = r1*a+r2*b+r3*c
        curr_points.append(p)
        curr_normal.append(normal[idx])

    points = np.array(curr_points)
    normal = np.array(curr_normal)
    return points, normal

def farthest_points_normalized_wfaces(obj_points, faces, num_points, normal):
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
    faces = faces - avg[np.newaxis, np.newaxis, :]
    dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
    res_points = res_points / dists
    faces = faces / dists

    return res_points, faces, res_normal

def farthest_points_normalized(obj_points, num_points, normal):
    first = np.random.randint(len(obj_points))
    selected = [first]
    dists = np.full(shape = len(obj_points), fill_value = np.inf)

    for _ in range(num_points - 1):
        dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
        selected.append(np.argmax(dists))
    res_points = np.array(obj_points[selected])
    res_normal = np.array(normal[selected])

    # normalize the points
    avg = np.average(res_points, axis = 0)
    res_points = res_points - avg[np.newaxis, :]
    dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
    res_points = res_points / dists

    return res_points, res_normal

def main():
    using_virscan = cfg.is_using_virscan
    # model
    model_path = os.path.join('Pretrained', cfg.arch, str(cfg.pre_trn_npoint), 'model_best.pth.tar')
    if cfg.arch == 'PointNet':
        from PointNet import PointNet
        net = PointNet(cfg.classes, npoint=cfg.pre_trn_npoint).cuda()
    elif cfg.arch == 'PointNetPP':
        from Model.PointNetPP_ssg import PointNet2ClassificationSSG
        net = PointNet2ClassificationSSG(use_xyz=True, use_normal=False).cuda()
    else:
        assert False

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('\nSuccessfully load pretrained-model from {}\n'.format(model_path))

    all_data = [[] for k in range(40)]
    all_normal = [[] for k in range(40)]
    all_label = [[] for k in range(40)]

    all_dense_data = [[] for k in range(40)]
    all_dense_normal = [[] for k in range(40)]


    if using_virscan:
        datadir = ROOT_DIR+'/Data/Ten_class_pc_normal'
        file_names = os.listdir(datadir)
        for i, file_name in enumerate(file_names):
            if '.obj' in file_name:
                continue

            ori_points, ori_normal = read_off_lines(os.path.join(datadir, file_name))
            points, normal = farthest_points_normalized(ori_points, cfg.npoint, ori_normal)
            if cfg.dense_npoints>0:
                dense_points, desne_normal = farthest_points_normalized(ori_points, cfg.dense_npoints, ori_normal)
            label = int(file_name.split('_')[1].split('.')[0])

            pc = torch.from_numpy(points).t().unsqueeze(0).float()
            normal = torch.from_numpy(normal).t().unsqueeze(0).float()
            if cfg.dense_npoints>0:
                dense_points = torch.from_numpy(dense_points).t().unsqueeze(0).float()
                desne_normal = torch.from_numpy(desne_normal).t().unsqueeze(0).float()
            label = torch.LongTensor([label])

            if label[0] in label_indexes:
                with torch.no_grad():
                    pc_var = Variable(pc.cuda(), requires_grad=False)
                    output_var = net(pc_var[:,[0,2,1],:])
                pred_label = torch.max(output_var.data.cpu(),1)[1]

                if pred_label[0] == label[0]:
                    print('[{0}/{1}] label {2}: pred successed!'.format(i, len(file_names), label[0]))

                    all_data[label[0]].append(pc[:,[0,2,1],:].clone())
                    all_normal[label[0]].append(normal[:,[0,2,1],:].clone())
                    if cfg.dense_npoints>0:
                        all_dense_data[label[0]].append(dense_points[:,[0,2,1],:].clone())
                        all_dense_normal[label[0]].append(desne_normal[:,[0,2,1],:].clone())
                    all_label[label[0]].append(label[0].clone().view(1,1))

                else:
                    print('[{0}/{1}] label {2}: pred failed!'.format(i, len(file_names), label[0]))
            else:
                print('[{0}/{1}] label {2}: pass!'.format(i, len(file_names), label[0]))

    else:
        #data
        from modelnet_trn_test import ModelNetDataset
        TEST_DATASET = ModelNetDataset(root=DATA_PATH, batch_size=1, npoints=cfg.npoint, split='test', normal_channel=True)

        i = 0
        while TEST_DATASET.has_next_batch():
            i += 1
            points, target = TEST_DATASET.next_batch(False)
            target = torch.Tensor(target).long()
            label = target.cuda()

            label = label[0]

            if label in label_indexes:
                points = torch.Tensor(points).contiguous()

                points = points.transpose(2, 1).cuda()

                pc = points[:,[0,2,1],:]
                normal = points[:,[3,5,4],:]

                with torch.no_grad():
                    pc_var = Variable(pc.cuda(), requires_grad=False)
                    output_var = net(pc_var)
                pred_label = torch.max(output_var.data.cpu(),1)[1]

                if pred_label[0] == label:
                    print('[{0}/{1}] label {2}: pred successed!'.format(i, len(TEST_DATASET), label))

                    all_data[label].append(pc.clone())
                    all_normal[label].append(normal.clone())
                    all_label[label].append(torch.LongTensor([label]).clone().view(1,1))

                else:
                    print('[{0}/{1}] label {2}: pred failed!'.format(i, len(TEST_DATASET), label))
            else:
                print('[{0}/{1}] label {2}: pass!'.format(i, len(TEST_DATASET), label))

    saved_data = []
    saved_normal = []
    saved_label = []

    save_dense_data = []
    save_dense_normal = []

    all_num = 0

    for j, k in enumerate(label_indexes):
        tmp_data = torch.cat(all_data[k], 0)
        tmp_normal = torch.cat(all_normal[k], 0)
        if using_virscan & (cfg.dense_npoints>0):
            tmp_dense_data = torch.cat(all_dense_data[k], 0)
            tmp_dense_normal = torch.cat(all_dense_normal[k], 0)
        tmp_label = torch.cat(all_label[k], 0)


        num = tmp_data.size(0)
        all_num += num
        print('{0}: {1}'.format(label_names[j], num))

        index = torch.randperm(num)[:cfg.max_out_num].long()
        saved_data.append(tmp_data[index])
        saved_normal.append(tmp_normal[index])
        if using_virscan & (cfg.dense_npoints>0):
            save_dense_data.append(tmp_dense_data[index])
            save_dense_normal.append(tmp_dense_normal[index])
        saved_label.append(tmp_label[index])

    saved_data = torch.cat(saved_data, 0).cpu().numpy()
    saved_normal = torch.cat(saved_normal, 0).cpu().numpy()
    if using_virscan & (cfg.dense_npoints>0):
        saved_dense_data = torch.cat(save_dense_data, 0).cpu().numpy()
        saved_dense_normal = torch.cat(save_dense_normal, 0).cpu().numpy()
    saved_label = torch.cat(saved_label, 0).cpu().numpy()

    sio.savemat(os.path.join(cfg.out_datadir, 'modelnet' + str(cfg.out_classes) + '_' + str(saved_data.shape[0]) + 'instances' + str(cfg.npoint) + '_' + str(cfg.arch) + '.mat'), {"data": saved_data, 'normal': saved_normal, 'label': saved_label})
    if using_virscan & (cfg.dense_npoints>0):
        sio.savemat(os.path.join(cfg.out_datadir, 'modelnet' + str(cfg.out_classes) + '_' + str(saved_dense_data.shape[0]) + 'instances' + str(cfg.dense_npoints) + '_' + str(cfg.arch) + '.mat'), {"data": saved_dense_data, 'normal': saved_dense_normal, 'label': saved_label})


if __name__ == '__main__':
    main()
