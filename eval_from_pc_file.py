from __future__ import absolute_import, division, print_function

import argparse
import os
import re
import sys
import time

import numpy as np
import scipy.io as sio
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import torch
import torch.nn as nn
from torch.autograd import Variable
from Lib.utility import natural_sort

parser = argparse.ArgumentParser(description='Evluate Single Obj File')
parser.add_argument('--datadir', default='./Vis/Post_Mesh/', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--trg_file_path', default=None, type=str, metavar='DIR', help='path to adv dataset')
parser.add_argument('--arch', default='PointNet', type=str, metavar='ARCH', help='')
parser.add_argument('--npoint', default=1024, type=int, help='')
parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--random_seed', default=0, type=int, help='')
parser.add_argument('--is_mesh', action='store_true', default=False, help='')
parser.add_argument('--is_debug', action='store_true', default=False, help='')
cfg  = parser.parse_args()
print(cfg)

type_to_index_map = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39
}


def read_off_lines_from_xyz(path, num_points):
    with open(path) as file:
        vertices = []
        if num_points == -1:
            num_points = len(open(path,'r').readlines())
        for i in range(num_points):
            line = file.readline()
            vertices.append([float(x) for x in line.split()[0:3]])

    return vertices

def read_off_lines_from_obj(path, num_points):
    with open(path) as file:
        vertices = []
        if num_points == -1:
            num_points = len(open(path,'r').readlines())
        for i in range(num_points):
            line = file.readline()
            vertices.append([float(x) for x in line.split()[1:4]])

    return vertices


def pc_normalize_torch(point):
    #point:[n,3]
    assert len(point.size()) == 2
    assert point.size(1) == 3
    # normalize the point and face
    with torch.no_grad():
        avg = torch.mean(point.t(), dim = 1)
        scale = torch.max(torch.norm(point, dim = 1), dim = 0)[0]
        #scale = torch.max(point.abs().max(0).values)

    normed_point = point - avg[np.newaxis, :]
    normed_point = normed_point / scale[np.newaxis, np.newaxis]
    return normed_point

def main():
    seed = cfg.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # model
    print('=>Loading model')
    model_path = os.path.join('Pretrained', cfg.arch, str(cfg.npoint), 'model_best.pth.tar')
    if cfg.arch == 'PointNet':
        from Model.PointNet import PointNet
        net = PointNet(cfg.classes, npoint=cfg.npoint).cuda()
    elif cfg.arch == 'PointNetPP':
        #from Model.PointNetPP_msg import PointNet2ClassificationMSG
        #net = PointNet2ClassificationMSG(use_xyz=True, use_normal=False).cuda()
        from Model.PointNetPP_ssg import PointNet2ClassificationSSG
        net = PointNet2ClassificationSSG(use_xyz=True, use_normal=False).cuda()
    elif cfg.arch == 'DGCNN':
        from Model.DGCNN import DGCNN_cls
        net = DGCNN_cls(k=20, emb_dims=cfg.npoint, dropout=0.5).cuda()
    else:
        assert False, 'Not support such arch.'

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('==>Successfully load pretrained-model from {}'.format(model_path))

    file_names = os.listdir(os.path.join(cfg.datadir))
    file_names = natural_sort(file_names)

    if cfg.trg_file_path is not None:
        cnt_attack_success = 0

        trg_save_path = os.path.join(cfg.trg_file_path, "../Dis")
        if not os.path.exists(trg_save_path):
            os.makedirs(trg_save_path)

        trg_file_names = os.listdir(os.path.join(cfg.trg_file_path))
        trg_file_names = natural_sort(trg_file_names)

        for i, trg_name in enumerate(trg_file_names):
            if "_" in trg_name:
                continue
            idx = trg_name.split("_")[1]
            gt_label = int(re.findall(r"\d+\.?\d*",trg_name.split("_")[2])[0])
            adv_label = int(re.findall(r"\d+\.?\d*",trg_name.split("_")[3].split(".")[0])[0])

            pc_adv = read_off_lines_from_obj(os.path.join(cfg.trg_file_path, trg_name), cfg.npoint)
            pc_adv = torch.FloatTensor(pc_adv[:])
            pc_adv = pc_adv.t().unsqueeze(0).cuda()

            points = read_off_lines_from_xyz(os.path.join(cfg.datadir, str(idx)+".xyzn"), -1)
            ori_pc = torch.FloatTensor(points[:])
            ori_pc = ori_pc.t().unsqueeze(0).cuda()

            b,_,n=pc_adv.size()
            inter_dis = ((pc_adv.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
            _, inter_idx = torch.topk(inter_dis, 1, dim=2, largest=False, sorted=True) #not the same variable, use k=k
            inter_idx = inter_idx.contiguous()
            curr_pc = torch.gather(ori_pc, 2, inter_idx.view(b,1,n).expand(b,3,n))

            pc_var = Variable(curr_pc, requires_grad=False)
            output_var = net(pc_var)

            pred_label = torch.max(output_var.data.cpu(),1)[1]


            if pred_label==adv_label:
                cnt_attack_success+=1

            print('[{0}/{1}] of \'{2}\', pred label {3}, attack still success {4:3.2f}%'.format(i, len(trg_file_names), trg_name, pred_label.item(), cnt_attack_success/float(i+1)*100))

            if cfg.is_debug:
                fout = open(os.path.join(trg_save_path, "curr_"+str((pred_label==adv_label).item())+"_"+trg_name.split(".")[0]+".xyz"), 'w')
                for m in range(curr_pc.shape[2]):
                    fout.write('%f %f %f 0 0 0\n' % (curr_pc[0, 0, m], curr_pc[0, 1, m], curr_pc[0, 2, m]))
                fout.close()

    elif cfg.is_mesh:
        cnt_attack_success = 0

        for i, mesh_name in enumerate(file_names):
            if "_" not in mesh_name:
                continue
            idx = mesh_name.split("_")[1]
            gt_label = int(re.findall(r"\d+\.?\d*",mesh_name.split("_")[2])[0])
            adv_label = int(re.findall(r"\d+\.?\d*",mesh_name.split("_")[3].split(".")[0])[0])

            if ".obj" in mesh_name:
                mesh = load_objs_as_meshes([os.path.join(cfg.datadir, mesh_name)])
                curr_pc = sample_points_from_meshes(mesh, cfg.npoint).permute(0,2,1)

                pc_var = Variable(curr_pc.cuda(), requires_grad=False)
                output_var = net(pc_var)

                pred_label = torch.max(output_var.data.cpu(),1)[1]

                if pred_label==adv_label:
                    cnt_attack_success+=1

                print('[{0}/{1}] of \'{2}\', pred label {3}, attack still success {4:3.2f}%'.format(i, len(file_names), mesh_name, pred_label.item(), cnt_attack_success/float(i+1)*100))
            else:
                pass
    else:
        for i, pc_name in enumerate(file_names):
            if ".xyz" in pc_name:
                curr_pc = []
                if ".xyz" in pc_name:
                    points = read_off_lines_from_xyz(os.path.join(cfg.datadir, pc_name), cfg.npoint)
                elif ".obj" in pc_name:
                    points = read_off_lines_from_obj(os.path.join(cfg.datadir, pc_name), cfg.npoint)

                ori_pc = torch.FloatTensor(points[:])
                curr_pc = ori_pc.t().unsqueeze(0).cuda()

                pc_var = Variable(curr_pc, requires_grad=False)
                output_var = net(pc_var)

                pred_label = torch.max(output_var.data.cpu(),1)[1]

                print('[{0}/{1}] of \'{2}\', pred label {3}'.format(i, len(file_names), pc_name, pred_label.item()))

                if cfg.is_debug:
                    fout = open(os.path.join(cfg.datadir, "../Test", "curr_"+pc_name), 'w')
                    for m in range(curr_pc.shape[2]):
                        fout.write('%f %f %f 0 0 0\n' % (curr_pc[0, 0, m], curr_pc[0, 1, m], curr_pc[0, 2, m]))
                    fout.close()
            else:
                pass


if __name__ == '__main__':
    main()





