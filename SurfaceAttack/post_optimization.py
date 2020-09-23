from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

import ipdb
import numpy as np
import scipy.io as sio
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Lib'))

from loss_utils import chamfer_loss
from utility import natural_sort

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post optimization from original mesh to adversarial point cloud")
    #------------Path-----------------------
    parser.add_argument('--adv_mat_path', default="Exps/PointNet_npoint1024/All/GeoA3_7_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16/Mat/", type=str, help='')
    parser.add_argument('--ori_mesh_path', default="Data/Ori_10000_PointNet_PSR/", type=str, help='')
    parser.add_argument('--save_dir', default=None, type=str, help='')
    #------------Opti-----------------------
    parser.add_argument('--optim', default='adam', type=str, help='adam| sgd')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--iter_max_steps', type=int, default=1000, help='')
    parser.add_argument('--is_debug', action='store_true', default=False, help='')
    cfg  = parser.parse_args()

    step_print_freq = 50
    i = 0

    edge_loss_weight = 0.5
    laplacian_loss_weight = 0.01
    normal_consis_loss_weight = 0.01

    save_dir = cfg.save_dir or os.path.join(cfg.adv_mat_path, "../Reconstruct_from_ori_mesh")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    adv_pc_file_name_list = os.listdir(cfg.adv_mat_path)
    adv_pc_file_name_list = natural_sort(adv_pc_file_name_list)

    ori_mesh_file_name_list = os.listdir(cfg.ori_mesh_path)
    ori_mesh_file_name_list = natural_sort(ori_mesh_file_name_list)

    for trg_pc_file in adv_pc_file_name_list:
        i+=1
        dataset = sio.loadmat(os.path.join(cfg.adv_mat_path, trg_pc_file))
        trg_pc = torch.FloatTensor(dataset["adversary_point_clouds"]).unsqueeze(0).cuda()

        obj_index = trg_pc_file.split("_")[1]

        ori_mesh_file = os.path.join(cfg.ori_mesh_path, ori_mesh_file_name_list[0]).replace("_0", "_"+obj_index).replace(".mtl", ".obj")
        ori_mesh = load_objs_as_meshes([ori_mesh_file]).cuda()

        deform_verts = torch.zeros(ori_mesh.verts_packed().shape).cuda()
        deform_verts = deform_verts.requires_grad_()
        if cfg.optim == 'adam':
            optimizer = torch.optim.Adam([deform_verts], lr=cfg.lr)
        elif cfg.optim == 'sgd':
            optimizer = torch.optim.SGD([deform_verts], lr=cfg.lr, momentum=0.9)
        else:
            assert False, 'Wrong optimizer!'
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)

        for step in range(cfg.iter_max_steps):

            new_src_mesh = ori_mesh.offset_verts(deform_verts)
            curr_pc = sample_points_from_meshes(new_src_mesh, 10000).permute(0,2,1)

            #loss, _ = chamfer_distance(curr_pc, trg_pc, batch_reduction='mean')
            loss = chamfer_loss(curr_pc, trg_pc) + edge_loss_weight * mesh_edge_loss(new_src_mesh) + laplacian_loss_weight * mesh_laplacian_smoothing(new_src_mesh, method="uniform") + normal_consis_loss_weight * mesh_normal_consistency(new_src_mesh)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step > (cfg.iter_max_steps * 1/10):
                lr_scheduler.step()

            info = '[{0}/{1}][{2}/{3}] \t loss: {4:6.5f}\t'.format(i, adv_pc_file_name_list.__len__(), step+1, cfg.iter_max_steps, loss.item())

            if (step+1) % step_print_freq == 0 or step == cfg.iter_max_steps - 1:
                print(info)

            if (step%50 == 0) and cfg.is_debug:
                if (step == 0):
                    file_name = os.path.join(save_dir, 'ori'+'.obj')
                    final_verts, final_faces = ori_mesh.get_mesh_verts_faces(0)
                    save_obj(file_name, final_verts, final_faces)
                else:
                    file_name = os.path.join(save_dir, str(step)+'.obj')
                    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
                    save_obj(file_name, final_verts, final_faces)

                fout = open(os.path.join(save_dir, str(step)+'_pc.xyz'), 'w')
                for m in range(curr_pc.shape[2]):
                    fout.write('%f %f %f 0 0 0\n' % (curr_pc[0, 0, m], curr_pc[0, 1, m], curr_pc[0, 2, m]))
                fout.close()

        file_name = os.path.join(save_dir, trg_pc_file.replace(".mat", ".obj"))
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        save_obj(file_name, final_verts, final_faces)

        fout = open(os.path.join(save_dir, trg_pc_file.replace(".mat", ".xyz")), 'w')
        for m in range(curr_pc.shape[2]):
            fout.write('%f %f %f 0 0 0\n' % (curr_pc[0, 0, m], curr_pc[0, 1, m], curr_pc[0, 2, m]))
        fout.close()

        print("[{0}/{1}]".format(i, adv_pc_file_name_list.__len__()))
        if cfg.is_debug:
            ipdb.set_trace()



