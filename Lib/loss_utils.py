from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time

import numpy as np
from pytorch3d.ops import knn_points, knn_gather
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Model'))
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from utility import _normalize


def norm_l2_loss(adv_pc, ori_pc):
    return ((adv_pc - ori_pc)**2).sum(1).sum(1)

def chamfer_loss(adv_pc, ori_pc):
    # Chamfer distance (two sides)
    intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    dis_loss = intra_dis.min(2)[0].mean(1) + intra_dis.min(1)[0].mean(1)
    return dis_loss

def pseudo_chamfer_loss(adv_pc, ori_pc):
    # Chamfer pseudo distance (one side)
    intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1) #b*n*n
    dis_loss = intra_dis.min(2)[0].mean(1)
    return dis_loss

def hausdorff_loss(adv_pc, ori_pc):
    dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    return torch.max(torch.min(dis, dim=2)[0], dim=1)[0]


def _get_kappa_ori(pc, normal, k=2):
    b,_,n=pc.size()
    #inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    #inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    #nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(pc.permute(0,2,1), pc.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = knn_gather(pc.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]
    vectors = nn_pts - pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2) # [b, n]

def _get_kappa_adv(adv_pc, ori_pc, ori_normal, k=2):
    b,_,n=adv_pc.size()
    # compute knn between advPC and oriPC to get normal n_p
    #intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    #intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    #normal = torch.gather(ori_normal, 2, intra_idx.view(b,1,n).expand(b,3,n))
    intra_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    # compute knn between advPC and itself to get \|q-p\|_2
    #inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
    #inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    #nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(adv_pc.permute(0,2,1), adv_pc.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = knn_gather(adv_pc.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2), normal # [b, n], [b, 3, n]

def curvature_loss(adv_pc, ori_pc, adv_kappa, ori_kappa, k=2):
    b,_,n=adv_pc.size()

    # intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
    # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    # knn_theta_normal = torch.gather(theta_normal, 1, intra_idx.view(b,n).expand(b,n))
    # curv_loss = ((curv_loss - knn_theta_normal)**2).mean(-1)

    intra_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    onenn_ori_kappa = torch.gather(ori_kappa, 1, intra_KNN.idx.squeeze(-1)).contiguous() # [b, n]

    curv_loss = ((adv_kappa - onenn_ori_kappa)**2).mean(-1)

    return curv_loss

def displacement_loss(adv_pc, ori_pc, k=16):
    b,_,n=adv_pc.size()
    with torch.no_grad():
        inter_dis = ((ori_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
        inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()

    theta_distance = ((adv_pc - ori_pc)**2).sum(1)
    nn_theta_distances = torch.gather(theta_distance, 1, inter_idx.view(b, n*k)).view(b,n,k)
    return ((nn_theta_distances-theta_distance.unsqueeze(2))**2).mean(2)

def corresponding_normal_loss(adv_pc, normal, k=2):
    b,_,n=adv_pc.size()

    inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
    inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)
    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2)

def repulsion_loss(pc, k=4, h=0.03):
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    dis = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)[0][:, :, 1:].contiguous()

    return -(dis * torch.exp(-(dis**2)/(h**2))).mean(2)

def distance_kmean_loss(pc, k):
    b,_,n=pc.size()
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12)**2).sum(1).sqrt()
    dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)
    dis_mean = dis[:, :, 1:].contiguous().mean(-1) #b*n
    idx = idx[:, :, 1:].contiguous()
    dis_mean_k = torch.gather(dis_mean, 1, idx.view(b, n*k)).view(b, n, k)

    return torch.abs(dis_mean.unsqueeze(2) - dis_mean_k).mean(-1)

def kNN_smoothing_loss(adv_pc, k, threshold_coef=1.05):
    b,_,n=adv_pc.size()
    #dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1) #[b,n,n]
    #dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)#[b,n,k+1]
    inter_KNN = knn_points(adv_pc.permute(0,2,1), adv_pc.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]

    knn_dis = inter_KNN.dists[:, :, 1:].contiguous().mean(-1)#[b,n]
    knn_dis_mean = knn_dis.mean(-1) #[b]
    knn_dis_std = knn_dis.std(-1) #[b]
    threshold = knn_dis_mean + threshold_coef * knn_dis_std #[b]

    condition = torch.gt(knn_dis, threshold.unsqueeze(1)).float() #[b,n]
    dis_mean = knn_dis * condition #[b,n]

    return dis_mean.mean(1) #[b]

def uniform_loss(adv_pc, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0, k=2):
    if adv_pc.size(1) == 3:
        adv_pc = adv_pc.permute(0,2,1).contiguous()
    b,n,_=adv_pc.size()
    npoint = int(n * 0.05)
    for p in percentages:
        p = p*4
        nsample = int(n*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi *(radius ** 2) * p/nsample
        expect_len = torch.sqrt(torch.Tensor([disk_area])).cuda()

        adv_pc_flipped = adv_pc.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(adv_pc_flipped, pointnet2_utils.furthest_point_sample(adv_pc, npoint)).transpose(1, 2).contiguous() # (batch_size, npoint, 3)

        idx = pointnet2_utils.ball_query(r, nsample, adv_pc, new_xyz) #(batch_size, npoint, nsample)

        grouped_pcd = pointnet2_utils.grouping_operation(adv_pc_flipped, idx).permute(0,2,3,1).contiguous()  # (batch_size, npoint, nsample, 3)
        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, axis=1), axis=0)

        grouped_pcd = grouped_pcd.permute(0,2,1).contiguous()
        #dis = torch.sqrt(((grouped_pcd.unsqueeze(3) - grouped_pcd.unsqueeze(2))**2).sum(1)+1e-12) # (batch_size*npoint, nsample, nsample)
        #dists, _ = torch.topk(dis, k+1, dim=2, largest=False, sorted=True) # (batch_size*npoint, nsample, k+1)
        inter_KNN = knn_points(grouped_pcd.permute(0,2,1), grouped_pcd.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]

        uniform_dis = inter_KNN.dists[:, :, 1:].contiguous()
        uniform_dis = torch.sqrt(torch.abs(uniform_dis)+1e-12)
        uniform_dis = uniform_dis.mean(axis=[-1])
        uniform_dis = (uniform_dis - expect_len)**2 / (expect_len + 1e-12)
        uniform_dis = torch.reshape(uniform_dis, [-1])

        mean = uniform_dis.mean()
        mean = mean*math.pow(p*100,2)

        #nothing 4
        try:
            loss = loss+mean
        except:
            loss = mean
    return loss/len(percentages)







