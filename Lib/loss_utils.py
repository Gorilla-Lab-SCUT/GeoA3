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
import pdb

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients

from utility import _normalize, compute_theta_normal

def norm_l2_loss(adv_pc, ori_pc, norm=2):
    return ((adv_pc - ori_pc)**2 + 1e-10).sum(1).sum(1).sqrt()

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

def normal_loss(adv_pc, ori_pc, ori_normal, theta_normal_var, k=2):
    b,_,n=adv_pc.size()
    # compute knn between advPC and oriPC to get normal n_p
    intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    normal = torch.gather(ori_normal, 2, intra_idx.view(b,1,n).expand(b,3,n))
    if theta_normal_var is not None:
        # print(theta_normal_var)
        the_normal_loss = torch.gather(theta_normal_var, 1, intra_idx.view(b, n))
    else:
        the_normal_loss = None

    # compute knn between advPC and itself to get \|q-p\|_2
    inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
    inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2), the_normal_loss

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
    # pdb.set_trace()
    b,_,n=pc.size()
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12)**2).sum(1).sqrt()
    dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)
    dis_mean = dis[:, :, 1:].contiguous().mean(-1) #b*n
    idx = idx[:, :, 1:].contiguous()
    dis_mean_k = torch.gather(dis_mean, 1, idx.view(b, n*k)).view(b, n, k)

    return torch.abs(dis_mean.unsqueeze(2) - dis_mean_k).mean(-1)

def kNN_smoothing_loss(adv_pc, k, threshold_coef=1.05):
    b,_,n=adv_pc.size()
    dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1) #[b,n,n]
    dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)#[b,n,k+1]

    knn_dis = dis[:, :, 1:].contiguous().mean(-1)#[b,n]
    knn_dis_mean = knn_dis.mean(-1) #[b]
    knn_dis_std = knn_dis.std(-1) #[b]
    threshold = knn_dis_mean + threshold_coef * knn_dis_std #[b]

    condition = torch.gt(knn_dis, threshold.unsqueeze(1)).float() #[b,n]
    dis_mean = knn_dis * condition #[b,n]

    return dis_mean.mean(1) #[b]
