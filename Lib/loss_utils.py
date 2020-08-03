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

def L2NormLoss(adv_pc, ori_pc, norm=2):
    return ((adv_pc - ori_pc)**2 + 1e-10).sum(1).sum(1).sqrt()

def HausdorffLoss(adv_pc, ori_pc):
    dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    return torch.max(torch.min(dis, dim=2)[0], dim=1)[0]

def ChamferLoss(adv_pc, ori_pc):
    dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    return torch.min(dis, dim=2)[0].mean(1)

def _normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def NormalLoss(adv_pc, ori_pc, ori_normal, theta_normal_var, k=2):
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

def ComputeThetaNormal(pc, normal, k):
    b,_,n=pc.size()
    inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    vectors = nn_pts - pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2) #b*n

def DisplacementLoss(adv_pc, ori_pc, k=16):
    b,_,n=adv_pc.size()
    with torch.no_grad():
        inter_dis = ((ori_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
        inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()

    theta_distance = ((adv_pc - ori_pc)**2).sum(1)
    nn_theta_distances = torch.gather(theta_distance, 1, inter_idx.view(b, n*k)).view(b,n,k)
    return ((nn_theta_distances-theta_distance.unsqueeze(2))**2).mean(2)

def CorrespondingNormalLoss(adv_pc, normal, k=2):
    b,_,n=adv_pc.size()

    inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
    inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)
    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2)

def RepulsionLoss(pc, k=4, h=0.03):
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    dis = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)[0][:, :, 1:].contiguous()

    return -(dis * torch.exp(-(dis**2)/(h**2))).mean(2)

def DistancekMeanLoss(pc, k):
    # pdb.set_trace()
    b,_,n=pc.size()
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12)**2).sum(1).sqrt()
    dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)
    dis_mean = dis[:, :, 1:].contiguous().mean(-1) #b*n
    idx = idx[:, :, 1:].contiguous() 
    dis_mean_k = torch.gather(dis_mean, 1, idx.view(b, n*k)).view(b, n, k)

    return torch.abs(dis_mean.unsqueeze(2) - dis_mean_k).mean(-1)

def KNNSmoothingLoss(pc, k, threshold_coef=1.1):
    # pdb.set_trace()
    b,_,n=pc.size()
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12)**2).sum(1) #[b,n,n] 
    dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)#[b,n,k+1] 
    dis_mean = dis[:, :, 1:].contiguous().mean(-1) #[b,n]
    dis_std = dis[:, :, 1:].contiguous().std(-1) #[b,n]
    
    threshold = threshold_coef * dis_std

    dis_mask = torch.gt(dis_mean, threshold).float() #[b,n]
    dis_mean = dis_mean* dis_mask

    return dis_mean.mean(1) #[b]

def jitter_input(data, sigma=0.01, clip=0.05):
    assert data.size(1) == 3
    assert(clip > 0)
    B, _, N = data.size()
    jittered_data = torch.clamp(sigma * torch.randn(B, 3, N), -1*clip, clip).cuda()
    return jittered_data

def estimate_normal(pc, k):
    with torch.no_grad():
        # pc : [b, 3, n]
        b,_,n=pc.size()
        # get knn point set matrix
        dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12)**2).sum(1).sqrt()
        dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)
        idx = idx[:, :, 1:].contiguous()    #idx:[b, n, k]
        nn_pts = torch.gather(pc, 2, idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)   #nn_pts:[b, 3, n, k]
        # get covariance matrix and smallest eig-vector of individual point
        normal_vector = []
        for i in range(b):
            if int(torch.__version__.split('.')[1])>=4:
                curr_point_set = nn_pts[i].detach().permute(1,0,2) #curr_point_set:[n, 3, k]
                curr_point_set_mean = torch.mean(curr_point_set, dim=2, keepdim=True) #curr_point_set_mean:[n, 3, 1]
                curr_point_set = curr_point_set - curr_point_set_mean #curr_point_set:[n, 3, k]
                curr_point_set_t = curr_point_set.permute(0,2,1) #curr_point_set_t:[n, k, 3]
                fact = 1.0 / (k-1)
                cov_mat = fact * torch.bmm(curr_point_set, curr_point_set_t) #curr_point_set_t:[n, 3, 3]
                eigenvalue, eigenvector=torch.symeig(cov_mat, eigenvectors=True)    # eigenvalue:[n, 3], eigenvector:[n, 3, 3]
                persample_normal_vector = torch.gather(eigenvector, 2, torch.argmin(eigenvalue, dim=1).unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_normal_vector:[n, 3]

                #recorrect the direction via neighbour direction
                nbr_sum = curr_point_set.sum(dim=2)  #curr_point_set:[n, 3]
                sign = -torch.sign(torch.bmm(persample_normal_vector.view(n, 1, 3), nbr_sum.view(n, 3, 1))).squeeze(2)
                persample_normal_vector = sign * persample_normal_vector

                normal_vector.append(persample_normal_vector.permute(1,0))

            else:
                persample_normal_vector = []
                for j in range(n):
                    curr_point_set = nn_pts[i,:,j,:].cpu()
                    curr_point_set_np = curr_point_set.detach().numpy()#curr_point_set_np:[3,k]
                    cov_mat_np = np.cov(curr_point_set_np)   #cov_mat:[3,3]
                    eigenvalue_np, eigenvector_np=np.linalg.eig(cov_mat_np)   #eigenvalue:[3], eigenvector:[3,3]; note that v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
                    curr_normal_vector_np = torch.from_numpy(eigenvector_np[:,np.argmin(eigenvalue_np)]) #curr_normal_vector:[3]
                    persample_normal_vector.append(curr_normal_vector_np)
                persample_normal_vector = torch.stack(persample_normal_vector, 1)

                #recorrect the direction via neighbour direction
                nbr_sum = curr_point_set.sum(dim=1)  #curr_point_set:[3]
                sign = -torch.sign(torch.bmm(persample_normal_vector.view(1, 3), nbr_sum.view(3, 1))).squeeze(1)
                persample_normal_vector = sign * persample_normal_vector

                normal_vector.append(persample_normal_vector.permute(1,0))

                normal_vector.append(persample_normal_vector)

        normal_vector = torch.stack(normal_vector, 0) #normal_vector:[b, 3, n]
    return normal_vector.float()

def get_perpendicular_jitter(vector, sigma=0.01, clip=0.05):
    b,_,n=vector.size()
    aux_vector1 = sigma * torch.randn(b,3,n).cuda()
    aux_vector2 = sigma * torch.randn(b,3,n).cuda()
    return torch.clamp(torch.cross(vector, aux_vector1), -1*clip, clip) + torch.clamp(torch.cross(vector, aux_vector2), -1*clip, clip)

def estimate_perpendicular(pc, k, sigma=0.01, clip=0.05):
    with torch.no_grad():
        # pc : [b, 3, n]
        b,_,n=pc.size()
        # get knn point set matrix
        dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12)**2).sum(1).sqrt()
        dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)
        idx = idx[:, :, 1:].contiguous()    #idx:[b, n, k]
        nn_pts = torch.gather(pc, 2, idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)   #nn_pts:[b, 3, n, k]
        # get covariance matrix and smallest eig-vector of individual point
        perpendi_vector_1 = []
        perpendi_vector_2 = []
        for i in range(b):
            curr_point_set = nn_pts[i].detach().permute(1,0,2) #curr_point_set:[n, 3, k]
            curr_point_set_mean = torch.mean(curr_point_set, dim=2, keepdim=True) #curr_point_set_mean:[n, 3, 1]
            curr_point_set = curr_point_set - curr_point_set_mean #curr_point_set:[n, 3, k]
            curr_point_set_t = curr_point_set.permute(0,2,1) #curr_point_set_t:[n, k, 3]
            fact = 1.0 / (k-1)
            cov_mat = fact * torch.bmm(curr_point_set, curr_point_set_t) #curr_point_set_t:[n, 3, 3]
            eigenvalue, eigenvector=torch.symeig(cov_mat, eigenvectors=True)    # eigenvalue:[n, 3], eigenvector:[n, 3, 3]

            larger_dim_idx = torch.topk(eigenvalue, 2, dim=1, largest=True, sorted=False, out=None)[1] # eigenvalue:[n, 2]

            persample_perpendi_vector_1 = torch.gather(eigenvector, 2, larger_dim_idx[:,0].unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_perpendi_vector_1:[n, 3]
            persample_perpendi_vector_2 = torch.gather(eigenvector, 2, larger_dim_idx[:,1].unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_perpendi_vector_2:[n, 3]

            perpendi_vector_1.append(persample_perpendi_vector_1.permute(1,0))
            perpendi_vector_2.append(persample_perpendi_vector_2.permute(1,0))

        perpendi_vector_1 = torch.stack(perpendi_vector_1, 0) #perpendi_vector_1:[b, 3, n]
        perpendi_vector_2 = torch.stack(perpendi_vector_2, 0) #perpendi_vector_1:[b, 3, n]

        aux_vector1 = sigma * torch.randn(b,n).unsqueeze(1).cuda() #aux_vector1:[b, 1, n]
        aux_vector2 = sigma * torch.randn(b,n).unsqueeze(1).cuda() #aux_vector2:[b, 1, n]

    return torch.clamp(perpendi_vector_1*aux_vector1, -1*clip, clip) + torch.clamp(perpendi_vector_2*aux_vector2, -1*clip, clip)
