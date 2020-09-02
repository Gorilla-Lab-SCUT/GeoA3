from __future__ import absolute_import, division, print_function

import argparse
import math
import os
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
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Lib'))

from utility import estimate_normal_via_ori_normal, _compare
from loss_utils import pseudo_chamfer_loss,kNN_smoothing_loss

def random_sample_pointcloud(pc, normal, pc_ori, num_samples):
    # pc:[b,3,n]
    assert pc.size() == normal.size()
    b, _, num_pts = pc.size()
    rind = torch.randperm(num_pts)[:num_samples].unsqueeze(0).unsqueeze(1).expand(b,3,num_samples).cuda()
    rpoints = torch.gather(pc, 2, rind)
    rnormals = torch.gather(normal, 2, rind)
    rpoints_ori = torch.gather(pc_ori, 2, rind)

    return rpoints, rnormals, rpoints_ori, rind

def offset_clipping(offset, normal, cc_linf, project='dir'):
    # offset: shape [b, 3, n], perturbation offset of each point
    # normal: shape [b, 3, n], normal vector of the object

    inner_prod = (offset * normal).sum(1) #[b, n]
    condition_inner = (inner_prod>=0).unsqueeze(1).expand_as(offset) #[b, 3, n]

    if project == 'dir':
        # 1) vng = Normal x Perturb
        # 2) vref = vng x Normal
        # 3) Project Perturb onto vref
        #    Note that the length of vref should be greater than zero

        vng = torch.cross(normal, offset) #[b, 3, n]
        vng_len = (vng**2).sum(1, keepdim=True).sqrt() #[b, n]

        vref = torch.cross(vng, normal) #[b, 3, n]
        vref_len = (vref**2).sum(1, keepdim=True).sqrt() #[b, 1, n]
        vref_len_expand = vref_len.expand_as(offset) #[b, 3, n]

        # add 1e-6 to avoid dividing by zero
        offset_projected = (offset * vref / (vref_len_expand + 1e-6)).sum(1,keepdim=True) * vref / (vref_len_expand + 1e-6)

        # if the length of vng < 1e-6, let projected vector = (0, 0, 0)
        # it means the Normal and Perturb are just in opposite direction
        condition_vng = vng_len > 1e-6
        offset_projected = torch.where(condition_vng, offset_projected, torch.zeros_like(offset_projected))

        # if inner_prod < 0, let perturb be the projected ones
        offset = torch.where(condition_inner, offset, offset_projected)
    else:
        # without projection, let the perturb be (0, 0, 0) if inner_prod < 0
        offset = torch.where(condition_inner, offset, torch.zeros_like(offset))

    # compute vector length
    # if length > cc_linf, clip it
    lengths = (offset**2).sum(1, keepdim=True).sqrt() #[b, 1, n]
    lengths_expand = lengths.expand_as(offset) # [b, 3, n]

    # scale the perturbation vectors to length cc_linf
    # except the ones with zero length
    condition = lengths > 1e-6
    offset_scaled = torch.where(condition, offset / lengths_expand * cc_linf, torch.zeros_like(offset))

    # check the length and clip if necessary
    condition = lengths < cc_linf
    offset = torch.where(condition, offset, offset_scaled)

    return offset

def _forward_step(net, pc_ori, input_curr_iter, target, scale_const, cfg, targeted):
    b,_,n=input_curr_iter.size()
    output_curr_iter = net(input_curr_iter)

    # Logits loss
    if cfg.cls_loss_type == 'Margin':
        target_onehot = torch.zeros(target.size() + (cfg.classes,)).cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)

        fake = (target_onehot * output_curr_iter).sum(1)
        other = ((1. - target_onehot) * output_curr_iter - target_onehot * 10000.).max(1)[0]

        if targeted:
            # if targeted, optimize for making the other class most likely
            cls_loss = torch.clamp(other - fake + cfg.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            cls_loss = torch.clamp(fake - other + cfg.confidence, min=0.)  # equiv to max(..., 0.)

    elif cfg.cls_loss_type == 'CE':
        if targeted:
            cls_loss = nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, target)
        else:
            cls_loss = - nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, target)
    elif cfg.cls_loss_type == 'None':
        cls_loss = torch.FloatTensor(b).zero_().cuda()
    else:
        assert False, 'Not support such clssification loss'

    info = 'cls_loss: {0:6.4f}\t'.format(cls_loss.mean().item())

    # Chamfer pseudo distance (one side)
    dis_loss = pseudo_chamfer_loss(input_curr_iter, pc_ori) #[b]
    constrain_loss = cfg.dis_loss_weight * dis_loss
    info = info + 'cd_loss: {0:6.4f}\t'.format(dis_loss.mean().item())

    # kNN distance loss
    knn_smoothing_loss = kNN_smoothing_loss(input_curr_iter, k=cfg.knn_smoothing_k, threshold_coef=cfg.knn_threshold_coef) #[b]
    constrain_loss = constrain_loss + cfg.knn_smoothing_loss_weight * knn_smoothing_loss
    info = info+'kNN_loss : {0:6.4f}\t'.format(knn_smoothing_loss.mean().item())

    # total loss
    scale_const = scale_const.float().cuda()
    loss_n = cls_loss + scale_const * constrain_loss
    loss = loss_n.mean()

    return output_curr_iter, loss, dis_loss, knn_smoothing_loss, constrain_loss, info

def attack(net, input_data, cfg, i, loader_len):

    if cfg.attack_label == 'Untarget':
        targeted = False
    else:
        targeted = True

    step_print_freq = 50

    pc = input_data[0]
    normal = input_data[1]
    gt_labels = input_data[2]
    if pc.size(3) == 3:
        pc = pc.permute(0,1,3,2)
    if normal.size(3) == 3:
        normal = normal.permute(0,1,3,2)

    bs, l, _, n = pc.size()
    b = bs*l

    pc_ori = pc.view(b, 3, n).cuda()
    normal_ori = normal.view(b, 3, n).cuda()
    gt_target = gt_labels.view(-1)

    if cfg.attack_label == 'Untarget':
        target = gt_target.cuda()
    else:
        target = input_data[3].view(-1).cuda()

    lower_bound = torch.ones(b) * 0
    scale_const = torch.ones(b) * cfg.initial_const
    upper_bound = torch.ones(b) * 1e10

    best_loss = [1e10] * b
    best_attack = torch.ones(b, 3, n).cuda()
    best_attack_step = [-1] * b
    best_attack_BS_idx = [-1] * b

    for search_step in range(cfg.binary_max_steps):
        iter_best_loss = [1e10] * b
        iter_best_score = [-1] * b
        constrain_loss = torch.ones(b) * 1e10

        init_pert = torch.FloatTensor(pc_ori.size())
        nn.init.normal_(init_pert, mean=0, std=1e-3)
        input_all = (pc_ori.clone() + init_pert.cuda())
        input_all.requires_grad_()

        if cfg.optim == 'adam':
            optimizer = optim.Adam([input_all], lr=cfg.lr)
        elif cfg.optim == 'sgd':
            optimizer = optim.SGD([input_all], lr=cfg.lr)
        else:
            assert False, 'Not support such optimizer.'

        for step in range(cfg.iter_max_steps):
            input_curr_iter, normal_curr_iter, pc_ori_curr_iter, rind = random_sample_pointcloud(input_all, normal_ori, pc_ori, num_samples=cfg.npoint)

            with torch.no_grad():
                output = net(input_curr_iter.clone())

                for k in range(b):
                    output_logit = output[k]
                    output_label = torch.argmax(output_logit).item()
                    metric = constrain_loss[k].item()

                    if _compare(output_label, target[k], gt_target[k].cuda(), targeted).item() and (metric <best_loss[k]):
                        best_loss[k] = metric
                        best_attack[k] = input_all.data[k].clone()
                        best_attack_BS_idx[k] = search_step
                        best_attack_step[k] = step
                    if _compare(output_label, target[k], gt_target[k].cuda(), targeted).item() and (metric <iter_best_loss[k]):
                        iter_best_loss[k] = metric
                        iter_best_score[k] = output_label

            output_curr_iter, loss, dis_loss, knn_smoothing_loss, constrain_loss, info = _forward_step(net, pc_ori_curr_iter, input_curr_iter, target, scale_const, cfg, targeted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Perturbation Projection and Clipping
            offset = input_curr_iter - pc_ori_curr_iter
            with torch.no_grad():
                proj_offset = offset_clipping(offset, normal_curr_iter, cfg.cc_linf)
                input_all.data[:,:,rind[0,0]] = (pc_ori_curr_iter + proj_offset).data

            info = '[{5}/{6}][{0}/{1}][{2}/{3}] \t loss: {4:6.4f}\t'.format(search_step+1, cfg.binary_max_steps, step+1, cfg.iter_max_steps, loss.item(), i, loader_len) + info

            if (step+1) % step_print_freq == 0 or step == cfg.iter_max_steps - 1:
                print(info)

        # adjust the scale constants
        for k in range(b):
            if _compare(output_label, target[k], gt_target[k].cuda(), targeted).item() and iter_best_score[k] != -1:
                lower_bound[k] = max(lower_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                else:
                    scale_const[k] *= 2
            else:
                upper_bound[k] = min(upper_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5

    return best_attack, target, (np.array(best_loss)<1e10), best_attack_step  #best_attack:[b, 3, n], target: [b], best_loss:[b], best_attack_step:[b]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust Point Cloud Attacking')
    #------------Model-----------------------
    parser.add_argument('--arch', default='PointNet', type=str, metavar='ARCH', help='')
    #------------Dataset-----------------------
    parser.add_argument('--data_dir_file', default='../Data/modelnet10_250instances_1024.mat', type=str, help='')
    parser.add_argument('--dense_data_dir_file', default='', type=str, help='')
    parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
    parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='B', help='batch_size (default: 2)')
    parser.add_argument('--npoint', default=1024, type=int, help='')
    #------------Attack-----------------------
    parser.add_argument('--attack_label', default='All', type=str, help='[All; ...; Untarget; SingleLabel; RandomTarget]')
    parser.add_argument('--initial_const', type=float, default=10, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--optim', default='adam', type=str, help='adam| sgd')
    parser.add_argument('--binary_max_steps', type=int, default=1, help='')
    parser.add_argument('--iter_max_steps',  default=2500, type=int, metavar='M', help='max steps')
    ## cls loss
    parser.add_argument('--cls_loss_type', default='Margin', type=str, help='Margin | CE')
    parser.add_argument('--confidence', type=float, default=15, help='confidence for margin based attack method')
    ## distance loss
    parser.add_argument('--dis_loss_weight', type=float, default=3.0, help='')
    ## KNN smoothing loss
    parser.add_argument('--knn_smoothing_loss_weight', type=float, default=5.0, help='')
    parser.add_argument('--knn_smoothing_k', type=int, default=5, help='')
    parser.add_argument('--knn_threshold_coef', type=float, default=1.10, help='')
    ## perturbation clip setting
    parser.add_argument('--cc_linf', type=float, default=0.1, help='Coefficient for infinity norm')
    ## eval metric
    parser.add_argument('--metric', default='Loss', type=str, help='[Loss | CDDis | HDDis | CurDis]')
    #------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--is_save_normal', action='store_true', default=False, help='')
    cfg  = parser.parse_args()


    sys.path.append(os.path.join(ROOT_DIR, 'Model'))
    sys.path.append(os.path.join(ROOT_DIR, 'Provider'))

    #data
    from modelnet10_instance250 import ModelNet40
    test_dataset = ModelNet40(data_mat_file=cfg.data_dir_file, attack_label=cfg.attack_label, resample_num=-1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
    test_size = test_dataset.__len__()
    if cfg.dense_data_dir_file != '':
        from modelnet_pure import ModelNet_pure
        dense_test_dataset = ModelNet_pure(data_mat_file=cfg.dense_data_dir_file)
        dense_test_loader = torch.utils.data.DataLoader(dense_test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
    else:
        dense_test_loader = None

    #model
    from PointNet import PointNet
    net = PointNet(cfg.classes, npoint=cfg.npoint).cuda()
    model_path = os.path.join('../Pretrained', 'pointnet_'+str(cfg.npoint)+'.pth.tar')
    log_state_key = 'state_dict'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint[log_state_key])
    net.eval()
    print('\nSuccessfully load pretrained-model from {}\n'.format(model_path))

    saved_root = os.path.join('../Exps', cfg.arch + '_npoint' + str(cfg.npoint))
    saved_dir = 'Test'
    trg_dir = os.path.join(saved_root, cfg.attack_label, saved_dir, 'Mat')
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    trg_dir = os.path.join(saved_root, cfg.attack_label, saved_dir, 'Obj')
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)

    for i, input_data in enumerate(test_loader):
        print('[{0}/{1}]:'.format(i, test_loader.__len__()))
        adv_pc, targeted_label, attack_success_indicator = attack(net, input_data, cfg, i, len(test_loader))
        print(adv_pc.shape)
        break
    print('\n Finish! \n')
