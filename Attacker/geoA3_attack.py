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

from utility import compute_theta_normal, estimate_perpendicular, _compare
from loss_utils import norm_l2_loss, chamfer_loss, hausdorff_loss, normal_loss

def _forward_step(net, pc_ori, input_curr_iter, normal_curr_iter, theta_normal, target, scale_const, cfg, targeted):
    #needed cfg:[arch, classes, cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn]
    b,_,n=input_curr_iter.size()
    output_curr_iter = net(input_curr_iter)

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
            cls_loss = nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
        else:
            cls_loss = - nn.CrossEntropyLoss(reduction='none').cuda()(output_curr_iter, Variable(target, requires_grad=False))
    elif cfg.cls_loss_type == 'None':
        cls_loss = torch.FloatTensor(b).zero_().cuda()
    else:
        assert False, 'Not support such clssification loss'

    info = 'cls_loss: {0:6.4f}\t'.format(cls_loss.mean().item())

    intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1) #b*n*n

    if cfg.dis_loss_type == 'CD':
        if cfg.is_cd_single_side:
            dis_loss = intra_dis.min(2)[0].mean(1)
        else:
            dis_loss = intra_dis.min(2)[0].mean(1) + intra_dis.min(1)[0].mean(1)

        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'cd_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'L2':
        assert cfg.hd_loss_weight ==0
        dis_loss = ((input_curr_iter - pc_ori)**2).sum(1).mean(1)
        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'l2_loss: {0:6.4f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'None':
        dis_loss = 0
        constrain_loss = 0
    else:
        assert False, 'Not support such distance loss'

    # hd_loss
    if cfg.hd_loss_weight !=0:
        hd_loss = intra_dis.min(2)[0].max(1)[0]
        constrain_loss = constrain_loss + cfg.hd_loss_weight * hd_loss
        info = info+'hd_loss : {0:6.4f}\t'.format(hd_loss.mean().item())
    else:
        hd_loss = 0

    # nor loss
    if cfg.curv_loss_weight !=0:
        curv_loss,_ = normal_loss(input_curr_iter, pc_ori, normal_curr_iter, None, cfg.curv_loss_knn)

        intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
        intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
        knn_theta_normal = torch.gather(theta_normal, 1, intra_idx.view(b,n).expand(b,n))
        curv_loss = ((curv_loss - knn_theta_normal)**2).mean(-1)

        constrain_loss = constrain_loss + cfg.curv_loss_weight * curv_loss
        info = info+'curv_loss : {0:6.4f}\t'.format(curv_loss.mean().item())
    else:
        curv_loss = 0

    scale_const = scale_const.float().cuda()
    loss_n = cls_loss + scale_const * constrain_loss
    loss = loss_n.mean()

    return output_curr_iter, loss, loss_n, cls_loss, dis_loss, hd_loss, curv_loss, constrain_loss, info

def attack(net, input_data, cfg, i, loader_len):
    #needed cfg:[arch, classes, attack_label, initial_const, lr, optim, binary_max_steps, iter_max_steps, metric,
    #  cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn,
    #  is_pre_jitter_input, calculate_project_jitter_noise_iter, jitter_k, jitter_sigma, jitter_clip,
    #  is_save_normal,
    #  ]

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

    if cfg.curv_loss_weight !=0:
        theta_normal = compute_theta_normal(pc_ori, normal_ori, cfg.curv_loss_knn)
    else:
        theta_normal = None

    lower_bound = torch.ones(b) * 0
    scale_const = torch.ones(b) * cfg.initial_const
    upper_bound = torch.ones(b) * 1e10

    best_loss = [1e10] * b
    best_attack = torch.ones(b, 3, n).cuda()
    best_attack_step = [-1] * b
    best_attack_BS_idx = [-1] * b
    all_loss_list = [[-1] * b] * cfg.iter_max_steps
    #dis_loss_hist = [[-1] * b] * cfg.iter_max_steps
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
            input_curr_iter = input_all
            normal_curr_iter = normal_ori

            with torch.no_grad():
                output = net(input_all.clone())

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

            if cfg.is_pre_jitter_input:
                if step % cfg.calculate_project_jitter_noise_iter == 0:
                    project_jitter_noise = estimate_perpendicular(input_curr_iter, cfg.jitter_k, sigma=cfg.jitter_sigma, clip=cfg.jitter_clip)
                else:
                    project_jitter_noise = project_jitter_noise.clone()
                input_curr_iter.data  = input_curr_iter.data  + project_jitter_noise

            _, loss, loss_n, cls_loss, dis_loss, hd_loss, nor_loss, constrain_loss, info = _forward_step(net, pc_ori, input_curr_iter, normal_curr_iter, theta_normal, target, scale_const, cfg, targeted)

            all_loss_list[step] = loss_n.detach().tolist()
            #dis_loss_hist[step] = cls_loss.detach().tolist()

            optimizer.zero_grad()
            if cfg.is_pre_jitter_input:
                input_curr_iter.retain_grad()
            loss.backward()
            if cfg.is_pre_jitter_input:
                input_all.grad = input_curr_iter.grad
            optimizer.step()

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

    return best_attack, target, (np.array(best_loss)<1e10), best_attack_step, all_loss_list  #best_attack:[b, 3, n], target: [b], best_loss:[b], best_attack_step:[b], all_loss_list:[iter_max_steps, b]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEOA3 Point Cloud Attacking')
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
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--optim', default='adam', type=str, help='adam| sgd')
    parser.add_argument('--binary_max_steps', type=int, default=10, help='')
    parser.add_argument('--iter_max_steps',  default=500, type=int, metavar='M', help='max steps')
    ## cls loss
    parser.add_argument('--cls_loss_type', default='CE', type=str, help='Margin | CE')
    parser.add_argument('--confidence', type=float, default=0, help='confidence for margin based attack method')
    ## distance loss
    parser.add_argument('--dis_loss_type', default='CD', type=str, help='CD | L2 | None')
    parser.add_argument('--dis_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--is_cd_single_side', action='store_true', default=False, help='')
    ## hausdorff loss
    parser.add_argument('--hd_loss_weight', type=float, default=0.1, help='')
    ## normal loss
    parser.add_argument('--curv_loss_weight', type=float, default=0.1, help='')
    parser.add_argument('--curv_loss_knn', type=int, default=16, help='')
    ## eval metric
    parser.add_argument('--metric', default='Loss', type=str, help='[Loss | CDDis | HDDis | CurDis]')
    ## Jitter
    parser.add_argument('--is_pre_jitter_input', action='store_true', default=False, help='')
    parser.add_argument('--calculate_project_jitter_noise_iter', default=50, type=int,help='')
    parser.add_argument('--jitter_k', type=int, default=16, help='')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help='')
    parser.add_argument('--jitter_clip', type=float, default=0.05, help='')
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
