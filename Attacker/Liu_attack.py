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

from utility import _compare

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class Cross_entropy_loss_onehot(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, nClass = 10):
        super(Cross_entropy_loss_onehot, self).__init__(weight, size_average)
        self.nClass = nClass

    def forward(self, input, target):
        target_onehot = torch.zeros(target.size() + (self.nClass,)).cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)

        target_var = Variable(target_onehot, requires_grad=False)
        _assert_no_grad(target_var)
        loss = torch.nn.functional.softmax(input, 1).mul(target_var).sum(1)
        return - loss.log()


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

    best_loss = [1e10] * b
    best_attack = torch.ones(b, 3, n).cuda()
    best_attack_step = [-1] * b

    init_pert = torch.FloatTensor(pc_ori.size())
    nn.init.normal_(init_pert, mean=0, std=1e-3)
    input_all = (pc_ori.clone() + init_pert.cuda())
    input_all.requires_grad_()

    continute_mask = torch.ones(b).byte().cuda()

    #FIXME: how is the normalizing method here should be?
    step_alpha = cfg.step_alpha/(255.0)

    for step in range(cfg.iter_max_steps):
        input_curr_iter = input_all
        normal_curr_iter = normal_ori

        with torch.no_grad():
            output = net(input_all.clone())

            for k in range(b):
                output_logit = output[k]
                output_label = torch.argmax(output_logit).item()
                try:
                    metric = modifier.norm(p=2,dim=1).mean(1)[k].item()
                except:
                    metric = (torch.ones(b) * 1e10)[k]

                if _compare(output_label, target[k], gt_target[k].cuda(), targeted).item() and (metric <best_loss[k]):
                    best_loss[k] = metric
                    best_attack[k] = input_all.data[k].clone()
                    best_attack_step[k] = step

        zero_gradients(input_all)
        output = net(input_all)
        loss = Cross_entropy_loss_onehot(nClass = cfg.classes)(output, target)

        loss.backward(continute_mask.float())

        # the following equals to optimizer.step()
        l2_norm_grad = input_all.grad.detach().view(b, -1).norm(p=2, dim =1)
        normed_grad = step_alpha * input_all.grad.detach() / (l2_norm_grad.unsqueeze(1).unsqueeze(2)+1e-12)

        if targeted:
            step_adv = torch.clamp(input_all.detach() - normed_grad, -1, 1)
        else:
            step_adv = torch.clamp(input_all.detach() + normed_grad, -1, 1)

        modifier = step_adv - pc_ori

        input_all.data = torch.clamp(pc_ori + modifier, -1, 1).data

        with torch.no_grad():
            test_adv = Variable(input_all)
            adv_output = net(test_adv)

            if targeted:
                continute_mask = torch.max(adv_output,1)[1] != target
            else:
                continute_mask = torch.max(adv_output,1)[1] == target

        info = '[{0}/{1}][{2}/{3}] loss: {4:6.4f}\t'.format(i, loader_len, step+1, cfg.iter_max_steps, loss.mean().item())

        if (step+1) % step_print_freq == 0 or step == cfg.iter_max_steps - 1:
            print(info)

    return best_attack, target, (np.array(best_loss)<1e10), best_attack_step  #best_attack:[b, 3, n], target: [b], best_loss:[b], best_attack_step:[b]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Liu\'s method (PDG-like) Point Cloud Attacking')
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
    parser.add_argument('--iter_max_steps',  default=500, type=int, metavar='M', help='max steps')
    parser.add_argument('--step_alpha', type=float, default=5, help='')
    #------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    cfg  = parser.parse_args()

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

