from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from Lib.utility import farthest_points_sample


def random_drop_fn(pc, drop_num):
    n = pc.size(2)
    idx = torch.randperm(n)[drop_num:].long().cuda()
    idx = torch.sort(idx, dim=0,descending=False)[0]
    # pdb.set_trace()
    return pc.clone()[:, :, idx].contiguous(), drop_num

def outlier_removal_fn(pc, defense_type, drop_num, alpha, outlier_knn):
    dis = (pc.unsqueeze(2)-pc.unsqueeze(3)+1e-10).pow(2).sum(dim=1).sqrt()
    dis = dis.topk(outlier_knn+1,dim=2,largest=False, sorted=True)[0][:, :, 1:].contiguous().mean(dim=-1)
    n = pc.size(2)

    if defense_type == 'outliers_variance':
        dis_mean = dis.mean(-1)
        dis_std = dis.std(-1)
        keep_mask = dis<(dis_mean + alpha*dis_std).unsqueeze(-1)
        output_pc = torch.masked_select(pc[0],keep_mask[0].unsqueeze(0).expand_as(pc[0])).view(1,3,-1)
        return output_pc, pc.size(2)-output_pc.size(2)
    elif defense_type == 'outliers_fixNum':
        # pdb.set_trace()
        idx = dis.topk(n-drop_num,dim=1,largest=False, sorted=True)[1].view(-1)
        idx = torch.sort(idx, dim=0,descending=False)[0]
        return pc.clone()[:, :, idx].contiguous(), n-idx.size(0)

def point_removal_fn(pc, defense_type, drop_num, alpha, outlier_knn):
    if defense_type == 'rand_drop':
        output_pc, num = random_drop_fn(pc, drop_num)
    elif defense_type == 'outliers_variance' or defense_type == 'outliers_fixNum':
        output_pc, num = outlier_removal_fn(pc, defense_type, drop_num, alpha, outlier_knn)
    else:
        assert False, 'Wrong defense type!'

    return output_pc, num

def main():
    if cfg.random_seed == 0:
        seed = cfg.random_seed
    else:
        seed = time.time()

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data
    test_dataset = ModelNet40(cfg.datadir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=True)
    test_size = test_dataset.__len__()

    # model
    model_path = os.path.join('Pretrained', cfg.arch, str(cfg.npoint), 'model_best.pth.tar')
    if cfg.arch == 'PointNet':
        from Model.PointNet import PointNet
        net = PointNet(cfg.classes, npoint=cfg.npoint).cuda()
    else:
        assert False, 'Not support such arch.'

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('\nSuccessfully load pretrained-model from {}\n'.format(model_path))

    cnt = 0
    num_defense_success =0
    num_attack_still_success =0
    num_drop_point = 0

    for i, (adv_pc, gt_label, attack_label) in enumerate(test_loader):
        b = adv_pc.size(0)
        assert b == 1
        cnt += 1

        if adv_pc.size(2) > cfg.npoint:
            #adv_pc = adv_pc[:,:,:cfg.npoint]
            adv_pc = farthest_points_sample(adv_pc.cuda(), cfg.npoint)

        with torch.no_grad():
            defense_pc, num = point_removal_fn(adv_pc.cuda(), cfg.defense_type, cfg.drop_num, cfg.alpha, cfg.outlier_knn)
            defense_pc_var = Variable(defense_pc)
            defense_output = net(defense_pc)

        if gt_label.view(-1) == attack_label.view(-1):
            defense_success = 1
            attack_still_success = 0
        else:
            defense_success = (torch.max(defense_output,1)[1].data.cpu() == gt_label.view(-1)).sum()
            attack_still_success = (torch.max(defense_output,1)[1].data.cpu() == attack_label.view(-1)).sum()
        num_defense_success += defense_success
        num_attack_still_success += attack_still_success
        num_drop_point += num

        saved_pc = defense_pc_var.data[0].clone().cpu().permute(1, 0).clone().numpy()

        if cfg.is_record_all:
            fout = open(os.path.join(os.path.split(cfg.datadir)[0], 'Defensed', 'Gt' + str(gt_label[0].item()) + '_record_' + str(i) + '_attack' + str(attack_label[0].item()) + '_defensedGT' + str(torch.max(defense_output,1)[1].data.item())+'.obj'), 'w')
            for m in range(saved_pc.shape[0]):
                fout.write('v %f %f %f 0 0 0\n' % (saved_pc[m, 0], saved_pc[m, 1], saved_pc[m, 2]))
            fout.close()
        elif cfg.is_record_wrong:
            if gt_label[0].item() != torch.max(defense_output,1)[1].data.item():
                fout = open(os.path.join(os.path.split(cfg.datadir)[0], 'Defensed', 'Gt' + str(gt_label[0].item()) + '_record_' + str(i) + '_attack' + str(attack_label[0].item()) + '_defensedGT' + str(torch.max(defense_output,1)[1].data.item())+'.obj'), 'w')
                for m in range(saved_pc.shape[0]):
                    fout.write('v %f %f %f 0 0 0\n' % (saved_pc[m, 0], saved_pc[m, 1], saved_pc[m, 2]))
                fout.close()

        if (i+1) % cfg.print_freq == 0:
            print('[{0}/{1}]  attack success: {2:.2f} still attack success: {3:.2f} avg drop num: {4:.2f}'.format(
                i+1, len(test_loader), (1-num_defense_success.item()/float(cnt))*100, num_attack_still_success.item()/float(cnt)*100,num_drop_point/float(cnt)))


    final_acc = num_defense_success.item()/float(test_loader.dataset.__len__())*100
    final_attack_acc = num_attack_still_success.item()/float(test_loader.dataset.__len__())*100
    avg_drop_point = num_drop_point/float(test_loader.dataset.__len__())
    assert 100-final_acc >= final_attack_acc, "Attack success must > or >= attack still success!"
    print('\nfinal attack success: {0:.2f}\n still attack success: {1:.2f}\n avg drop point: {2:.2f}'.format(100-final_acc, final_attack_acc, avg_drop_point))

    with open(os.path.join(os.path.split(cfg.datadir)[0],  'defense_result.txt'), 'at') as f:
        if cfg.defense_type == 'rand_drop':
            f.write('[{0:.2f}%, {1:.2f}%, {2:.2f}n] random drop: drop_num {3}\n'.format(final_acc, final_attack_acc, avg_drop_point, cfg.drop_num))
        elif cfg.defense_type == 'outliers_variance':
            f.write('[{0:.2f}%, {1:.2f}%, {2:.2f}n] outlier alpha removal: k{3}, alpha{4}\n'.format(
                final_acc, final_attack_acc, avg_drop_point, cfg.outlier_knn, cfg.alpha))
        elif cfg.defense_type == 'outliers_fixNum':
            f.write('[{0:.2f}%, {1:.2f}%, {2:.2f}n] outlier ramdom drop: drop_num {3}\n'.format(final_acc, final_attack_acc, avg_drop_point, cfg.drop_num))
        else:
            assert False

    print('\n Finished!')


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'Model'))
    sys.path.append(os.path.join(ROOT_DIR, 'Lib'))
    sys.path.append(os.path.join(ROOT_DIR, 'Provider'))
    from Lib.loss_utils import *
    from Provider.defense_modelnet10_instance250 import ModelNet40

    parser = argparse.ArgumentParser(description='Point Cloud Defense')
    #------------Dataset-----------------------
    parser.add_argument('--datadir', default='Data/modelnet40_1024_processed', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--npoint', default=1024, type=int, help='')
    parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
    #------------Model-----------------------
    parser.add_argument('--arch', default='PointNet', type=str, metavar='ARCH', help='')
    parser.add_argument('--defense_type', default='outliers_fixNum', type=str, help='[rand_drop, outliers_variance, outliers_fixNum]')
    #------------Defense-----------------------
    # outlier removal
    parser.add_argument('--outlier_knn', type=int, default=2, help='')
    parser.add_argument('--alpha', type=float, default=1.1, help='')
    parser.add_argument('--drop_num', type=int, default=128, help='')
    parser.add_argument('--is_record_all', action='store_true', default=False, help='')
    parser.add_argument('--is_record_wrong', action='store_true', default=False, help='')
    #------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--random_seed', default=0, type=int, help='')
    parser.add_argument('--print_freq', default=50, type=int, help='')

    cfg  = parser.parse_args()
    print(cfg)

    assert cfg.datadir[-1] != '/'

    if cfg.is_record_all or cfg.is_record_wrong:
        if not os.path.exists(os.path.join(os.path.split(cfg.datadir)[0], 'Defensed')):
            os.mkdir(os.path.join(os.path.split(cfg.datadir)[0], 'Defensed'))

    main()
