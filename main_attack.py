from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.io as sio
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from Attacker import geoA3_attack, Xiang_attack, robust_attack, Liu_attack, geoA3_mesh_attack
from Lib.utility import estimate_normal_via_ori_normal, _compare, farthest_points_sample, Count_converge_iter

ten_label_indexes = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']
'''
ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']
'''
parser = argparse.ArgumentParser(description='Point Cloud Attacking')
#------------Model-----------------------
parser.add_argument('--id', type=int, default=0, help='')
parser.add_argument('--arch', default='PointNet', type=str, metavar='ARCH', help='')
#------------Dataset-----------------------
parser.add_argument('--data_dir_file', default='Data/modelnet10_250instances1024_PointNet.mat', type=str, help='')
parser.add_argument('--dense_data_dir_file', default=None, type=str, help='')
parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='B', help='batch_size (default: 2)')
parser.add_argument('--npoint', default=1024, type=int, help='')
#------------Attack-----------------------
parser.add_argument('--attack', default=None, type=str, help='GeoA3 | Xiang | RA | Liu | GeoA3_mesh')
parser.add_argument('--attack_label', default='All', type=str, help='[All; ...; Untarget]')
parser.add_argument('--binary_max_steps', type=int, default=10, help='')
parser.add_argument('--initial_const', type=float, default=10, help='')
parser.add_argument('--iter_max_steps',  default=500, type=int, metavar='M', help='max steps')
parser.add_argument('--optim', default='adam', type=str, help='adam| sgd')
parser.add_argument('--lr', type=float, default=0.01, help='')
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
parser.add_argument('--curv_loss_weight', type=float, default=1.0, help='')
parser.add_argument('--curv_loss_knn', type=int, default=16, help='')
## KNN smoothing loss
parser.add_argument('--knn_smoothing_loss_weight', type=float, default=5.0, help='')
parser.add_argument('--knn_smoothing_k', type=int, default=5, help='')
parser.add_argument('--knn_threshold_coef', type=float, default=1.10, help='')
## perturbation clip setting
parser.add_argument('--cc_linf', type=float, default=0.1, help='Coefficient for infinity norm')
## Jitter
parser.add_argument('--is_pre_jitter_input', action='store_true', default=False, help='')
parser.add_argument('--is_previous_jitter_input', action='store_true', default=False, help='')
parser.add_argument('--calculate_project_jitter_noise_iter', default=50, type=int,help='')
parser.add_argument('--jitter_k', type=int, default=16, help='')
parser.add_argument('--jitter_sigma', type=float, default=0.01, help='')
parser.add_argument('--jitter_clip', type=float, default=0.05, help='')
## PGD-like attack
parser.add_argument('--step_alpha', type=float, default=5, help='')
#------------Recording settings-------
parser.add_argument('--is_record_converged_steps', action='store_true', default=False, help='')
#------------OS-----------------------
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--is_save_normal', action='store_true', default=False, help='')
parser.add_argument('--is_debug', action='store_true', default=False, help='')

cfg  = parser.parse_args()
print(cfg, '\n')

if cfg.attack_label == 'Untarget':
    targeted = False
else:
    targeted = True

saved_root = os.path.join('Exps', cfg.arch + '_npoint' + str(cfg.npoint))

if cfg.attack == 'GeoA3' or cfg.attack == 'Xiang' or cfg.attack == 'RA' or cfg.attack == 'GeoA3_mesh':
    saved_dir = str(cfg.attack) + '_' +  str(cfg.id) +  '_BiStep' + str(cfg.binary_max_steps) + '_IterStep' + str(cfg.iter_max_steps) + '_Opt' + cfg.optim  +  '_Lr' + str(cfg.lr) + '_Initcons' + str(cfg.initial_const) + '_' + cfg.cls_loss_type + '_' + str(cfg.dis_loss_type) + 'Loss' + str(cfg.dis_loss_weight)

    if cfg.attack == 'GeoA3' or cfg.attack == 'GeoA3_mesh':
        if cfg.hd_loss_weight != 0:
            saved_dir = saved_dir + '_HDLoss' + str(cfg.hd_loss_weight)

        if cfg.curv_loss_weight != 0:
            saved_dir = saved_dir + '_CurLoss' + str(cfg.curv_loss_weight) + '_k' + str(cfg.curv_loss_knn)

        if cfg.is_pre_jitter_input:
            saved_dir = saved_dir + '_PreJitter' + str(cfg.jitter_sigma) + '_' + str(cfg.jitter_clip)
            if cfg.is_previous_jitter_input:
                saved_dir = saved_dir + '_PreviousMethod'
            else:
                saved_dir = saved_dir + '_estNormalVery' + str(cfg.calculate_project_jitter_noise_iter)

    if cfg.attack == 'RA':
        if cfg.knn_smoothing_loss_weight != 0:
            saved_dir = saved_dir + '_KnnLoss' + str(cfg.knn_smoothing_loss_weight) + '_k' + str(cfg.knn_smoothing_k) + '_coe' + str(cfg.knn_threshold_coef)
        if cfg.cc_linf != 0:
            saved_dir = saved_dir + '_cclinf' + str(cfg.cc_linf)

elif cfg.attack == 'Liu':
    saved_dir = str(cfg.attack) + '_' +  str(cfg.id) + '_IterStep' + str(cfg.iter_max_steps) + '_StepAlpha' + str(cfg.step_alpha)

else:
    assert cfg.attack == None
    saved_dir = 'Evaluating_' + str(cfg.id)

saved_dir = os.path.join(saved_root, cfg.attack_label, saved_dir)

print(saved_dir)

if cfg.attack == 'GeoA3_mesh':
    trg_dir = os.path.join(saved_dir, 'Mesh')
else:
    trg_dir = os.path.join(saved_dir, 'Obj')
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)
trg_dir = os.path.join(saved_dir, 'Mat')
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)
trg_dir = os.path.join(saved_dir, 'Records')
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)

class Average_meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    if cfg.id == 0:
        seed = 0
    else:
        seed = int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data
    if cfg.attack == 'GeoA3_mesh':
        from Provider.modelnet10_instance250_mesh import ModelNet10_250instance_mesh
        test_dataset = ModelNet10_250instance_mesh(resume=cfg.data_dir_file, attack_label= cfg.attack_label)
    else:
        if (str(cfg.npoint) in cfg.data_dir_file):
            resample_num = -1
        else:
            resample_num = cfg.npoint

        from Provider.modelnet10_instance250 import ModelNet40
        test_dataset = ModelNet40(data_mat_file=cfg.data_dir_file, attack_label=cfg.attack_label, resample_num=resample_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
    test_size = test_dataset.__len__()

    if (cfg.is_save_normal) & (cfg.dense_data_dir_file is not None):
        # from Provider.modelnet_pure import ModelNet_pure
        # dense_test_dataset = ModelNet_pure(data_mat_file=cfg.dense_data_dir_file)
        dense_test_dataset = ModelNet40(data_mat_file=cfg.dense_data_dir_file, attack_label=cfg.attack_label, resample_num=-1)
        dense_test_loader = torch.utils.data.DataLoader(dense_test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
        dense_test_size = dense_test_dataset.__len__()
        dense_iter = iter(dense_test_loader)
    else:
        dense_iter = None

    # model
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
    print('\nSuccessfully load pretrained-model from {}\n'.format(model_path))

    # recording settings
    if cfg.is_record_converged_steps:
        cci = Count_converge_iter(os.path.join(saved_dir, 'Records'))

    test_acc = Average_meter()
    batch_vertice = []
    batch_faces_idx = []
    batch_gt_label = []

    num_attack_success = 0
    cnt_ins = test_dataset.start_index
    cnt_all = 0

    if cfg.attack_label == 'Untarget':
        targeted = False
        num_attack_classes = 1
    else:
        targeted = True
        num_attack_classes = 9

    for i, data in enumerate(test_loader):
        #print('[{0}/{1}]:'.format(i, test_loader.__len__()), end='')
        if cfg.attack == 'GeoA3_mesh':
            _, _, gt_label = data[0], data[1], data[2]
            gt_target = gt_label.view(-1).cuda()
        else:
            pc = data[0]
            normal = data[1]
            gt_labels = data[2]
            if pc.size(3) == 3:
                pc = pc.permute(0,1,3,2)
            if normal.size(3) == 3:
                normal = normal.permute(0,1,3,2)

            bs, l, _, n = pc.size()
            b = bs*l

            pc = pc.view(b, 3, n).cuda()
            normal = normal.view(b, 3, n).cuda()
            gt_target = gt_labels.view(-1).cuda()

            if dense_iter is not None:
                dense_data = dense_iter.next()
                dense_point = dense_data[0]
                dense_normal = dense_data[1]

                if dense_point.size(3) == 3:
                    dense_point = dense_point.permute(0,1,3,2)
                if dense_normal.size(3) == 3:
                    dense_normal = dense_normal.permute(0,1,3,2)

                bs, l, _, n = dense_point.size()
                b = bs*l

                dense_point = dense_point.view(b, 3, n).cuda()
                dense_normal = dense_normal.view(b, 3, n).cuda()

        if cfg.attack is None:
            with torch.no_grad():
                output = net(pc)
            acc = accuracy(output.data, gt_target.data, topk=(1, ))
            test_acc.update(acc[0][0], output.size(0))
            print("Prec@1 {top1.avg:.3f}".format(top1=test_acc))

        elif cfg.attack == 'GeoA3':
            adv_pc, targeted_label, attack_success_indicator, best_attack_step = geoA3_attack.attack(net, data, cfg, i, len(test_loader))
            eval_num = 1
        elif cfg.attack == 'Xiang':
            adv_pc, targeted_label, attack_success_indicator, best_attack_step = Xiang_attack.attack(net, data, cfg, i, len(test_loader))
            eval_num = 1
        elif cfg.attack == 'RA':
            adv_pc, targeted_label, attack_success_indicator, best_attack_step = robust_attack.attack(net, dense_data, cfg, i, len(test_loader))
            eval_num = 16
        elif cfg.attack == 'Liu':
            adv_pc, targeted_label, attack_success_indicator, best_attack_step = Liu_attack.attack(net, data, cfg, i, len(test_loader))
            eval_num = 1
        elif cfg.attack == 'GeoA3_mesh':
            adv_mesh, targeted_label, attack_success_indicator, best_attack_step, best_score = geoA3_mesh_attack.attack(net, data, cfg, i, len(test_loader), saved_dir)
            eval_num = 1
        else:
            assert False, "Wrong type of attack."

        if cfg.attack == 'GeoA3' or cfg.attack == 'RA' or cfg.attack == 'Xiang' or cfg.attack == 'Liu':
            if cfg.is_record_converged_steps:
                cci.record_converge_iter(best_attack_step)
            if cfg.is_save_normal:
                with torch.no_grad():
                    # the loop here is for memory save
                    knn_normal = torch.zeros_like(dense_normal)
                    for idx in range(b):
                        knn_normal[idx] = estimate_normal_via_ori_normal(adv_pc[idx].unsqueeze(0), dense_point[idx].unsqueeze(0), dense_normal[idx].unsqueeze(0), k=3)
                saved_normal = knn_normal.cpu().numpy()

            for _ in range(0,eval_num):
                with torch.no_grad():
                    if adv_pc.size(2) > cfg.npoint:
                        eval_points = farthest_points_sample(adv_pc, cfg.npoint)
                    else:
                        eval_points = adv_pc
                    test_adv_output = net(eval_points)
                attack_success_iter = _compare(torch.max(test_adv_output,1)[1].data, targeted_label, gt_target.cuda(), targeted)

                try:
                    attack_success += attack_success_iter
                except:
                    attack_success = attack_success_iter
            saved_pc = adv_pc.cpu().clone().numpy()

            for k in range(b):
                if attack_success[k].item() and attack_success_indicator[k]:
                    num_attack_success += 1
                    name = 'adv_' + str(cnt_ins+k//num_attack_classes) + '_gt' + str(gt_target[k].item()) + '_attack' + str(torch.max(test_adv_output,1)[1].data[k].item())

                    if cfg.is_save_normal:
                        sio.savemat(os.path.join(saved_dir, 'Mat', name+'.mat'),
                        {"adversary_point_clouds": saved_pc[k], 'gt_label': gt_target[k].item(), 'attack_label': torch.max(test_adv_output,1)[1].data[k].item(), 'est_normal':saved_normal[k]})
                    else:
                        sio.savemat(os.path.join(saved_dir, 'Mat', name+'.mat'),
                        {"adversary_point_clouds": saved_pc[k], 'gt_label': gt_target[k].item(), 'attack_label': torch.max(test_adv_output,1)[1].data[k].item()})

                    fout = open(os.path.join(saved_dir, 'Obj', name+'.obj'), 'w')
                    for m in range(saved_pc.shape[2]):
                        fout.write('v %f %f %f 0 0 0\n' % (saved_pc[k, 0, m], saved_pc[k, 1, m], saved_pc[k, 2, m]))
                    fout.close()

            cnt_ins = cnt_ins + bs
            cnt_all = cnt_all + b
        elif cfg.attack == 'GeoA3_mesh':
            if attack_success_indicator[k].item() and best_score[k] != -1:
                num_attack_success += 1
                name = 'adv_' + str(cnt_ins+k//num_attack_classes) + '_gt' + str(gt_target[k].item()) + '_attack' + str(best_score[k])
                final_verts, final_faces = adv_mesh[k].get_mesh_verts_faces(0)
                #save .mat
                sio.savemat(os.path.join(saved_dir, 'Mat', name+'.mat'), {"vert": final_verts, "faces":final_faces})
                #save .obj mesh
                file_name = os.path.join(saved_dir, 'Mesh', name+'.obj')
                save_obj(file_name, final_verts, final_faces)

            cnt_ins = cnt_ins + bs
            cnt_all = cnt_all + b

    if cfg.is_record_converged_steps:
        cci.save_converge_iter()
        cci.plot_converge_iter_hist()

    if cfg.attack == 'GeoA3' or cfg.attack == 'RA' or cfg.attack == 'Xiang' or cfg.attack == 'Liu':
        print('attack success: {0:.2f}\n'.format(num_attack_success/float(cnt_all)*100))
        with open(os.path.join(saved_dir, 'attack_result.txt'), 'at') as f:
            f.write('attack success: {0:.2f}\n'.format(num_attack_success/float(cnt_all)*100))
        print('saved_dir: {0}'.format(os.path.join(saved_dir)))

    print('Finish!')


if __name__ == '__main__':
    main()
