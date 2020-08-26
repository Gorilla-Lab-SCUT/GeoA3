from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.io as sio
from pytorch3d.io import load_obj, save_obj
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Lib'))

from utility import compute_theta_normal, estimate_normal, estimate_perpendicular, _compare, pad_larger_tensor_with_index
from loss_utils import chamfer_loss, hausdorff_loss, normal_loss

def _forward_step(net, pc_ori, input_curr_iter, normal_curr_iter, theta_normal, new_src_mesh, target, scale_const, cfg, targeted):
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

    info = 'cls_loss: {0:6.3f}\t'.format(cls_loss.mean().item())

    intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1) #b*n*n

    if cfg.dis_loss_type == 'CD':
        if cfg.is_cd_single_side:
            dis_loss = intra_dis.min(2)[0].mean(1)
        else:
            dis_loss = intra_dis.min(2)[0].mean(1) + intra_dis.min(1)[0].mean(1)

        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'cd_loss: {0:6.3f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'L2':
        assert cfg.hd_loss_weight ==0
        dis_loss = ((input_curr_iter - pc_ori)**2).sum(1).mean(1)
        constrain_loss = cfg.dis_loss_weight * dis_loss
        info = info + 'l2_loss: {0:6.3f}\t'.format(dis_loss.mean().item())
    elif cfg.dis_loss_type == 'None':
        dis_loss = 0
        constrain_loss = 0
    else:
        assert False, 'Not support such distance loss'

    # hd_loss
    if cfg.hd_loss_weight != 0:
        hd_loss = intra_dis.min(2)[0].max(1)[0]
        constrain_loss = constrain_loss + cfg.hd_loss_weight * hd_loss
        info = info+'hd_loss : {0:6.3f}\t'.format(hd_loss.mean().item())
    else:
        hd_loss = 0

    # nor loss
    if cfg.curv_loss_weight != 0:
        curv_loss,_ = normal_loss(input_curr_iter, pc_ori, normal_curr_iter, None, cfg.curv_loss_knn)

        intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
        intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
        knn_theta_normal = torch.gather(theta_normal, 1, intra_idx.view(b,n).expand(b,n))
        curv_loss = ((curv_loss - knn_theta_normal)**2).mean(-1)

        constrain_loss = constrain_loss + cfg.curv_loss_weight * curv_loss
        info = info+'curv_loss : {0:6.3f}\t'.format(curv_loss.mean().item())
    else:
        curv_loss = 0

    if cfg.laplacian_loss_weight != 0:
        laplacian_loss = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        constrain_loss = constrain_loss + cfg.laplacian_loss_weight * laplacian_loss
        info = info+'Laplacian_loss : {0:6.3f} '.format(cfg.laplacian_loss_weight * (laplacian_loss).mean().item())

    if cfg.edge_loss_weight != 0:
        edge_loss = mesh_edge_loss(new_src_mesh)
        constrain_loss = constrain_loss + cfg.edge_loss_weight * edge_loss
        info = info+'MeshEG_loss : {0:6.3f} | '.format(cfg.edge_loss_weight * (edge_loss).mean().item())

    scale_const = scale_const.float().cuda()
    loss_n = cls_loss + scale_const * constrain_loss
    loss = loss_n.mean()

    return output_curr_iter, loss, dis_loss, hd_loss, curv_loss, constrain_loss, info

def attack(net, input_data, cfg, i, loader_len, saved_dir):
    #needed cfg:[arch, classes, attack_label, initial_const, lr, optim, binary_max_steps, iter_max_steps, metric,
    #  cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn,
    #  is_pre_jitter_input, calculate_project_jitter_noise_iter, jitter_k, jitter_sigma, jitter_clip,
    #  is_save_normal,
    #  ]
    assert cfg.batch_size == 1
    eval_num = 50
    step_print_freq = 50

    if cfg.attack_label == 'Untarget':
        targeted = False
    else:
        targeted = True

    vertice, faces_idx, gt_label = input_data[0], input_data[1], input_data[2]

    bs, l, vn, _ = vertice.size()
    _, _, fn, _ = faces_idx.size()
    b = bs*l
    #b = cfg.batch_size * num_attack_classes
    vertice = vertice.view(b, vn, 3).cuda()
    faces_idx = faces_idx.view(b, fn, 3).cuda()
    gt_target = gt_label.cuda()

    if cfg.attack_label == 'Untarget':
        num_attack_classes = 1
        # attack_label:[1]
        target = gt_label.view(-1).cuda()
    else:
        num_attack_classes = 9
        # target_labels:[9]
        target = input_data[3].view(-1).cuda()

    ori_mesh = Meshes(verts=vertice, faces=faces_idx).cuda()

    lower_bound = torch.ones(b) * 0
    scale_const = torch.ones(b) * cfg.initial_const
    upper_bound = torch.ones(b) * 1e10

    best_loss = [1e10] * b
    best_attack = [ori_mesh] * b
    best_score = [-1] * b
    best_attack_step = [-1] * b
    best_attack_BS_idx = [-1] * b

    for search_step in range(cfg.binary_max_steps):
        iter_best_loss = [1e10] * b
        iter_best_score = [-1] * b
        constrain_loss = torch.ones(b) * 1e10


        pc_ori = sample_points_from_meshes(ori_mesh, 1024).permute(0,2,1)
        if cfg.curv_loss_weight !=0:
            normal_ori = estimate_normal(pc_ori, k=8)
            theta_normal = compute_theta_normal(pc_ori, normal_ori, cfg.curv_loss_knn)
        else:
            theta_normal = None

        if cfg.is_partial_var:
            with torch.no_grad():
                e0, e1 = ori_mesh.edges_packed().unbind(1)  #e0 works as the index for the anchor points, and e1 works as the adj_point_idx to the anchor points (without repeat).

                idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
                idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
                idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

        # deform_verts = torch.zeros(src_mesh.verts_packed().shape).cuda()
        # nn.init.normal_(deform_verts, mean=0, std=1e-3)
        # deform_verts = deform_verts.requires_grad_()

        # if cfg.optim == 'adam':
        #     optimizer = torch.optim.Adam([deform_verts], lr=cfg.lr)
        # elif cfg.optim == 'sgd':
        #     optimizer = torch.optim.SGD([deform_verts], lr=cfg.lr, momentum=0.9)
        # else:
        #     assert False, 'Wrong optimizer!'
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)

        attack_success = torch.zeros(b).cuda()
        for step in range(cfg.iter_max_steps):

            if cfg.is_partial_var:
                if step%50 == 0:
                    with torch.no_grad():
                        #FIXME: how about using the critical points?
                        init_point = np.random.randint(ori_mesh.verts_packed().shape[0])
                        #FIXME: BFS or DFS?
                        involved_idx_list = []
                        visited_idx_list = []
                        involved_idx_list.append(init_point)
                        last_involved_idx_list = involved_idx_list
                        for _ in range(cfg.knn_range):
                            iter_involved_idx_list = []
                            for involved_idx in last_involved_idx_list:
                                if involved_idx not in visited_idx_list:
                                    visited_idx_list.append(involved_idx)
                                    iter_idx_list = (idx == involved_idx).nonzero()
                                    iter_idx_list = iter_idx_list[:int(iter_idx_list.size(0)/2),:]

                                    iter_eachpoint_involved_idx_list = idx[1,iter_idx_list[:, 1]].tolist()
                                    # yields the elements in `iter_eachpoint_involved_idx_list` that are NOT in `involved_idx_list`
                                    iter_involved_idx_list += np.setdiff1d(iter_eachpoint_involved_idx_list,involved_idx_list).tolist()

                            last_involved_idx_list = iter_involved_idx_list
                            involved_idx_list = involved_idx_list + iter_involved_idx_list

                    deform_verts = torch.zeros(involved_idx_list.__len__(), 3).cuda()
                    nn.init.normal_(deform_verts, mean=0, std=1e-3)
                    deform_verts = deform_verts.requires_grad_()
                    if cfg.optim == 'adam':
                        optimizer = torch.optim.Adam([deform_verts], lr=cfg.lr)
                    elif cfg.optim == 'sgd':
                        optimizer = torch.optim.SGD([deform_verts], lr=cfg.lr, momentum=0.9)
                    else:
                        assert False, 'Wrong optimizer!'
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)

                    try:
                        with torch.no_grad():
                            periodical_src_mesh = new_src_mesh.clone()
                    except:
                        periodical_src_mesh = ori_mesh.clone()

            else:
                if step == 0:
                    deform_verts = torch.zeros(ori_mesh.verts_packed().shape).cuda()
                    nn.init.normal_(deform_verts, mean=0, std=1e-3)
                    deform_verts = deform_verts.requires_grad_()
                    if cfg.optim == 'adam':
                        optimizer = torch.optim.Adam([deform_verts], lr=cfg.lr)
                    elif cfg.optim == 'sgd':
                        optimizer = torch.optim.SGD([deform_verts], lr=cfg.lr, momentum=0.9)
                    else:
                        assert False, 'Wrong optimizer!'
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9990, last_epoch=-1)

            if cfg.is_partial_var:
                full_deform_verts = pad_larger_tensor_with_index(deform_verts, involved_idx_list, ori_mesh.verts_packed().shape[0])
                new_src_mesh = periodical_src_mesh.offset_verts(full_deform_verts)
            else:
                new_src_mesh = ori_mesh.offset_verts(deform_verts)

            # new_src_mesh = src_mesh.offset_verts(deform_verts)
            input_curr_iter = sample_points_from_meshes(new_src_mesh, 1024).permute(0,2,1)
            if cfg.curv_loss_weight !=0:
                normal_curr_iter = estimate_normal(input_curr_iter, k=8)
            else:
                normal_curr_iter = 0

            # for saving
            if (step%50 == 0) and cfg.is_debug:
                if (step == 0):
                    file_name = os.path.join(saved_dir, 'Mesh', 'ori'+'.obj')
                    final_verts, final_faces = ori_mesh.get_mesh_verts_faces(0)
                    save_obj(file_name, final_verts, final_faces)
                else:
                    file_name = os.path.join(saved_dir, 'Mesh', str(step)+'.obj')
                    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
                    save_obj(file_name, final_verts, final_faces)

            with torch.no_grad():
                if cfg.is_partial_var:
                    eval_full_deform_verts = pad_larger_tensor_with_index(deform_verts, involved_idx_list, ori_mesh.verts_packed().shape[0])
                    eval_meshes = periodical_src_mesh.offset_verts(eval_full_deform_verts)
                else:
                    eval_meshes = ori_mesh.offset_verts(deform_verts)

                for k in range(b):
                    batch_k_meshes = join_meshes_as_batch([eval_meshes[k]]*eval_num)
                    batch_k_attack_pc = sample_points_from_meshes(batch_k_meshes, 1024).permute(0, 2, 1)
                    batch_k_adv_output = net(batch_k_attack_pc)

                    attack_success[k] = _compare(torch.max(batch_k_adv_output,1)[1].data, target[k], gt_target[k], targeted).sum() > 0.2 * eval_num

                    # shape metric
                    batch_k_src_meshes = join_meshes_as_batch([ori_mesh[k]]*eval_num)
                    batch_k_src_pc = sample_points_from_meshes(batch_k_src_meshes, 1024).permute(0, 2, 1)
                    metric = chamfer_loss(batch_k_attack_pc, batch_k_src_pc).mean()

                    # randomly pick a point cloud for visualization
                    if attack_success[k] and (metric <best_loss[k]):
                        best_loss[k] = metric
                        verts, faces = eval_meshes.get_mesh_verts_faces(k)
                        best_attack[k] = Meshes(verts=[verts.detach().clone()], faces=[faces.detach().clone()])
                        best_attack_BS_idx[k] = search_step
                        best_attack_step[k] = step
                        best_score[k] = torch.max(batch_k_adv_output,1)[1].mode().values.item()
                    if attack_success[k] and (metric <iter_best_loss[k]):
                        iter_best_loss[k] = metric
                        iter_best_score[k] = torch.max(batch_k_adv_output,1)[1].mode().values.item()

            _, loss, dis_loss, hd_loss, nor_loss, constrain_loss, info = _forward_step(net, pc_ori, input_curr_iter, normal_curr_iter, theta_normal, new_src_mesh, target, scale_const, cfg, targeted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cfg.is_use_lr_scheduler:
                lr_scheduler.step()

            info = '[{5}/{6}][{0}/{1}][{2}/{3}] \t loss: {4:6.4f}\t'.format(search_step+1, cfg.binary_max_steps, step+1, cfg.iter_max_steps, loss.item(), i, loader_len) + info

            if (step+1) % step_print_freq == 0 or step == cfg.iter_max_steps - 1:
                print(info)

        # adjust the scale constants
        for k in range(b):
            if  iter_best_score[k] != -1:   #success
                lower_bound[k] = max(lower_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                else:
                    scale_const[k] *= 2
            else:
                upper_bound[k] = min(upper_bound[k], scale_const[k])
                if upper_bound[k] < 1e9:
                    scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5

    return best_attack, target, (np.array(best_loss)<1e10), best_attack_step, best_score  #best_attack:[b, 3, n], target: [b], best_loss:[b], best_attack_step:[b], best_score:[b]

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
    trg_dir = os.path.join(saved_root, cfg.attack_label, saved_dir, 'Mesh')
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)

    for i, input_data in enumerate(test_loader):
        print('[{0}/{1}]:'.format(i, test_loader.__len__()))
        adv_pc, targeted_label, attack_success_indicator = attack(net, input_data, cfg, i, len(test_loader), saved_dir)
        print(adv_pc.shape)
        break
    print('\n Finish! \n')
