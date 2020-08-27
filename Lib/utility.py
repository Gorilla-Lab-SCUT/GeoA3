import copy
import math
import os
import shutil
import sys
import time

import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
seaborn.set(rc={'figure.figsize':(11.7000,8.27000)})
linewidth = 4.0
fontsize = 15.0 + 11.0
markersize = 10.0
fixpoint_markersize = 15.0
bins=20
color_list = ['r','b','g']

def _normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def compute_theta_normal(pc, normal, k):
    b,_,n=pc.size()
    inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    vectors = nn_pts - pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2) #b*n

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

def estimate_normal_via_ori_normal(pc_adv, pc_ori, normal_ori, k):
    # pc_adv, pc_ori, normal_ori : [b,3,n]
    b,_,n=pc_adv.size()
    inter_dis = ((pc_adv.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
    inter_value, inter_idx = torch.topk(inter_dis, k, dim=2, largest=False, sorted=True) #not the same variable, use k=k
    inter_value = inter_value[:, :, 0].contiguous()
    inter_idx = inter_idx.contiguous()
    normal_pts = torch.gather(normal_ori, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    normal_pts_avg = normal_pts.mean(dim=-1)
    normal_pts_avg = normal_pts_avg/(normal_pts_avg.norm(dim=1)+1e-12)

    # If the points are not modified (distance = 0), use the normal directly from the original
    # one. Otherwise, use the mean of the normals of the k-nearest points.
    normal_ori_select = torch.gather(normal_ori, 2, inter_idx[:,:,0].view(b,1,n).expand(b,3,n)).view(b,3,n)
    condition = (inter_value<1e-6).unsqueeze(1).expand_as(normal_ori_select)
    normals_estimated = torch.where(condition, normal_ori_select, normal_pts_avg)

    return normals_estimated

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

def _compare(output, target, gt, targeted):
    if targeted:
        return output == target
    else:
        return output != gt

def farthest_points_normalized_single_numpy(obj_points, num_points):
    first = np.random.randint(len(obj_points))
    selected = [first]
    dists = np.full(shape = len(obj_points), fill_value = np.inf)

    for _ in range(num_points - 1):
        dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
        selected.append(np.argmax(dists))
    res_points = np.array(obj_points[selected])

    # normalize the points
    avg = np.average(res_points, axis = 0)
    res_points = res_points - avg[np.newaxis, :]
    dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
    res_points = res_points / dists

    return res_points

def farthest_points_sample(obj_points, num_points):
    assert obj_points.size(1) == 3
    b,_,n = obj_points.size()

    selected = torch.randint(obj_points.size(2), [obj_points.size(0),1]).cuda()
    dists = torch.full([obj_points.size(0), obj_points.size(2)], fill_value = np.inf).cuda()

    for _ in range(num_points - 1):
        dists = torch.min(dists, torch.norm(obj_points - torch.gather(obj_points, 2, selected[:,-1].unsqueeze(1).unsqueeze(2).expand(b,3,1)), dim = 1))
        selected = torch.cat([selected, torch.argmax(dists, dim=1, keepdim=True)], dim = 1)
    res_points = torch.gather(obj_points, 2, selected.unsqueeze(1).expand(b, 3, num_points))

    return res_points

def pad_larger_tensor_with_index(small_verts, small_in_larger_idx_list, larger_tensor_shape):
    full_deform_verts = torch.zeros(larger_tensor_shape,3).cuda()
    full_deform_verts[small_in_larger_idx_list] = small_verts
    return full_deform_verts

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    step_time = Average_meter()
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: {0}'.format(format_time(step_time)))
    L.append(' | Tot: {0}'.format(format_time(tot_time)))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

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

class Training_aux(object):
    def __init__(self, fsave):
        self.fsave = fsave
        if not os.path.exists(self.fsave):
            os.makedirs(self.fsave)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        '''
        usage:
        Training_aux.save_checkpoint(
            state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
            }, is_best = is_best)
        '''
        filename = os.path.join(self.fsave, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '%s/'%(self.fsave) + 'modelBest.pth.tar')
        return

    def load_checkpoint(self, model, optimizer, is_best):
        """Loads checkpoint from disk"""
        '''
        usage:
        start_epoch, best_prec1 = Training_aux.load_checkpoint(model = model, is_best = is_best)
        '''
        if is_best:
            filename = os.path.join(self.fsave, 'modelBest.pth.tar')
            print("=> loading best model '{}'".format(filename))
        else:
            filename = os.path.join(self.fsave, 'checkpoint.pth.tar')
            print("=> loading checkpoint '{}'".format(filename))

        if os.path.isfile(filename):
            checkpoint = torch.load(filename)

            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model_dict = model.state_dict()
                pretrained_dict = checkpoint['state_dict']

                new_dict = {k: v for k, v in pretrained_dict.items()[:-2] if k in model_dict.keys()}
                model_dict.update(new_dict)
                print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
                model.load_state_dict(model_dict)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("=> No optimizer loaded in '{}'".format(filename))
            print("==> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return start_epoch, best_prec1

    def write_err_to_file(self, info):
        """write error to txt"""
        fpath = os.path.join(self.fsave, 'state.txt')
        if os.path.isfile(fpath):
            file = open(fpath, 'a')
        else:
            file = open(fpath, "w")

        file.write(info)

        file.close()
        return

class Count_converge_iter(object):
    def __init__(self, fsave):
        self.fsave = fsave
        if not os.path.exists(self.fsave):
            os.makedirs(self.fsave)
        self.attack_step_list = []

    def record_converge_iter(self, attack_step_list):
        if -1 in attack_step_list:
            attack_step_list.remove(-1)
        self.attack_step_list += attack_step_list

    def save_converge_iter(self):
        fpath = os.path.join(self.fsave, 'converge_iter.mat')
        sio.savemat(fpath, {"attack_step_list": self.attack_step_list})

    def plot_converge_iter_hist(self):
        fpath = os.path.join(self.fsave, 'converge_iter.png')
        used_bins=np.histogram(np.hstack((self.attack_step_list)), bins=bins)[1]
        fig = plt.figure()
        ax = seaborn.distplot(self.attack_step_list, bins=used_bins)
        ax.set_xlabel('Converged iteration', fontsize=fontsize)
        ax.set_ylabel('Number of Samples', fontsize=fontsize)
        plt.savefig(fpath)


class Count_loss_iter(object):
    def __init__(self, fsave):
        self.fsave = fsave
        if not os.path.exists(self.fsave):
            os.makedirs(self.fsave)

    def record_loss_iter(self, loss_list):
        # loss_list:[steps, b]
        try:
            self.loss_numpy = np.concatenate((self.loss_numpy, np.array(loss_list)), axis=1)
        except:
            self.loss_numpy = np.array(loss_list)

    def save_loss_iter(self):
        fpath = os.path.join(self.fsave, 'loss_iter.mat')
        sio.savemat(fpath, {"loss": self.loss_numpy})

    def plot_loss_iter_hist(self):
        fpath = os.path.join(self.fsave, 'loss_iter.png')
        num_iter, num_sample = self.loss_numpy.shape

        start_iter = 1
        x = np.arange(start_iter, num_iter+start_iter)
        x.astype(int)
        loss_mean = self.loss_numpy.mean(1)
        loss_std = self.loss_numpy.std(1)

        f, ax = plt.subplots(1,1)
        ax.plot(x, loss_mean, color=color_list[0])
        r1 = list(map(lambda x: x[0]-x[1], zip(loss_mean, loss_std)))
        r2 = list(map(lambda x: x[0]+x[1], zip(loss_mean, loss_std)))
        ax.fill_between(x, r1, r2, color=color_list[0], alpha=0.2)
        ax.set_xlabel('Number of iteration', fontsize=fontsize)
        ax.set_ylabel('Magnitude of loss', fontsize=fontsize)
        plt.savefig(fpath)

