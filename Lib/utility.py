import os
import sys
import time
import math
import shutil
import copy
import numpy as np

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

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

def normalize_inverse(input_imgs, dataset_type = 'CIFAR10'):
    if isinstance(input_imgs, Variable):
        imgs = copy.deepcopy(input_imgs.data)
    else:
        imgs = copy.deepcopy(input_imgs)
    if dataset_type == 'CIFAR10' or dataset_type == 'Sep_CIFAR10' or dataset_type == 'Bi_CIFAR10':
        imgs[:,0,:,:] = imgs[:,0,:,:] * (63.0/255.0) + (125.3/255.0)
        imgs[:,1,:,:] = imgs[:,1,:,:] * (62.1/255.0) + (123.0/255.0)
        imgs[:,2,:,:] = imgs[:,2,:,:] * (66.7/255.0) + (113.9/255.0)
    elif dataset_type == 'ImageNet' or dataset_type == 'tiny_ImageNet' or dataset_type == 'SBD':
        imgs[:,0,:,:] = imgs[:,0,:,:] * 0.229 + 0.485
        imgs[:,1,:,:] = imgs[:,1,:,:] * 0.224 + 0.456
        imgs[:,2,:,:] = imgs[:,2,:,:] * 0.225 + 0.406
    else:
        raise Exception('DO NOT support inverse-normalizing such dataset yet!')
    return imgs

def normalize(input_imgs, dataset_type = 'CIFAR10'):
    if isinstance(input_imgs, Variable):
        imgs = copy.deepcopy(input_imgs.data)
    else:
        imgs = copy.deepcopy(input_imgs)
    if dataset_type == 'CIFAR10' or dataset_type == 'Sep_CIFAR10' or dataset_type == 'Bi_CIFAR10':
        imgs[:,0,:,:] = (imgs[:,0,:,:] - (125.3/255.0)) / (63.0/255.0)
        imgs[:,1,:,:] = (imgs[:,1,:,:] - (123.0/255.0)) / (62.1/255.0)
        imgs[:,2,:,:] = (imgs[:,2,:,:] - (113.9/255.0)) / (66.7/255.0)
    elif dataset_type == 'ImageNet' or dataset_type == 'tiny_ImageNet' or dataset_type == 'SBD':
        imgs[:,0,:,:] = (imgs[:,0,:,:] - 0.485) / 0.229
        imgs[:,1,:,:] = (imgs[:,1,:,:] - 0.456) / 0.224
        imgs[:,2,:,:] = (imgs[:,2,:,:] - 0.406) / 0.225
    else:
        raise Exception('DO NOT support normalizing such dataset yet!')
    return imgs

def normalize_epsilon(epsilon_ori, dataset_type = 'CIFAR10', norm_type = 'interval'):
    assert epsilon_ori.size(0) == 3
    epsilon = copy.deepcopy(epsilon_ori)
    if norm_type == 'interval':
        if dataset_type == 'CIFAR10' or dataset_type == 'Sep_CIFAR10' or dataset_type == 'Bi_CIFAR10':
            epsilon[0] = (epsilon[0] / (63.0/255.0))
            epsilon[1] = (epsilon[1] / (62.1/255.0))
            epsilon[2] = (epsilon[2] / (66.7/255.0))
        elif dataset_type == 'ImageNet' or dataset_type == 'tiny_ImageNet' or dataset_type == 'SBD':
            epsilon[0] = (epsilon[0] / 0.229)
            epsilon[1] = (epsilon[1] / 0.224)
            epsilon[2] = (epsilon[2] / 0.225)
        else:
            raise Exception('DO NOT support normalizing such epsilon yet!')
    elif norm_type == 'value':
        if dataset_type == 'CIFAR10' or dataset_type == 'Sep_CIFAR10' or dataset_type == 'Bi_CIFAR10':
            epsilon[0] = (epsilon[0] - (125.3/255.0)) / (63.0/255.0)
            epsilon[1] = (epsilon[1] - (123.0/255.0)) / (62.1/255.0)
            epsilon[2] = (epsilon[2] - (113.9/255.0)) / (66.7/255.0)
        elif dataset_type == 'ImageNet' or dataset_type == 'tiny_ImageNet' or dataset_type == 'SBD':
            epsilon[0] = (epsilon[0] - 0.485) / 0.229
            epsilon[1] = (epsilon[1] - 0.456) / 0.224
            epsilon[2] = (epsilon[2] - 0.406) / 0.225        
        else:
            raise Exception('DO NOT support normalizing such epsilon yet!')
    return epsilon

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

class Img_show(object):
    """show image"""  
    def __init__(self, fsave, show_time = 1):
        self.time = show_time
        self.fsave = fsave
        if not os.path.exists(self.fsave):
            os.makedirs(self.fsave)

    def close(self):
        time.sleep(self.time)
        plt.close()

    def imshow_ImageNet(self, imgs, fname = 'showed_img'):
        assert imgs.size().__len__() == 4
        if isinstance(imgs, Variable):
            imgs = imgs.data
        imgs_show = []
        for i in range(0, imgs.size(0)):
            img_show = imgs.cpu()[i].clone()
            img_show[0] = img_show[0] * 0.229 + 0.485    # unnormalize
            img_show[1] = img_show[1] * 0.224 + 0.456     # unnormalize
            img_show[2] = img_show[2] * 0.225 + 0.406     # unnormalize
            imgs_show.append(img_show)
        imgs_show = torchvision.utils.make_grid(imgs_show)
        fpath = os.path.join(self.fsave, fname) + '.png'
        torchvision.utils.save_image(imgs_show, fpath)

    def imshow_CIFAR10(self, imgs, fname = 'showed_img'):
        assert imgs.size().__len__() == 4
        if isinstance(imgs, Variable):
            imgs = imgs.data
        imgs_show = []
        for i in range(0, imgs.size(0)):
            img_show = imgs[i].cpu().clone()
            img_show[0] = img_show[0] * (63.0/255.0) + (125.3/255.0)     # unnormalize
            img_show[1] = img_show[1] * (62.1/255.0) + (123.0/255.0)     # unnormalize
            img_show[2] = img_show[2] * (66.7/255.0) + (113.9/255.0)     # unnormalize
            imgs_show.append(img_show)
        imgs_show = torchvision.utils.make_grid(imgs_show)
        fpath = os.path.join(self.fsave, fname) + '.png'
        torchvision.utils.save_image(imgs_show, fpath)

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

    def plt_curve(self, x, y, label, x_axis, y_axis, tile, save_name):
        fpath = os.path.join(self.fsave, save_name+'.png')

        seaborn.set()
        seaborn.set(rc={'figure.figsize':(11.7000,8.27000)})
        linewidth = 4.0
        fontsize = 15.0
        markersize = 10.0
        color_list = ['r','b','g']

        f, ax = plt.subplots()
        ax.plot(x,y,color=color_list[0], linewidth=linewidth, markersize=markersize, label=label)

        ax.set_xlabel(x_axis, fontsize=fontsize)
        ax.set_ylabel(y_axis, fontsize=fontsize)
        ax.set_title(tile, fontsize=fontsize+5)

        ax.tick_params(labelsize=fontsize)
        ax.legend(fontsize=fontsize)
        plt.savefig(fpath)
        plt.close()

    def plt_hist(self, y, label, x_axis, y_axis='Number of Values', tile='Histogram', save_name='Hist', kde=True, bins=50):
        fpath = sos.path.join(self.fsave, save_name+'.png')

        seaborn.set()
        seaborn.set(rc={'figure.figsize':(11.7000,8.27000)})
        linewidth = 4.0
        fontsize = 15.0
        markersize = 10.0
        color_list = ['r','b','g']

        f, ax = plt.subplots()
        seaborn.distplot(y, kde=kde, bins=bins, color=color_list[0], label=label, ax=ax)

        ax.set_xlabel(x_axis, fontsize=fontsize)
        ax.set_ylabel(y_axis, fontsize=fontsize)
        ax.set_title(tile, fontsize=fontsize+5)

        ax.tick_params(labelsize=fontsize)
        ax.legend(fontsize=fontsize)
        plt.savefig(fpath)
        plt.close()


    def plt_weight(self, weight, padding=1, save_name='Vis_heatmap'):
        assert weight.size().__len__() == 4
        if isinstance(weight, Variable):
            weight = weight.data
        if weight.size().__len__() == 4:
            c_out = weight.size(0)
            c_in = weight.size(1)
            kw = weight.size(2)
            kh = weight.size(3)

            to_plot_weight = c_in * (kw+padding) - padding
            to_plot_height = c_out * (kh+padding) - padding

            out_img = np.ones((to_plot_height, to_plot_weight)) * 0
            h_offset = 0
            for out_idx in range(0, c_out):
                w_offset = 0
                for in_idx in range(0, c_in):
                    out_img[out_idx*kh + h_offset:(out_idx+1)*kh + h_offset, in_idx*kw + w_offset: (in_idx+1)*kw + w_offset] = weight[out_idx, in_idx, :, :].cpu().numpy()
                    w_offset += 1
                h_offset += 1

        fpath = os.path.join(self.fsave, save_name+'.png')
        seaborn.set()
        seaborn.set(rc={'figure.figsize':(11.7000,8.27000)})

        f, ax = plt.subplots()
        seaborn.heatmap(data=out_img, center=0, xticklabels=False, yticklabels=False, square=True, robust=True,)

        plt.savefig(fpath)
        plt.close()

    def get_heatmap_figure_multiC(self, featuremap, padding=1, aggregate=False, vmax=1):
        assert featuremap.size().__len__() == 3
        if isinstance(featuremap, Variable):
            featuremap = featuremap.data
        seaborn.set('poster')
        if aggregate == True:
            channel = featuremap.size(0)
            fw = featuremap.size(1)
            fh = featuremap.size(2)

            to_plot_width = channel * (fw+padding) - padding
            to_plot_height = fh
            out_img = np.ones((to_plot_height, to_plot_width)) * 0.5

            w_offset = 0
            for in_idx in range(0, channel):
                out_img[0:fh, in_idx*fw + w_offset: (in_idx+1)*fw + w_offset] = featuremap[in_idx, :, :].cpu().numpy()
                w_offset += 1

            f, ax = plt.subplots()
            seaborn.heatmap(data=out_img, vmin=0, vmax=vmax, xticklabels=False, yticklabels=False, square=True, robust=True,)
            return f
        else:
            f_list = []
            channel = featuremap.size(0)
            for in_idx in range(0, channel):
                f, ax = plt.subplots()
                seaborn.heatmap(data=featuremap[in_idx, :, :].cpu().numpy(), vmin=0, vmax=vmax, xticklabels=False, yticklabels=False, square=True, robust=True,)
                f_list.append(f)
                plt.close()
            return f_list