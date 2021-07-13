from __future__ import absolute_import, division, print_function

import argparse
import gc
import os
import pdb
import pprint
import shutil
import sys
import time

import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import Provider.provider
from Lib.utility import Average_meter, progress_bar
from Provider.modelnet_trn_test import ModelNetDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Model'))
sys.path.append(os.path.join(ROOT_DIR, 'Lib'))
sys.path.append(os.path.join(ROOT_DIR, 'Provider'))


parser = argparse.ArgumentParser(description='Point Cloud Training')
# ========================= Random seed Configs ==========================
parser.add_argument('--id', default=0, type=int, help='')
parser.add_argument('--random_seed', default=0, type=int, help='')
# ========================= Data loader Configs ==========================
parser.add_argument('--datadir', default='/data/modelnet40_normal_resampled/', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
parser.add_argument('--npoint', default=1024, type=int, help='')
parser.add_argument('--is_aug_data', dest='is_aug_data', action='store_true', default=False, help=' ')
# ========================= Model Configs ==========================
parser.add_argument('--arch', default='PointNet', type=str, metavar='ARCH', help='')
# ========================= Training Configs ==========================
parser.add_argument('-g', '--mGPU', default=1, type=int, metavar='N', help='num of GPUs (default: 1)')
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',  help='number of total epochs to run')
parser.add_argument('--lr',  default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--decay-epochs', default=20, type=int, metavar='N',  help='number of epochs to decay')
parser.add_argument('--bn_momentum',  default=0.5, type=float, metavar='BN', help='initial bn momentum')
parser.add_argument('--wd', default=0.0001, type=float, metavar='W', help='weight decay (default: 1e-4)')
# ========================= Runtime Configs ==========================
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# ========================= Monitor Configs ==========================
parser.add_argument('--is_use_tb', dest='is_use_tb', action='store_true', default=False, help='Using Tensorboard')

cfg  = parser.parse_args()
print(cfg)

# =========================
#          save_dir
# =========================
#tmp = 'Id_' + str(cfg.id) + '_batchsize_' + str(cfg.batch_size) + '_wd' + str(cfg.wd)

DATA_PATH = cfg.datadir
modeldir = os.path.join('Pretrained', cfg.arch, str(cfg.npoint))

if not os.path.exists(modeldir):
    os.makedirs(modeldir)

# =========================
#        set tb_board
# =========================
tb_writer = 0
if cfg.is_use_tb:
    assert (int((torch.__version__).split('.')[0]) >= 1 and int((torch.__version__).split('.')[1]) >= 1)
    from torch.utils.tensorboard import SummaryWriter
    if not os.path.exists(os.path.join(modeldir, 'TB_event')):
        os.makedirs(os.path.join(modeldir, 'TB_event'))
    tb_writer = SummaryWriter(log_dir=os.path.join(modeldir, 'TB_event'))

# =========================
#    addi function & class
# =========================
class softmax_with_smoothing_label_loss(nn.Module):
    def __init__(self, num_classes=40, label_smoothing=0.2):
        super(softmax_with_smoothing_label_loss, self).__init__()
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.ones = None

    def forward(self, output, target):
        if self.ones is None:
            self.ones = Variable(torch.eye(self.num_classes).to(output).cuda())
        output = -1*self.log_softmax(output)
        one_hot = self.ones.index_select(0,target)
        one_hot = one_hot*(1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        loss = one_hot * output
        loss = loss.sum(dim=1)
        loss = loss.mean()

        return loss

def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir, filename), os.path.join(dir, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.7 ** (epoch // 20))
    lr = max(0.00001, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

# =========================
#           main
# =========================
def main():
    if cfg.id == 0:
        seed = cfg.random_seed
    else:
        seed = int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # dataset
    TRAIN_DATASET = ModelNetDataset(root=DATA_PATH, batch_size=cfg.batch_size, npoints=cfg.npoint, split='train', normal_channel=False)
    TEST_DATASET = ModelNetDataset(root=DATA_PATH, batch_size=cfg.batch_size, npoints=cfg.npoint, split='test', normal_channel=False)

    # model
    if cfg.arch == 'PointNet':
        from Model.PointNet import PointNet
        net = PointNet(cfg.classes).cuda()
    elif cfg.arch == 'PointNetPP':
        from Model.PointNetPP_ssg import PointNet2ClassificationSSG
        net = PointNet2ClassificationSSG(use_xyz=True, use_normal=False).cuda()
    else:
        assert False
    criterion = softmax_with_smoothing_label_loss().cuda()

    params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value], 'lr': cfg.lr, 'weight_decay': cfg.wd}]

    optimizer = torch.optim.Adam(params)

    # resume
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            start_epoch = checkpoint['epoch']+1
            best_prec = checkpoint['best_prec']
            class_prec = checkpoint['class_prec']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("\n=> loaded checkpoint '{}' (epoch {})" .format(cfg.resume, checkpoint['epoch']))
        else:
            assert False, 'WRONG RESUME PATH!'
    else:
        start_epoch = 1
        best_prec = 0
        class_prec = 0

    if cfg.mGPU>1:
        net = torch.nn.DataParallel(net,device_ids=range(0, cfg.mGPU)).cuda()

    # train & test
    for epoch in range(start_epoch, cfg.epochs+1):
        # train
        trn_batch_time = Average_meter()
        trn_data_time = Average_meter()
        trn_losses = Average_meter()
        trn_acc = Average_meter()

        net.train()
        end = time.time()
        i = 0
        while TRAIN_DATASET.has_next_batch():
            i+=1
            trn_data_time.update(time.time() - end)
            net.zero_grad()

            points, target = TRAIN_DATASET.next_batch(cfg.is_aug_data)
            points = torch.Tensor(points).contiguous()
            target = torch.Tensor(target).long()

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            points = points[:,[0,2,1],:]
            pc_var = Variable(points).cuda()
            label_var = Variable(target.long()).cuda()

            trn_output, transform = net(pc_var)

            trn_loss = criterion(trn_output, label_var)

            K = transform.size(1)
            mat_diff = torch.bmm(transform, transform.permute(0, 2, 1))
            mat_diff -= Variable(torch.eye(K).float().cuda().unsqueeze(0))
            mat_diff_loss = torch.sum(mat_diff**2)/2
            trn_loss = trn_loss + mat_diff_loss * 0.001

            acc = accuracy(trn_output.data, label_var.data, topk=(1, ))
            trn_losses.update(trn_loss.item(), trn_output.size(0))
            trn_acc.update(acc[0][0], trn_output.size(0))

            optimizer.zero_grad()
            trn_loss.backward()
            optimizer.step()

            trn_batch_time.update(time.time() - end)
            end = time.time()

            process_length = len(TRAIN_DATASET)/float(TRAIN_DATASET.batch_size)
            progress_bar(i, process_length, 'Loss: {loss.avg:.4f} | Prec@1 {top1.avg:.3f} '.format(loss=trn_losses, top1=trn_acc))
            if cfg.is_use_tb:
                tb_writer.add_scalar('Train Loss', trn_losses.avg, epoch * process_length + i)
                tb_writer.add_scalar('Train Top1', trn_acc.avg, epoch * process_length + i)

            gc.collect()


        adjust_learning_rate(optimizer, epoch, cfg.lr)
        if cfg.arch == 'PointNet' or cfg.arch == 'PointNetPP':
            if cfg.mGPU>1:
                net.module.adjust_bn_momentum(epoch, cfg.bn_momentum)
            else:
                net.adjust_bn_momentum(epoch, cfg.bn_momentum)


        TRAIN_DATASET.reset()
        with open(os.path.join(modeldir, 'result.txt'), 'at') as f:
            f.write('epoch[{:3d}] train-acc: {acc.avg:.3f}'.format(epoch, acc=trn_acc))

        # val
        test_batch_time = Average_meter()
        test_losses = Average_meter()
        test_acc = Average_meter()

        net.eval()
        end = time.time()
        total_seen_class = [0 for _ in range(cfg.classes)]
        total_correct_class = [0 for _ in range(cfg.classes)]

        i = 0
        with torch.no_grad():
            while TEST_DATASET.has_next_batch():
                i += 1
                points, target = TEST_DATASET.next_batch(False)
                points = torch.Tensor(points).contiguous()
                target = torch.Tensor(target).long()

                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                optimizer.zero_grad()

                points = points[:,[0,2,1],:]

                pc_var = Variable(points).cuda()
                label_var = Variable(target.long()).cuda()

                test_output = net(pc_var)
                test_loss = criterion(test_output, label_var)

                acc = accuracy(test_output.data, label_var.data, topk=(1, ))
                test_losses.update(test_loss.item(), test_output.size(0))
                test_acc.update(acc[0][0], test_output.size(0))

                _, predicted_idx = torch.max(test_output.data, dim=1, keepdim=False)

                for instance in range(test_output.size(0)):
                    L = int(label_var.data[instance])
                    total_seen_class[L] += 1
                    total_correct_class[L] += (int(predicted_idx[instance]) == L)

                test_batch_time.update(time.time() - end)
                end = time.time()

                process_length = len(TEST_DATASET)/float(TEST_DATASET.batch_size)
                progress_bar(i, process_length, 'Loss: {loss.avg:.4f} | Prec@1 {top1.avg:.3f} '.format(loss=test_losses, top1=test_acc))
                if cfg.is_use_tb:
                    tb_writer.add_scalar('Test Loss', test_losses.avg, epoch * process_length + i)
                    tb_writer.add_scalar('Test Top1', test_acc.avg, epoch * process_length + i)

            avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))*100
            with open(os.path.join(modeldir, 'result.txt'), 'at') as f:
                 f.write('\t\ttest: C-acc {:.3f}  I-acc {:.3f}'.format(avg_class_acc, test_acc.avg))
            TEST_DATASET.reset()
        # store checkpoint
        prec = test_acc.avg
        if prec > best_prec:
            is_best = True
        elif (prec == best_prec ) and (class_prec<avg_class_acc):
            is_best = True
        else:
            is_best = False

        if is_best:
            best_prec = prec
            class_prec = avg_class_acc

        if cfg.mGPU>1:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.module.state_dict(),
                'best_prec': best_prec,
                'class_prec': class_prec,
                'optimizer' : optimizer.state_dict(),
                }, is_best, modeldir)
        else:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_prec': best_prec,
                'class_prec': class_prec,
                'optimizer' : optimizer.state_dict(),
                }, is_best, modeldir)
        with open(os.path.join(modeldir, 'result.txt'), 'at') as f:
            if is_best:
                f.write('\t\tbest: C-acc {:.3f}  I-acc {:.3f}\n'.format(class_prec, best_prec))
            else:
                f.write('\n')

        print('===> epoch [{:3d}]:  avg_class_acc  {:.4f}    avg_instance_acc {:.4f}    |    best: avg_class_acc  {:.4f}'
            '    avg_instance_acc {:.4f}\n'.format(epoch, avg_class_acc, test_acc.avg, class_prec, best_prec))

if __name__ == '__main__':
    main()
