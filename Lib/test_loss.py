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

from loss_utils import uniform_loss

def test_uniform_loss(rand_input):
    loss = uniform_loss(rand_input)
    return loss


if __name__ == '__main__':
    #rand_input = torch.randn([1,3,10000]).cuda()
    input_file = "test.mat"
    dataset = sio.loadmat(input_file)
    rand_input = torch.FloatTensor(dataset['adversary_point_clouds']).unsqueeze(0).cuda()
    rand_input.requires_grad_()

    optimizer = optim.SGD([rand_input], lr=0.001)
    total_step = 1000
    for i in range(0, total_step):
        loss = test_uniform_loss(rand_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print('[{0}/{1}]'.format(i,total_step))
            fout = open(os.path.join('Test', str(i)+'.xyz'), 'w')
            for m in range(rand_input.shape[2]):
                fout.write('%f %f %f 0 0 0\n' % (rand_input[0, 0, m], rand_input[0, 1, m], rand_input[0, 2, m]))
            fout.close()
