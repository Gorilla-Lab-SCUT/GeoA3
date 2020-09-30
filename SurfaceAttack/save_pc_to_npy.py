from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Lib'))
from utility import natural_sort, read_obj

def save_pc_to_npy(cfg):
    save_dir = cfg.input_pc_dir+'/../04_pts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name_list = os.listdir(cfg.input_pc_dir)
    file_name_list = natural_sort(file_name_list)
    file_test_set = os.path.join(save_dir, 'testset.txt')
    cnt = 0
    for input_file in file_name_list:
        cnt+=1
        vertices, _ = read_obj(os.path.join(cfg.input_pc_dir, input_file))
        np.save(os.path.join(save_dir, input_file.split(".")[0]+'.npy'), np.array(vertices).astype(np.float32))
        print('[{0}/{1}] Finished.'.format(cnt, file_name_list.__len__()))

        with open(file_test_set, "a") as text_file:
            text_file.write(input_file.split(".")[0]+"\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save .npy file for Points2Sur method')
    parser.add_argument('--input_pc_dir', default='../Exps/PointNet_npoint1024/All/GeoA3_840_BiStep5_IterStep500_Optadam_Lr0.01_Initcons500.0_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16_UniLoss1.0_LRExp_ProGrad_cclinf0.1/Obj', type=str, help='')
    cfg  = parser.parse_args()
    save_pc_to_npy(cfg)
