from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import shutil
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'SurfaceAttack'))
sys.path.append(os.path.join(ROOT_DIR, '3rdParty', 'points2surf'))


from eval_from_pc_file import eval_from_pc_file
from full_eval import full_eval
from main_attack import main
from save_pc_to_npy import save_pc_to_npy

'''
ten_label_indexes = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']
'''
ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']

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
parser.add_argument('--eval_num', type=int, default=1, help='')
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
## uniform loss
parser.add_argument('--uniform_loss_weight', type=float, default=0.0, help='')
## KNN smoothing loss
parser.add_argument('--knn_smoothing_loss_weight', type=float, default=5.0, help='')
parser.add_argument('--knn_smoothing_k', type=int, default=5, help='')
parser.add_argument('--knn_threshold_coef', type=float, default=1.10, help='')
## Laplacian loss for mesh
parser.add_argument('--laplacian_loss_weight', type=float, default=0, help='')
parser.add_argument('--edge_loss_weight', type=float, default=0, help='')
## Mesh opt
parser.add_argument('--is_partial_var', dest='is_partial_var', action='store_true', default=False, help='')
parser.add_argument('--knn_range', type=int, default=3, help='')
parser.add_argument('--is_subsample_opt', dest='is_subsample_opt', action='store_true', default=False, help='')
parser.add_argument('--is_use_lr_scheduler', dest='is_use_lr_scheduler', action='store_true', default=False, help='')
## perturbation clip setting
parser.add_argument('--cc_linf', type=float, default=0.0, help='Coefficient for infinity norm')
## Proj offset
parser.add_argument('--is_real_offset', action='store_true', default=False, help='')
parser.add_argument('--is_pro_grad', action='store_true', default=False, help='')
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
parser.add_argument('--is_record_loss', action='store_true', default=False, help='')
#------------OS-----------------------
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--is_save_normal', action='store_true', default=False, help='')
parser.add_argument('--is_debug', action='store_true', default=False, help='')
parser.add_argument('--is_low_memory', action='store_true', default=False, help='')
#===========================================
parser.add_argument('--input_pc_dir', default=None, type=str, help='')
#===========================================
parser.add_argument('--datadir', default='./Vis/Post_Mesh/', type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--trg_file_path', default=None, type=str, metavar='DIR', help='path to adv dataset')
parser.add_argument('--save_dir', default=None, type=str, help='')
parser.add_argument('--is_mesh', action='store_true', default=False, help='')
#===========================================
parser.add_argument('--indir', type=str, default='3rdParty/points2surf/datasets', help='input folder (meshes)')
parser.add_argument('--outdir', type=str, default='3rdParty/points2surf/results',
                    help='output folder (estimated point cloud properties)')
parser.add_argument('--dataset', nargs='+', type=str, default=['testset.txt'], help='shape set file name')
parser.add_argument('--reconstruction', type=bool, default=False, help='do reconstruction instead of evaluation')
parser.add_argument('--query_grid_resolution', type=int, default=256,
                    help='resolution of sampled volume used for reconstruction')
parser.add_argument('--epsilon', type=int, default=3,
                    help='neighborhood size for reconstruction')
parser.add_argument('--certainty_threshold', type=float, default=13, help='')
parser.add_argument('--sigma', type=int, default=5, help='')
parser.add_argument('--up_sampling_factor', type=int, default=10,
                    help='Neighborhood of points that is queried with the network. '
                            'This enables you to set the trade-off between computation time and tolerance for '
                            'sparsely sampled surfaces.')
parser.add_argument('--modeldir', type=str, default='3rdParty/points2surf/models', help='model folder')
parser.add_argument('--models', type=str, default='p2s_max',
                    help='names of trained models, can evaluate multiple models')
parser.add_argument('--modelpostfix', type=str, default='_model_249.pth', help='model file postfix')
parser.add_argument('--parampostfix', type=str, default='_params.pth', help='parameter file postfix')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
parser.add_argument('--sparse_patches', type=int, default=False,
                    help='evaluate on a sparse set of patches, given by a '
                            '.pidx file containing the patch center point indices.')
parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                    'full: evaluate all points in the dataset\n'
                    'sequential_shapes_random_patches: pick n random '
                    'points from each shape as patch centers, shape order is not randomized')
parser.add_argument('--patches_per_shape', type=int, default=1000,
                    help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
parser.add_argument('--query_points_per_patch', type=int, default=1,
                    help='number of query points per patch')
parser.add_argument('--sub_sample_size', type=int, default=500,
                    help='number of points of the point cloud that are trained with each patch')
parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
parser.add_argument('--batchSize', type=int, default=501, help='batch size, if 0 the training batch size is used')
parser.add_argument('--workers', type=int, default=7,
                    help='number of data loading workers - 0 means same thread as main execution')
parser.add_argument('--cache_capacity', type=int, default=5,
                    help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
cfg  = parser.parse_args()

saved_dir = main(cfg)
#saved_dir = 'Exps/PointNet_npoint1024/Test/GeoA3_999_BiStep5_IterStep500_Optadam_Lr0.05_Initcons500.0_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16_UniLoss1.0_LRExp_ProGradRO_cclinf0.1'
saved_dir_pc= os.path.join(saved_dir, 'PC')

cfg.input_pc_dir = saved_dir_pc
save_pc_to_npy(cfg)

saved_dir_pts = os.path.join(saved_dir, '04_pts')
data_set_substr = saved_dir.split('/')[-1].replace('.', '')
dataset_dir = os.path.join('3rdParty', 'points2surf', 'datasets', data_set_substr)

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

saved_dir_pts_txt = os.path.join(dataset_dir, '04_pts', 'testset.txt')

shutil.move(saved_dir_pts, dataset_dir)
shutil.move(saved_dir_pts_txt, dataset_dir)

cfg.dataset = os.path.join(data_set_substr, 'testset.txt')
full_eval(cfg)


output_mesh_dir = os.path.join('3rdParty/points2surf/results/p2s_max_model_249', data_set_substr, 'rec/mesh')
save_mesh_dir = os.path.join(saved_dir, 'Reconstruct_from_p2s')
shutil.move(output_mesh_dir, save_mesh_dir)

cfg.save_mesh_dir = save_mesh_dir
cfg.is_mesh = True
eval_from_pc_file(cfg)

