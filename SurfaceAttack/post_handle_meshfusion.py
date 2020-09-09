import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/../'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Lib'))

from utility import read_off, write_obj, pc_normalize_torch

filepath = "Ori_10000_PointNet_BPA_WT_MeshFusion"
output_path = "Ori_10000_PointNet_BPA_WT_MeshFusion_post"

if not os.path.exists(output_path):
    os.makedirs(output_path)

file_name_list = os.listdir(filepath)
file_name_list.sort()
for input_file in file_name_list:
    if os.path.splitext(input_file)[1] == '.off':
        vertices, faces = read_off(os.path.join(filepath, input_file))
        vertices = pc_normalize_torch(torch.Tensor(vertices))
        write_obj(os.path.join(output_path, input_file.split(".")[0]+".obj"), vertices[:,[2,1,0]].tolist(), torch.LongTensor(faces)[:,1:4].tolist())
    else:
        pass

