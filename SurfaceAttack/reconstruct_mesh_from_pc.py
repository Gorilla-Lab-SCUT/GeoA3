from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np
import open3d as o3d
from pytorch3d.ops import estimate_pointcloud_normals
import scipy.io as sio
import torch


def reconstruction_from_mat(input_file, reconstruct_type, save_dir, curr_idx, length):
    dataset = sio.loadmat(input_file)
    try:
        pc = torch.FloatTensor(dataset['adversary_point_clouds'])
    except:
        pc = torch.FloatTensor(dataset['data'])

    try:
        normal = torch.FloatTensor(dataset['est_normal'])
    except:
        normal = torch.FloatTensor(dataset['normal'])
    else:
        normal = None


    # pc and normal should be [(b), n, 3]
    if pc.size().__len__() == 2:
        pc = pc.unsqueeze(0)
    if pc.size(1) == 3:
        pc = pc.permute(0,2,1)

    if (normal is not None) and (not cfg.is_use_est_normal):
        if normal.size().__len__() == 2:
            normal = normal.unsqueeze(0)
        if normal.size(1) == 3:
            normal = normal.permute(0,2,1)
    else:
        normal = estimate_pointcloud_normals(pc.permute(0,2,1), neighborhood_size=8, disambiguate_directions=True).permute(0,2,1)

    for i in range(pc.size(0)):
        if (os.path.join(save_dir, input_file.split("/")[-1].split(".")[0]+".obj") in os.listdir(os.path.join(save_dir))) or (os.path.join(save_dir, input_file.split("/")[-1].split(".")[0]+"_"+str(i)+".obj") in os.listdir(os.path.join(save_dir))):
            print('[{0}/{1}][{2}/{3}] Already exits.'.format(curr_idx, length, i+1, pc.size(0)))
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[i])
        pcd.normals = o3d.utility.Vector3dVector(normal[i])

        if reconstruct_type == 'BPA':
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist

            bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

            output_mesh = bpa_mesh.simplify_quadric_decimation(100000)
            output_mesh.remove_degenerate_triangles()
            output_mesh.remove_duplicated_triangles()
            output_mesh.remove_duplicated_vertices()
            output_mesh.remove_non_manifold_edges()
        elif reconstruct_type == 'PSR':
            poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
            bbox = pcd.get_axis_aligned_bounding_box()
            output_mesh = poisson_mesh.crop(bbox)

        if pc.size(0) == 1:
            o3d.io.write_triangle_mesh(os.path.join(save_dir, input_file.split("/")[-1].split(".")[0]+".obj"), output_mesh)
        else:
            o3d.io.write_triangle_mesh(os.path.join(save_dir, input_file.split("/")[-1].split(".")[0]+"_"+str(i)+".obj"), output_mesh)

        print('[{0}/{1}][{2}/{3}] Finished.'.format(curr_idx, length, i+1, pc.size(0)))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization of original point cloud')
    parser.add_argument('--input', default='', type=str, help='Mat file or Mat file path')
    parser.add_argument('--save_dir', default=None, type=str, help='')
    parser.add_argument('--reconstruct_type', default=None, type=str, help='PSR | BPA')
    parser.add_argument('--is_use_est_normal', action='store_true', default=False, help='')
    cfg  = parser.parse_args()

    save_dir = cfg.save_dir or os.path.join(cfg.input, "../Reconstruct_from_mat_"+str(cfg.reconstruct_type))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.splitext(cfg.input)[-1] == '.mat':
        reconstruction_from_mat(cfg.input, cfg.reconstruct_type, save_dir, 1, 1)
    else:
        file_name_list = os.listdir(cfg.input)
        file_name_list.sort()
        cnt = 0
        for input_file in file_name_list:
            cnt+=1
            if os.path.splitext(input_file)[1] == '.mat':
                reconstruction_from_mat(os.path.join(cfg.input, input_file), cfg.reconstruct_type, save_dir, cnt, len(file_name_list))
            else:
                pass


