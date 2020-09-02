import os
import sys

import numpy as np
from scipy.io import loadmat, savemat

def __farthest_points_normalized(obj_points, num_points, normal):
    first = np.random.randint(len(obj_points))
    selected = [first]
    dists = np.full(shape = len(obj_points), fill_value = np.inf)

    for _ in range(num_points - 1):
        dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
        selected.append(np.argmax(dists))
    res_points = np.array(obj_points[selected])
    res_normal = np.array(normal[selected])

    # normalize the points and faces
    avg = np.average(res_points, axis = 0)
    res_points = res_points - avg[np.newaxis, :]
    dists = np.max(np.linalg.norm(res_points, axis = 1), axis = 0)
    res_points = res_points / dists

    return res_points, res_normal

data_root = "../Data/modelnet10_250instances10000_PointNet.mat"
out_datadir = "../Data"
resample_num = 5000

if not os.path.isfile(data_root):
    assert False, 'No exists .mat file!'

dataset = loadmat(data_root)
data = dataset['data']
normal = dataset['normal']
saved_label = dataset['label']

tmp_data_set = []
tmp_normal_set = []
for j in range(data.shape[0]):
    tmp_data, tmp_normal  = __farthest_points_normalized(data[j].T, resample_num, normal[j].T)
    tmp_data_set.append(tmp_data.T)
    tmp_normal_set.append(tmp_normal.T)
saved_dense_data = np.stack(tmp_data_set)
saved_dense_normal = np.stack(tmp_normal_set)

savemat(os.path.join(out_datadir, data_root.replace("10000", str(resample_num))), {"data": saved_dense_data, 'normal': saved_dense_normal, 'label': saved_label})
