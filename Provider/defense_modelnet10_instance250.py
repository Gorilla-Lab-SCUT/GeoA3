import os
import os.path
import sys
import numpy as np
import struct
import math

import torch
from torch.utils.data.dataloader import default_collate
from scipy.io import loadmat

ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23, 15]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']


class ModelNet40():
	def __init__(self, advdatadir):
		self.advdatadir = advdatadir
		self.filename = os.listdir(advdatadir)

	def __len__(self):
		return len(self.filename)

	def __getitem__(self, index):
		data = loadmat(os.path.join(self.advdatadir, self.filename[index]))

		pc = torch.FloatTensor(data['adversary_point_clouds'])
		gt_label = data['gt_label']
		attack_label = data['attack_label']

		return [pc, gt_label, attack_label]
