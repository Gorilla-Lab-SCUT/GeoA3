import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from PointNetPP_ssg import PointNet2ClassificationSSG


class PointNet2ClassificationMSG(PointNet2ClassificationSSG):
    def _build_model(self):
        super()._build_model()
        if self.use_normal:
            ori_channel = 3
        else:
            ori_channel = 0
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[ori_channel, 32, 32, 64], [ori_channel, 64, 64, 128], [ori_channel, 64, 96, 128]],
                use_xyz=self.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        )
