# Geometry-Aware Generation of Adversarial Point Clouds
By Yuxin Wen, Jiehong Lin, Ke Chen, C. L. Philip Chen, Kui Jia.

## Introduction
This repository contains the implementation of our paper <https://arxiv.org/abs/1912.11171>.
(This documentation is still under construction, please refer to our paper for more details)

<!--
## Requirements
* A computer running on Linux
* NVIDIA GPU and NCCL
* Python 3.6 or higher version
* Pytorch 1.1 or higher version

## Usage
Use `python main.py` to train a new model. Here is an example settings for PointNet:
```
python main_train.py --datadir /data/modelnet40_normal_resampled/ --npoint 1024 --arch PointNet --epochs 200
```
Note that `/data/modelnet40_normal_resampled/` is the path of your ModelNet40 dataset. We use the dataset (ModelNet40) of [PointNet++](https://github.com/charlesq34/pointnet2) which can be download [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).

And then use `python attack.py` to generate adversarial point clouds:
```
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 --arch PointNet \
--attack GeoA3 --attack_label All --binary_max_steps 10 --iter_max_steps 500 \
--cls_loss_type CE --dis_loss_type CD --dis_loss_weight 1.0 --hd_loss_weight 0.1 --curv_loss_weight 1.0 --curv_loss_knn 16 \
--lr 0.01
```

`defense.py` is used for evaluating the defense results on the corresponding adversarial point clouds:
```
python defense.py --datadir Exps/PointNet_npoint1024/All/Pertub_0_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16/Mat \
	--npoint 1024 --arch PointNet \
	--defense_type outliers_fixNum --drop_num 128
```
-->
## Citation
If you use this method or this code in your paper, then please cite it:

```
@article{wen2019geometry,
  title={Geometry-aware Generation of Adversarial and Cooperative Point Clouds},
  author={Wen, Yuxin and Lin, Jiehong and Chen, Ke and Jia, Kui},
  journal={arXiv preprint arXiv:1912.11171},
  year={2019}
}
```
