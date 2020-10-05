#!/bin/dash;
mlx_script=~/Project/PyTorch/3D_Adversarial/Adversarial_pointcloud/Script/make_watertight.mlx
data_dir=~/Project/PyTorch/3D_Adversarial/Adversarial_pointcloud/Exps/
data_dir=${data_dir}/PointNet_npoint1024/Untarget/GeoA3_9271_BiStep5_IterStep500_Optadam_Lr0.01_Initcons500.0_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16_UniLoss1.0_LRExp_ProGradRO_cclinf0.1

shapes_in_dir=${data_dir}/Mesh
shapes_out_dir=${data_dir}/Mesh_WT

mkdir -p $shapes_out_dir

for i in $( ls $shapes_in_dir);
do
    ~/Project/Package/meshlab/distrib/meshlabserver -i $shapes_in_dir"/"$i -o $shapes_out_dir"/"$i -s $mlx_script
done

python eval_from_pc_file.py \
    --datadir ${shapes_out_dir} --is_mesh --npoint 1024 --arch PointNet
