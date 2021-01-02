#!/bin/dash;
mlx_script=~/Project/PyTorch/3D_Adversarial/Adversarial_pointcloud/Script/make_watertight.mlx
data_dir=~/Desktop/Meshing/
data_dir=${data_dir}/RA_fail_m

shapes_in_dir=${data_dir}/Mesh
shapes_out_dir=${data_dir}/Mesh_WT

mkdir -p $shapes_out_dir

for i in $( ls $shapes_in_dir);
do
    ~/Project/Package/meshlab/distrib/meshlabserver -i $shapes_in_dir"/"$i -o $shapes_out_dir"/"$i -s $mlx_script
done

