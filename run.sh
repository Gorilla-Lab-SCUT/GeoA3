# For PointNet
python main_train.py --datadir /data/modelnet40_normal_resampled/ --npoint 1024 --arch PointNet --epochs 200

python Provider/gen_data_mat.py --out_datadir ./Data -outc 10 -outn 25 --npoint 1024
# if want dense npoints
#python Provider/gen_data_mat.py --out_datadir ./Data -outc 10 -outn 25 --npoint 1024 --is_using_virscan --dense_npoints 10000

## GeoA3 attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 \
    --arch PointNet --attack GeoA3 --attack_label All \
    --binary_max_steps 10 --iter_max_steps 500 --lr 0.01 \
    --cls_loss_type CE \
    --dis_loss_type CD --dis_loss_weight 1.0 \
    --hd_loss_weight 0.1 \
    --curv_loss_weight 1.0 --curv_loss_knn 16
## Xiang's attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 \
    --arch PointNet --attack Xiang --attack_label All \
    --binary_max_steps 10 --iter_max_steps 500 --lr 0.01 \
    --cls_loss_type Margin --confidence 0 \
    --dis_loss_type L2 --dis_loss_weight 1.0
## Robust Attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 \
    --dense_data_dir_file Data/modelnet10_250instances10000_PointNet.mat --is_save_normal \
    --arch PointNet --attack RA --attack_label All \
    --binary_max_steps 5 --iter_max_steps 2500  --lr 1e-3 \
    --cls_loss_type Margin --confidence 15 \
    --dis_loss_type CD --dis_loss_weight 3.0 \
    --knn_smoothing_loss_weight 5.0 --knn_smoothing_k 5 --knn_threshold_coef 1.1 \
    --cc_linf 0.1
## Liu's attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 \
    --arch PointNet --attack Liu --attack_label All \
    --iter_max_steps 500 --step_alpha 5

## Defense
python defense.py --datadir Exps/PointNet_npoint1024/All/Pertub_0_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16/Mat \
    --npoint 1024 --arch PointNet \
    --defense_type outliers_fixNum --drop_num 128



# For PointNetPP
pip install Model/Pointnet2_PyTorch/pointnet2_ops_lib/.

python main_train.py --datadir /data/modelnet40_normal_resampled/ --npoint 1024 --arch PointNetPP --epochs 200 \
    --lr 0.001 --bn_momentum 0.5 -b 16

python Provider/gen_data_mat.py --out_datadir ./Data -outc 10 -outn 25 --npoint 1024 --arch PointNetPP
# if want dense npoints
#python Provider/gen_data_mat.py --out_datadir ./Data -outc 10 -outn 25 --npoint 1024 --is_using_virscan --dense_npoints 10000 --arch PointNetPP

## GeoA3 attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNetPP.mat --npoint 1024 \
    --arch PointNetPP --attack GeoA3 --attack_label All \
    --binary_max_steps 10 --iter_max_steps 500 --lr 1e-3 \
    --cls_loss_type CE \
    --dis_loss_type CD --dis_loss_weight 1.0 \
    --hd_loss_weight 0.1 \
    --curv_loss_weight 1.0 --curv_loss_knn 16
## Xiang's attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNetPP.mat --npoint 1024 \
    --arch PointNetPP --attack Xiang --attack_label All \
    --binary_max_steps 10 --iter_max_steps 500 --lr 1e-3 \
    --cls_loss_type Margin --confidence 0 \
    --dis_loss_type L2 --dis_loss_weight 1.0
## Robust Attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNetPP.mat --npoint 1024 \
    --dense_data_dir_file Data/modelnet10_250instances10000_PointNetPP.mat --is_save_normal \
    --arch PointNetPP --attack RA --attack_label All \
    --binary_max_steps 5 --iter_max_steps 2500  --lr 1e-3 \
    --cls_loss_type Margin --confidence 15 \
    --dis_loss_type CD --dis_loss_weight 3.0 \
    --knn_smoothing_loss_weight 5.0 --knn_smoothing_k 5 --knn_threshold_coef 1.1 \
    --cc_linf 0.1
## Liu's attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNetPP.mat --npoint 1024 \
    --arch PointNetPP --attack Liu --attack_label All \
    --iter_max_steps 500 --step_alpha 5 \
    --id 3

## Defense
python defense.py --datadir Exps/PointNetPP_npoint1024/All/Pertub_0_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16/Mat \
    --npoint 1024 --arch PointNetPP \
    --defense_type outliers_fixNum --drop_num 128



# For DGCNN
## GeoA3 attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_DGCNN.mat --npoint 1024 \
    --arch DGCNN --attack GeoA3 --attack_label All \
    --binary_max_steps 10 --iter_max_steps 500 --lr 1e-3 \
    --cls_loss_type CE \
    --dis_loss_type CD --dis_loss_weight 1.0 \
    --hd_loss_weight 0.1 \
    --curv_loss_weight 1.0 --curv_loss_knn 16
## Xiang's attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_DGCNN.mat --npoint 1024 \
    --arch DGCNN --attack Xiang --attack_label All \
    --binary_max_steps 10 --iter_max_steps 500 --lr 1e-3 \
    --cls_loss_type Margin --confidence 0 \
    --dis_loss_type L2 --dis_loss_weight 1.0
## Robust Attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_DGCNN.mat --npoint 1024 \
    --dense_data_dir_file Data/modelnet10_250instances10000_DGCNN.mat --is_save_normal \
    --arch DGCNN --attack RA --attack_label All \
    --binary_max_steps 5 --iter_max_steps 2500  --lr 1e-3 \
    --cls_loss_type Margin --confidence 15 \
    --dis_loss_type CD --dis_loss_weight 3.0 \
    --knn_smoothing_loss_weight 5.0 --knn_smoothing_k 5 --knn_threshold_coef 1.1 \
    --cc_linf 0.1
## Liu's attack
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_DGCNN.mat --npoint 1024 \
    --arch DGCNN --attack Liu --attack_label All \
    --iter_max_steps 500 --step_alpha 5 \
    --id 3




# For evaluating the \beta
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 --arch PointNet --attack GeoA3 --attack_label All --binary_max_steps 1 --iter_max_steps 2500 --initial_const x --lr 0.01 --cls_loss_type CE --dis_loss_type CD --dis_loss_weight 1.0 --hd_loss_weight 0.1 --curv_loss_weight 1.0 --curv_loss_knn 16 --id 1


# Mesh attack
## On mesh
python main_attack.py --data_dir_file Data/modelnet10_250instances_mesh_PointNet.mat --npoint 1024 -b 1 \
    --arch PointNet --attack GeoA3_mesh --attack_label Untarget \
    --binary_max_steps 10 --iter_max_steps 500 --lr 1e-3 --is_use_lr_scheduler \
    --cls_loss_type CE \
    --dis_loss_type CD --dis_loss_weight 1.0 \
    --hd_loss_weight 0.1 \
    --curv_loss_weight 0.1 --curv_loss_knn 16 \
    --laplacian_loss_weight 0.1 \
    --edge_loss_weight 0.1 \
    --is_partial_var --knn_range 3 \
    --id 4

## Reconstruction
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 \
    --dense_data_dir_file Data/modelnet10_250instances10000_PointNet.mat --is_save_normal \
    --arch PointNet --attack GeoA3 --attack_label All \
    --binary_max_steps 10 --iter_max_steps 500 --lr 0.01 \
    --cls_loss_type CE \
    --dis_loss_type CD --dis_loss_weight 1.0 \
    --hd_loss_weight 0.1 \
    --curv_loss_weight 1.0 --curv_loss_knn 16 \
    --is_pro_grad --id 7



