#For PointNet
python main_train.py --datadir /data/modelnet40_normal_resampled/ --npoint 1024 --arch PointNet --epochs 200

python Provider/gen_data_mat.py --out_datadir ./Data -outc 10 -outn 25 --npoints 1024

python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 --arch PointNet \
	--attack GeoA3 --attack_label All --binary_max_steps 10 --iter_max_steps 500 \
	--cls_loss_type CE --dis_loss_type CD --dis_loss_weight 1.0 --hd_loss_weight 0.1 --curv_loss_weight 1.0 --curv_loss_knn 16 \
	--lr 0.01

python defense.py --datadir Exps/PointNet_npoint1024/All/Pertub_0_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16/Mat \
	--npoint 1024 --arch PointNet \
	--defense_type outliers_fixNum --drop_num 128

#For PointNetPP
pip install Model/Pointnet2_PyTorch/pointnet2_ops_lib/.

python main_train.py --datadir /data/modelnet40_normal_resampled/ --npoint 1024 --arch PointNetPP --epochs 200 \
	--lr 0.001 --bn_momentum 0.5 -b 16

python Provider/gen_data_mat.py --out_datadir ./Data -outc 10 -outn 25 --npoint 1024 --arch PointNetPP

python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNetPP.mat --npoint 1024 --arch PointNetPP \
	--attack GeoA3 --attack_label All --binary_max_steps 10 --iter_max_steps 500 \
	--dis_loss_weight 3.0 --knn_smoothing_loss_weight 5.0 --knn_smoothing_k 5 --knn_threshold_coef 1.1 \
    --cc_linf 0.1 --lr 0.01 \

python defense.py --datadir Exps/PointNetPP_npoint1024/All/Pertub_0_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16/Mat \
	--npoint 1024 --arch PointNetPP \
	--defense_type outliers_fixNum --drop_num 128


python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNetPP.mat --npoint 1024 --arch PointNetPP \
	--attack GeoA3 --attack_label All --binary_max_steps 1 --iter_max_steps 2500 \
	--cls_loss_type CE --dis_loss_type CD --dis_loss_weight 1.0 --hd_loss_weight 0.1 --curv_loss_weight 1.0 --curv_loss_knn 16 \
	--lr 0.01
