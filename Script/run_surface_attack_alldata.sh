saved_dir=python main_attack.py \
    --data_dir_file Data/modelnet40_2111instances10000_PointNet.mat -b 32 --npoint 1024 \
    --arch ${1} --attack ${2} --attack_label ${3} \
    --binary_max_steps 5 --iter_max_steps 500 --lr 0.05 \
    --initial_const 500 \
    --cls_loss_type CE \
    --dis_loss_type CD --dis_loss_weight 1.0 \
    --hd_loss_weight 0.1 \
    --curv_loss_weight 1.0 --curv_loss_knn 16 \
    --uniform_loss_weight 1.0 \
    --is_pro_grad --is_real_offset \
    --cc_linf 0.1 \
    --is_use_lr_scheduler \
    --id 9270

if [ "$?"= "0" ]; then

    saved_dir_pc=${saved_dir}/PC

    python SurfaceAttack/save_pc_to_npy.py \
        --input_pc_dir ${saved_dir_pc}

    saved_dir_pts=${saved_dir}/04_pts

    data_set_substr=${saved_dir##*/}
    data_set_substr=$(echo $data_set_substr | tr -d ".")
    dataset_dir=3rdParty/points2surf/datasets/${data_set_substr}

    mkdir ${dataset_dir}
    mv $saved_dir_pts $dataset_dir
    mv ${dataset_dir}/04_pts/testset.txt $dataset_dir

    cd 3rdParty/points2surf

    python full_eval.py \
        --indir 'datasets' \
        --outdir 'results' \
        --modeldir 'models' \
        --dataset ${data_set_substr}/testset.txt \
        --models 'p2s_max' \
        --modelpostfix '_model_249.pth' \
        --batchSize 501 \
        --workers 7 \
        --cache_capacity 5 \
        --query_grid_resolution 256 \
        --epsilon 3 \
        --certainty_threshold 13 \
        --sigma 5

    cd ../../

    output_mesh_dir=3rdParty/points2surf/results/p2s_max_model_249/${data_set_substr}/rec/mesh
    save_mesh_dir=${saved_dir}/Reconstruct_from_p2s
    mv $output_mesh_dir $save_mesh_dir

    python eval_from_pc_file.py \
        --datadir ${save_mesh_dir} --is_mesh --npoint 1024 --arch ${1}
else
   echo "Not successfully run main_attack.py" 1>&2
   exit 1
fi


