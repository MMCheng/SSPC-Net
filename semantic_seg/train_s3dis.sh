#!/bin/sh
export PYTHONPATH=./
PYTHON=python
now=$(date +"%Y%m%d_%H%M%S")
today=$(date +"%Y%m%d")

s3dis_path=$1

train_gpu=0
val_fold=5 # test area for s3dis

exp_name=${today}
odir=results/s3dis/${exp_name}/cv${val_fold}


mkdir -p ${odir}
CUDA_VISIBLE_DEVICES=${train_gpu} ${PYTHON} -u main.py \
    --dataset 's3dis' \
    --S3DIS_PATH ${s3dis_path} \
    --cvfold $val_fold \
    --data_mode voxel \
    --epochs 400 \
    --batch_size 2 \
    --lr 0.01 \
    --lr_steps '[275,320]' \
    --nworkers 8 \
    --test_nth_epoch 10 \
    --model_config 'gru_10,f_13' \
    --ptn_nfeat_stn 14 \
    --extension_th 0.9\
    --ext_epoch_gap 40 \
    --ext_drop 0.95 \
    --odir $odir 2>&1 | tee $odir/train-$now.log