#!/bin/bash

root='/home/zhangxin/Codes/EMIP/snapshots/log/'

model_names=(EMIP)

for MODEL_NAME in ${model_names[*]}
do
  echo $MODEL_NAME
  echo $root$MODEL_NAME
  python moca_evaluator.py \
  --pred_root $root/MoCA_test \
  --gt_root /home/zhangxin/Datasets/VCOD/MoCA_Video/TestDataset_per_sq/ --data_lst MoCA_test --model_lst $MODEL_NAME \

  python moca_evaluator.py \
  --pred_root $root/CAD_eval \
  --gt_root /home/zhangxin/Datasets/VCOD/CamouflagedAnimalDataset/ --data_lst CAD_eval --model_lst $MODEL_NAME \

done