#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco', 'potsdam']
# method: ['baseline_sup', 'baseline_semi', 'allspark']
# split:
## 1. for 'pascal' select:
###  - original ['92', '183', '366', '732' ,'1464']
###  - augmented ['1_16', '1_4', '1_8', '1_2']
###  - augmented (U2PL splits) ['u2pl_1_16', 'u2pl_1_8', 'u2pl_1_4', 'u2pl_1_2']
## 2. for 'cityscapes' select: ['1_16', '1_4', '1_8', '1_2']
## 3. for 'coco' select: ['1_512', '1_256', '1_128', '1_32']
## 4. for 'potsdam' select: ['standard']


dataset='potsdam'
method='allspark'
split='1_2_patch'
version='0'


config=configs/${dataset}_${method}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$split"_v"$version

mkdir -p $save_path

# python3 -m torch.distributed.launch \
#     --nproc_per_node=$1 \
#     --master_addr=localhost \
#     --master_port=$2 \
export LOCAL_RANK=0
export PYTHONPATH=..
python3 train_$method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path 2>&1 | tee $save_path/$now.log
