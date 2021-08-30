#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

budget_latency=0.34e-3


save_dir=../save_dir_debug_latency
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,2,1)SuperResK3K3(8,16,2,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

python evolution_search.py --gpu 0 \
  --zero_shot_score Zen \
  --search_space SearchSpace/search_space_XXBL.py \
  --budget_latency ${budget_latency} \
  --max_layers 24 \
  --batch_size 64 \
  --input_image_size 224 \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 1000 \
  --evolution_max_iter 4800 \
  --save_dir ${save_dir}


python analyze_model.py \
  --input_image_size 224 \
  --num_classes 1000 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt

#python train_image_classification.py --dataset cifar100 --num_classes 100 \
#  --dist_mode single --workers_per_gpu 6 \
#  --input_image_size 32 --epochs 1440 --warmup 5 \
#  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
#  --label_smoothing --random_erase --mixup --auto_augment \
#  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
#  --arch Masternet.py:MasterNet \
#  --plainnet_struct_txt ${save_dir}/best_structure.txt \
#  --batch_size_per_gpu 64 \
#  --save_dir ${save_dir}/cifar100_1440epochs
