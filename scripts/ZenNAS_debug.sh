#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

resolution=224
budget_flops=1.8e9
budget_model_size=11e6


save_dir=../save_debug/
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,16,2,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)\
SuperResK3K3(64,128,2,64,1)SuperResK3K3(128,256,2,128,1)\
SuperConvK1BNRELU(256,512,1,1)" > ${save_dir}/initial_plainnet.txt

python evolution_search.py \
  --zero_shot_score Zen \
  --search_space SearchSpace/search_space_XXBL.py \
  --budget_model_size ${budget_model_size} \
  --budget_flops ${budget_flops} \
  --max_layers 24 \
  --batch_size 64 \
  --input_image_size 224 \
  --plainnet_struct_txt ${save_dir}/initial_plainnet.txt \
  --num_classes 1000 \
  --evolution_max_iter 480 \
  --save_dir ${save_dir}

