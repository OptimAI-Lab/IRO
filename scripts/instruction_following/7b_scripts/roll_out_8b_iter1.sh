#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=256gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=ir
#SBATCH --requeue
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=h100-8

conda init bash
source ~/.bashrc
module load cuda
source activate decoding

my_world_size=4
data_frac=1
frac_len=8192
save_dir="output/instruction_following/8b/iter0_0.6_0.9/gen_${data_frac}_${frac_len}"
model_name=meta-llama/Meta-Llama-3-8B-Instruct


CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=1 --world_size=$my_world_size \
    --gen.w=1 --gen.k=1 --gen.l=None \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} &

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=2 --world_size=$my_world_size \
    --gen.w=1 --gen.k=1 --gen.l=None \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} &

CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=3 --world_size=$my_world_size \
    --gen.w=1 --gen.k=1 --gen.l=None \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} &

CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=4 --world_size=$my_world_size \
    --gen.w=1 --gen.k=1 --gen.l=None \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} &


wait

python scripts/instruction_following/utils/merge_data.py \
    --base_path ${save_dir} \
    --output_dir ${save_dir}/merge_data.json \
    --num_datasets ${my_world_size}

conda activate value_train

accelerate launch --main_process_port 29700 scripts/instruction_following/get_reward.py \
    --data_path ${save_dir}/merge_data.json \
    --output_path ${save_dir}/train_merge_reward.json \
    --reward_name_or_path SiliangZ/RM_Mistral_sft_init_ultrafeedbck_lr_5e6 \



