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
w=4
k=4
l=16

data_frac=2
frac_len=8192
decoding_strategy=1
selection_temperature=0.7

scorer_name=output/instruction_following/8b/gen_iter1/value_model

reward_model_path=SiliangZ/RM_Mistral_sft_init_ultrafeedbck_lr_5e6
generator="rollout_iter1_w${w}_k${k}_l${l}_0.6_0.9_${reward_model_path}_${frac_len}_${data_frac}"

save_dir="output/instruction_following/8b/rollout_iter1/gen_${data_frac}_${frac_len}"
model_name=meta-llama/Meta-Llama-3-8B-Instruct


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=1 --world_size=$my_world_size \
    --gen.w=$w --gen.k=$k --gen.l=$l \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --score_model_args "${scorer_name}, 1" \
    --generator=${generator} \
    --decoding_strategy=${decoding_strategy} --selection_temperature=${selection_temperature} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} --batch_size 8 &

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=2 --world_size=$my_world_size \
    --gen.w=$w --gen.k=$k --gen.l=$l \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --score_model_args "${scorer_name}, 1" \
    --generator=${generator} \
    --decoding_strategy=${decoding_strategy} --selection_temperature=${selection_temperature} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} --batch_size 8 &

CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=3 --world_size=$my_world_size \
    --gen.w=$w --gen.k=$k --gen.l=$l \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --score_model_args "${scorer_name}, 1" \
    --generator=${generator} \
    --decoding_strategy=${decoding_strategy} --selection_temperature=${selection_temperature} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} --batch_size 8 &

CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=4 --world_size=$my_world_size \
    --gen.w=$w --gen.k=$k --gen.l=$l \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --score_model_args "${scorer_name}, 1" \
    --generator=${generator} \
    --decoding_strategy=${decoding_strategy} --selection_temperature=${selection_temperature} \
    --output_dir=${save_dir} \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --frac_len ${frac_len} --data_frac ${data_frac} --batch_size 8 &


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



