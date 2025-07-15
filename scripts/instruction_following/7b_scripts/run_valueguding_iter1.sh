#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=128gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=ir
#SBATCH --requeue
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=h100-8

conda init bash
source ~/.bashrc
module load cuda


my_world_size=1
w=4
k=4
l=16

frac_len=-1
decoding_strategy=1
selection_temperature=0.7


scorer_name=output/instruction_following/8b/gen_iter1/value_model

reward_model_path=SiliangZ/RM_Mistral_sft_init_ultrafeedbck_lr_5e6
defined_name=8196_epoch10_ultrafeedbck_lr_5e6_mse_8b_iter1

generator="${defined_name}_w${w}_k${k}_l${l}_0.6_0.9_${reward_model_path}_${frac_len}"


save_dir="output/instruction_following/8b/value_guiding/fullset/${defined_name}_w${w}_k${k}_l${l}_0.6_0.9_${reward_model_path}_${frac_len}_epoch10/decoding_${decoding_strategy}"


conda activate decoding

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=1 --world_size=$my_world_size \
    --gen.w=$w --gen.k=$k --gen.l=$l \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
    --score_model_args "${scorer_name}, 1" \
    --generator=${generator} \
    --decoding_strategy=${decoding_strategy} --selection_temperature=${selection_temperature} \
    --output_dir=${save_dir} --batch_size 16 --frac_len ${frac_len} &

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=2 --world_size=$my_world_size \
    --gen.w=$w --gen.k=$k --gen.l=$l \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
    --score_model_args "${scorer_name}, 1" \
    --generator=${generator} \
    --decoding_strategy=${decoding_strategy} --selection_temperature=${selection_temperature} \
    --output_dir=${save_dir} --batch_size 16 --frac_len ${frac_len} &

wait
echo "Launched process on GPU $local_index"

python scripts/instruction_following/utils/merge_data.py \
    --base_path ${save_dir} \
    --output_dir ${save_dir}/merge_data.json \
    --num_datasets ${my_world_size}

conda activate value_train

accelerate launch --main_process_port 29600 scripts/instruction_following/get_reward.py \
    --data_path ${save_dir}/merge_data.json \
    --output_path ${save_dir}/train_merge_reward.json \
    --reward_name_or_path $reward_model_path \

accelerate launch --main_process_port 29600 scripts/instruction_following/get_reward.py \
    --data_path ${save_dir}/train_merge_reward.json \
    --output_path ${save_dir}/train_merge_reward_eval.json \
    --reward_name_or_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \

conda activate alpacaEval

alpaca_eval --model_outputs ${save_dir}/train_merge_reward.json  \
    --output_path ${save_dir} \
    --reference_outputs ${reference_outputs}