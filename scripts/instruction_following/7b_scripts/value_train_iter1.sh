#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=256gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=train
#SBATCH --requeue
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=h100-8

conda init bash
source ~/.bashrc
module load cuda
conda activate value_train

save_dir="output/instruction_following/8b/gen_iter1"

intial_model=SiliangZ/RM_Mistral_sft_init_ultrafeedbck_lr_5e6
dataset=${save_dir}/train_merge_reward.json
output_dir=${save_dir}/value_model


accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/value_training/c_train.py \
    --query_dataset ${dataset} \
    --ppo.gamma 1 \
    --num_train_epochs 10 \
    --output_dir ${output_dir} \
    --lr 3e-6 \
    --base_model SiliangZ/RM_Mistral_sft_init_ultrafeedbck_lr_5e6 \
    --revision main \
    --reward_model_path ${intial_model} \
    --local_micro_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --params.prompt_len 512 \
    --params.max_response_length 2048 \
    --params.text_field instruction \
    --params.truncate_field instruction \
    --local_rollout_forward_batch_size 8 \
    --deepspeed \
    --track \
    --wandb_entity wandb_name \
    --wandb_project_name IRO_train \
    --run_name "7b_8b_value_8k_iter1" \
    --mse \