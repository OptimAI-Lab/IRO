conda init bash
source ~/.bashrc
module load cuda
conda activate value_train

output_dir=output/tldr/1b_1b/rollout_iter2/gen_2_8192
rm_revision=main
critc_model=output/tldr/1b_1b/rollout_iter1/gen_1_8192/value_model_8k


accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    scripts/value_training/c_train.py \
    --query_dataset ${output_dir}/train_merge_reward.json \
    --ppo.gamma 1 \
    --num_train_epochs 10 \
    --output_dir ${output_dir}/value_model_8k \
    --lr 3e-6 \
    --base_model EleutherAI/pythia-1b-deduped \
    --revision ${rm_revision} \
    --reward_model_path ${critc_model} \
    --local_micro_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --params.prompt_len 512 \
    --params.max_response_length 53 \
    --params.text_field prompt \
    --params.truncate_field prompt \
    --local_rollout_forward_batch_size 8 \
    --deepspeed \
    --mse \
    --track \
    --wandb_entity wandb_name \
    --wandb_project_name IRO_train \
    --run_name "tldr_value_8k_iter2"