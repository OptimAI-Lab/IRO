conda init bash
source ~/.bashrc
module load cuda
conda activate decoding

my_world_size=1

output_dir="output/instruction_following/8b/bon/n16_0.6_0.9"

PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=1 --world_size=$my_world_size \
    --gen.w=1 --gen.k=16 --gen.l=None \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
    --generator="bon16_0.6_0.9" \
    --output_dir=${output_dir} &

wait

python scripts/instruction_following/utils/merge_data.py \
    --base_path ${output_dir} \
    --output_dir ${output_dir}/merge_data.json \
    --num_datasets ${my_world_size}

reward_model_path=SiliangZ/RM_Mistral_sft_init_ultrafeedbck_lr_5e6

conda activate value_train

accelerate launch --main_process_port 29600 scripts/instruction_following/get_reward.py \
    --data_path ${output_dir}/train_merge_reward.json \
    --output_path ${output_dir}/train_merge_reward_eval.json \
    --reward_name_or_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \


conda activate alpacaEval

alpaca_eval --model_outputs ${output_dir}/train_merge_reward.json  \
    --output_path ${output_dir} \
    --annotators_config 'alpaca_eval_gpt4_turbo_fn' \