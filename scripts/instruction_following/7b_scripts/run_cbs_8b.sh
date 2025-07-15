conda init bash
source ~/.bashrc
module load cuda


conda activate decoding
my_world_size=2

output_dir="output/instruction_following/8b/cbs/w4k4l16"

reward_model_path=SiliangZ/RM_Mistral_sft_init_ultrafeedbck_lr_5e6

PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=1 --world_size=1 \
    --gen.w=4 --gen.k=4 --gen.l=16 \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
    --scorer_name="HuggingFaceH4/zephyr-7b-beta" \
    --generator="weak-to-strong-search-4-4-16" \
    --output_dir="output/instruction_following/8b/cbs/w4k4l16"

wait

python scripts/instruction_following/utils/merge_data.py \
    --base_path output/instruction_following/8b/cbs/w4k4l16 \
    --output_dir output/instruction_following/8b/cbs/w4k4l16/merge_data.json \
    --num_datasets ${my_world_size}

conda activate value_train

accelerate launch --main_process_port 29600 scripts/instruction_following/get_reward.py \
    --data_path ${output_dir}/merge_data.json \
    --output_path ${output_dir}/train_merge_reward.json \
    --reward_name_or_path ${reward_model_path} \
    --generator "bon16_0.6_0.9_8b" \
    --output_first

conda activate value_train

accelerate launch --main_process_port 29600 scripts/instruction_following/get_reward.py \
    --data_path ${output_dir}/train_merge_reward.json \
    --output_path ${output_dir}/train_merge_reward_eval.json \
    --reward_name_or_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \


conda activate alpacaEval

alpaca_eval --model_outputs output/instruction_following/8b/cbs/w4k4l16/train_merge_reward.json  \
    --output_path output/instruction_following/8b/cbs/w4k4l16 \
    --annotators_config 'alpaca_eval_gpt4_turbo_fn' \