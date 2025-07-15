conda init bash
source ~/.bashrc
module load cuda
source activate decoding

my_world_size=1
save_dir="output/instruction_following/70b/base/"

model_name=meta-llama/Meta-Llama-3-70B-Instruct

PYTHONPATH=. python scripts/instruction_following/generate_iro.py \
    --rank=1 --world_size=$my_world_size \
    --gen.w=1 --gen.k=1 --gen.l=None \
    --gen.others.temperature 0.6 --gen.others.top_p 0.9 \
    --model_name=${model_name} \
    --scorer_name="HuggingFaceH4/zephyr-7b-beta" \
    --output_dir=${save_dir} &

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


conda activate alpacaEval


alpaca_eval --model_outputs ${save_dir}/train_merge_reward.json  \
    --output_path ${save_dir} \
    --annotators_config 'alpaca_eval_gpt4_turbo_fn' \
