conda init bash
source ~/.bashrc

conda activate decoding

my_world_size=2
data_frac=1
frac_len=8192
save_dir="output/tldr/1b_1b/rollout_iter0/gen_${data_frac}_${frac_len}"

base_model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr
# base_model=vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr

for local_index in $(seq 1 $((my_world_size))); do
    CUDA_VISIBLE_DEVICES=$((local_index - 1)) PYTHONPATH=. python scripts/tldr/generate_iro.py \
        --rank=$local_index --world_size=$my_world_size \
        --gen.w=1 --gen.k=1 --gen.l=None \
        --model_name=${base_model} \
        --revision=sft__44413__1708611267 \
        --split="train" \
        --output_dir=${save_dir} \
        --frac_len ${frac_len} --data_frac ${data_frac} &
    echo "Launched process on GPU $local_index"
done

wait

python scripts/instruction_following/utils/merge_data.py \
    --base_path ${save_dir} \
    --output_dir ${save_dir}/merge_data.json \
    --num_datasets ${my_world_size}

conda activate value_train

accelerate launch --main_process_port 29700 scripts/tldr/get_reward.py \
    --data_path ${save_dir}/merge_data.json \
    --output_path ${save_dir}/train_merge_reward.json \
    --reward_name_or_path vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr

