source activate decoding

my_world_size=1
data_frac=1
frac_len=300
N=1


save_dir="output/tldr/1b_1b/dpo"

base_model=vwxyzjn/EleutherAI_pythia-1b-deduped__dpo__tldr
revision=dpo__44413__1707268320

for local_index in $(seq 1 $((my_world_size))); do
    CUDA_VISIBLE_DEVICES=$((local_index - 1)) PYTHONPATH=. python scripts/tldr/generate_iro.py \
        --rank=$local_index --world_size=$my_world_size \
        --gen.w=1 --gen.k=$N --gen.l=None \
        --model_name=${base_model} \
        --revision=$revision \
        --split="test" \
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

conda activate alpacaEval

python scripts/tldr/gpt_evaluate.py \
    --input_path ${save_dir} \