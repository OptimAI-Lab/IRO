conda init bash
source ~/.bashrc
module load cuda
conda activate decoding

my_world_size=2
beta=1

w=1
k=16
l=1

data_frac=1
frac_len=300
decoding_strategy=4

base_model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr
# base_model=vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr

save_dir="output/tldr/1b_1b/cbs_modified/${w}_${k}_${l}_args"
# save_dir="output/tldr/1b_6.9b/cbs_modified/${w}_${k}_${l}_args"

score_model=vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr
score_revision=reward__44413__1706651113


CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/tldr/generate_iro.py \
    --rank=1 --world_size=$my_world_size \
    --gen.w=${w} --gen.k=${k} --gen.l=${l} \
    --gen.others.max_new_tokens=53 \
    --model_name=${base_model} \
    --revision=sft__44413__1708611267 \
    --split="test" \
    --output_dir=${save_dir} \
    --score_model_args="${score_model},${beta}" \
    --score_revision ${score_revision} \
    --decoding_strategy ${decoding_strategy} \
    --frac_len ${frac_len} --data_frac ${data_frac}  --batch_size 16 &

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/tldr/generate_iro.py \
    --rank=2 --world_size=$my_world_size \
    --gen.w=${w} --gen.k=${k} --gen.l=${l} \
    --model_name=${base_model} \
    --revision=sft__44413__1708611267 \
    --split="test" \
    --output_dir=${save_dir} \
    --score_model_args="${score_model},${beta}" \
    --score_revision ${score_revision} \
    --decoding_strategy ${decoding_strategy} \
    --frac_len ${frac_len} --data_frac ${data_frac} --batch_size 16 &


python scripts/instruction_following/utils/merge_data.py \
    --base_path ${save_dir} \
    --output_dir ${save_dir}/merge_data.json \
    --num_datasets ${my_world_size}

conda activate value_train

accelerate launch --main_process_port 29700 scripts/tldr/get_reward.py \
    --data_path ${save_dir}/merge_data.json \
    --output_path ${save_dir}/train_merge_reward.json \
    --reward_name_or_path vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr


accelerate launch --main_process_port 29700 scripts/tldr/get_reward.py \
    --data_path ${save_dir}/train_merge_reward.json \
    --output_path ${save_dir}/eval_reward.json \
    --reward_name_or_path vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr


conda activate alpacaEval

python scripts/tldr/gpt_evaluate.py \
    --input_path ${save_dir} \