import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Text
import json
import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available
from datasets import Dataset

from src.inference_time_alignment.decoders.cbs_modified import CBSPosthocGenerationMixin
from src.inference_time_alignment.utils import extract_responses
from scripts.instruction_following.utils import *
from scripts.utils import (
    set_seeds, get_local_model_path, split_dataset_per_rank,
    get_output_path, INT_INFINITY, GenConfig
)
import warnings
warnings.filterwarnings("ignore", message="Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.")
import logging
logging.getLogger().setLevel(logging.ERROR)  # Suppress warning logs
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

@dataclass
class CheckpointConfig:
    checkpoint_interval: int = 5  # Save every N samples
    checkpoint_dir: str = "checkpoints"

@dataclass
class CBSGenConfig:
    # CBS related args (default to disable CBS)
    w: Optional[int] = 2  # n. hypotheses to keep (beam width)
    k: Optional[int] = 2  # n. successors per hypothethis
    l: Optional[int] = 10 # chunk length
    # other args
    others: GenConfig = field(default_factory=lambda: GenConfig(max_new_tokens=2048, temperature=0.7, top_p=1.0))

    def __post_init__(self):
        if self.l == None: self.l = INT_INFINITY


@dataclass
class ScriptArguments:
    model_name:            Optional[Text] = "meta-llama/Meta-Llama-3-8B-Instruct"
    scorer_name:           Optional[Text] = None
    ref_score:             Optional[Text] = None
    dataset_name:          Optional[Text] = "tatsu-lab/alpaca_eval"
    output_dir:            Optional[Text] = "tmp/instruction_following/cbs/gen"
    overwrite:             Optional[bool] = True
    rank:                  Optional[int]  = 1 # one-based indexing
    world_size:            Optional[int]  = 1
    seed:                  Optional[int]  = 1
    load_in_4bit:          Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True
    gen:  CBSGenConfig = field(default_factory=lambda: CBSGenConfig()) # cbs related configs
    generator:             Optional[str] = None
    max_prompt_len:        Optional[int] = 512
    frac_len:              Optional[int] = -1
    data_frac:             Optional[int] = 0
    debug:                 Optional[int] = 0
    batch_size:            Optional[int] = None
    decoding_strategy:     Optional[int] = 1
    selection_temperature: Optional[float] = 1.0
    combine_type:          Optional[str] = "implicit"
    beta:                 Optional[float] = 1.0
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    score_model_args: List[str] = field(default_factory=list, metadata={"nargs": "+"})
    


script_args = tyro.cli(ScriptArguments)
print(script_args)
set_seeds(script_args.seed)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
prompt_template = get_chat_prompt_template(script_args.model_name, tokenizer)
# load dataset
print(f"loading dataset {script_args.dataset_name} ...")
dataset = get_dataset(script_args.dataset_name, tokenizer, script_args)
# split dataset by rank and append rank suffix to output path, e.g., "00001-00008.jsonl"
dataset = split_dataset_per_rank(dataset, script_args.rank, script_args.world_size)
output_path = get_output_path(script_args.output_dir, script_args.rank, script_args.world_size)
# skip if previous generation result exists and we do not want to overwrite it
if os.path.exists(output_path) and not script_args.overwrite: exit()
os.makedirs(script_args.output_dir, exist_ok=True)
# load base model, tokenizer and prompt template for base model
print(f"loading base model {script_args.model_name} ...")
base = AutoModelForCausalLM.from_pretrained(
    # get_local_model_path(script_args.model_name),
    script_args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True) if script_args.load_in_4bit and is_bitsandbytes_available() else None,
    attn_implementation="flash_attention_2" if script_args.use_flash_attention_2 and is_flash_attn_2_available() else None,
)

# get cbs model
cbs_model = CBSPosthocGenerationMixin(base, tokenizer)
# get scorer
if script_args.gen.w * script_args.gen.k > 1 and script_args.gen.l < INT_INFINITY:
    print(f"loading scorer ...")
    if script_args.scorer_name == "HuggingFaceH4/zephyr-7b-beta":
        scorer = get_scorer(
            scorer_name=script_args.scorer_name,
            load_in_4bit=script_args.load_in_4bit,
            use_flash_attention_2=script_args.use_flash_attention_2,
            batch_size=script_args.batch_size,
            combine_type=script_args.combine_type,
        )
    else:
        script_args.score_models = parse_score_models(script_args)
        scorer = get_weighted_scorer(
            score_models=script_args.score_models,
            load_in_4bit=script_args.load_in_4bit,
            use_flash_attention_2=script_args.use_flash_attention_2,
            batch_size=script_args.batch_size,
        )
else:
    print("CBS is disabled, no need to load scorer.")
    scorer = None

#-----------------------------------------------------------------------------#
#---------------------------------- search -----------------------------------#
#-----------------------------------------------------------------------------#
# Create checkpoint directory
checkpoint_path = get_checkpoint_path(script_args.output_dir, script_args.rank, script_args.world_size)
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Load existing results if any
results, start_idx = load_checkpoint(checkpoint_path)
print(f"Resuming from index {start_idx} with {len(results)} existing results")

# for raw_prompt in tqdm.tqdm(dataset["raw_prompt"]):
try:
    for idx, raw_prompt in enumerate(tqdm.tqdm(dataset["raw_prompt"][start_idx:])):
        prompt = prompt_template.format(raw_prompt=raw_prompt)
        prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if scorer is not None:
            scorer.set_raw_prompt(raw_prompt)
        else:
            scorer = None
        outputs = cbs_model.search(
            input_ids=prompt_tokenized["input_ids"].cuda(),
            attention_mask=prompt_tokenized["attention_mask"].cuda(),
            scorer=scorer,
            split_by_prompt_text=False,
            w=script_args.gen.w,
            k=script_args.gen.k,
            l=script_args.gen.l, 
            **asdict(script_args.gen.others),
            decoding_strategy=script_args.decoding_strategy,
            selection_temperature=script_args.selection_temperature,
        )
        outputs = outputs.squeeze(1)
        response = extract_responses(outputs, tokenizer, prompt_len=prompt_tokenized["input_ids"].size(1))
        results.append({
            "instruction": raw_prompt,
            "output": response,
            "generator": f"{script_args.model_name}({str(script_args.generator)})",
            "datasplit": "eval"
        })
        
        # Save checkpoint every N samples
        current_idx = start_idx + idx
        if (current_idx + 1) % script_args.checkpoint.checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, results)
            # print(f"Checkpoint saved at index {current_idx + 1}")

except KeyboardInterrupt:
    print("\nInterrupted by user. Saving checkpoint...")
    save_checkpoint(checkpoint_path, results)
    print(f"Checkpoint saved at index {start_idx + len(results)}")
    raise

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("Saving checkpoint...")
    save_checkpoint(checkpoint_path, results)
    print(f"Checkpoint saved at index {start_idx + len(results)}")
    raise

# Save final results
print(f"Saving final output to {output_path} ...")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Clean up checkpoint if everything completed successfully
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("Checkpoint file removed after successful completion")
