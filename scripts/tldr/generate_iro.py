import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Text, List

import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from src.inference_time_alignment.decoders.cbs_modified import CBSPosthocGenerationMixin
from src.inference_time_alignment.utils import extract_responses
from scripts.tldr.utils import get_dataset, parse_score_models, get_scorer
from scripts.instruction_following.utils import load_checkpoint, save_checkpoint, get_checkpoint_path
from scripts.utils import (
    set_seeds, get_local_model_path, split_dataset_per_rank,
    get_output_path, INT_INFINITY, GenConfig
)
from accelerate import Accelerator
accelerator = Accelerator()
import json
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
    others: GenConfig = field(default_factory=lambda: GenConfig(max_new_tokens=53, temperature=0.7, top_p=1.0))

    def __post_init__(self):
        if self.l == None: self.l = INT_INFINITY


@dataclass
class ScriptArguments:
    model_name:            Optional[Text] = "trl-lib/pythia-1b-deduped-tldr-sft"
    revision:              Optional[str] = "main"
    base_prompt_template:  Optional[str]  = "{raw_prompt}" # zero-shot prompt for base models
    dataset_name:          Optional[Text] = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    output_dir:            Optional[Text] = "output/tldr/test"
    split:                 Optional[Text] = "test"
    overwrite:             Optional[bool] = True
    rank:                  Optional[int]  = 1 # one-based indexing
    world_size:            Optional[int]  = 1
    seed:                  Optional[int]  = 1
    load_in_4bit:          Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True
    gen:  CBSGenConfig = field(default_factory=lambda: CBSGenConfig()) # cbs related configs
    data_frac:             Optional[int] = 0
    frac_len:              Optional[int] = 128
    tokenizer_path:        Optional[str] = None
    sample_method:         Optional[str] = 'greedy'
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    score_model_args: List[str] = field(default_factory=list, metadata={"nargs": "+"})
    # score_model_args: List[str] = field(default_factory=list)
    batch_size: Optional[int] = 16
    score_revision: Optional[str] = "main"
    decoding_strategy: Optional[int] = 1
    

script_args = tyro.cli(ScriptArguments)
print(script_args)
set_seeds(script_args.seed)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
if script_args.tokenizer_path is None:
    script_args.tokenizer_path = script_args.model_name

tokenizer = AutoTokenizer.from_pretrained(
    script_args.tokenizer_path,
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# load dataset
print(f"loading dataset {script_args.dataset_name} ...")
dataset = get_dataset(script_args.dataset_name, split=script_args.split, script_args=script_args)
# dataset = dataset.select(range(100))
# split dataset by rank and append rank suffix to output path, e.g., "00001-00008.jsonl"
dataset = split_dataset_per_rank(dataset, script_args.rank, script_args.world_size)
output_path = get_output_path(script_args.output_dir, script_args.rank, script_args.world_size)

if os.path.exists(output_path) and not script_args.overwrite: exit()
os.makedirs(script_args.output_dir, exist_ok=True)

# load base model, tokenizer and prompt template for base model
print(f"loading base model {script_args.model_name} ...")
base = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=None,
    attn_implementation="flash_attention_2" if script_args.use_flash_attention_2 and is_flash_attn_2_available() else None,
    revision=script_args.revision,
)

script_args.score_models = parse_score_models(script_args)
    
# get model
cbs_model = CBSPosthocGenerationMixin(base, tokenizer)

if script_args.gen.w * script_args.gen.k > 1:
    print(f"loading scorer ...")

    scorer = get_scorer(
        score_models=script_args.score_models,
        load_in_4bit=script_args.load_in_4bit,
        batch_size=script_args.batch_size,
        tokenizer=tokenizer,
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

try:
    for idx, item in enumerate(tqdm.tqdm(dataset.select(range(start_idx, len(dataset))), initial=start_idx)):
        reference = item["reference_response"]
        prompt = item["raw_prompt"]
        
        prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = prompt_tokenized["input_ids"].cuda()
        attention_mask = prompt_tokenized["attention_mask"].cuda()

        if scorer is not None:
            scorer.set_raw_prompt(prompt)
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
            selection_temperature=0,
        )
        outputs = outputs.squeeze(1)
        responses = extract_responses(outputs, tokenizer, prompt=prompt)
        results.append({
            "prompt": prompt,
            "output": responses,
            "reference": reference
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
