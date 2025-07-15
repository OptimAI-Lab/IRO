from typing import Optional, Text

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available
from typing import List, Union
from src.inference_time_alignment.scorers import ImplicitValueScorer, WeightedMultiValueScorer
from scripts.utils import get_local_model_path
import os   
import json
from typing import Dict
from datasets import Dataset
from dataclasses import dataclass, field

def get_chat_prompt_template(model_name: Text, tokenizer: PreTrainedTokenizer) -> Text:
    if model_name in ("meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        ) + " " # add a trailing space
    elif model_name in ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        )
    elif model_name in ("HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/mistral-7b-sft-beta"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        )
    elif model_name in ("berkeley-nest/Starling-LM-7B-alpha", "openchat/openchat_3.5"):
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        ) + " " # add a trailing space
    elif model_name in ("allenai/tulu-2-dpo-7b", "allenai/tulu-2-7b"):
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        )
    # modify here to support your customized models
    else:
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        )


def get_scorer(
    scorer_name: Union[str, List[str]], 
    load_in_4bit: Optional[bool] = False, 
    use_flash_attention_2: Optional[bool] = False,
    batch_size: Optional[int] = None,
    combine_type: Optional[str] = "add",
    beta: Optional[float] = None,
):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit and is_bitsandbytes_available() else None,
        # "attn_implementation": "flash_attention_2" if use_flash_attention_2 and is_flash_attn_2_available() else None,
    }

    # map score_name to (tuned name and untuned name)
    scorer_map = {
        "HuggingFaceH4/zephyr-7b-beta": ("HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/mistral-7b-sft-beta"),
        "berkeley-nest/Starling-LM-7B-alpha": ("berkeley-nest/Starling-LM-7B-alpha", "openchat/openchat_3.5"),
        "allenai/tulu-2-dpo-7b": ("allenai/tulu-2-dpo-7b", "allenai/tulu-2-7b"),
        # modify here to support your customized models
    }
    tuned_name = scorer_map[scorer_name][0]
    untuned_name = scorer_map[scorer_name][1]

    tuned_model = AutoModelForCausalLM.from_pretrained(tuned_name, **model_kwargs)
    untuned_model = AutoModelForCausalLM.from_pretrained(untuned_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(tuned_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    implicit_value_scorer = ImplicitValueScorer(
        model=tuned_model,
        ref_model=untuned_model,
        tokenizer=tokenizer,
        model_prompt_template=get_chat_prompt_template(tuned_name, tokenizer),
        ref_model_prompt_template=get_chat_prompt_template(untuned_name, tokenizer),
    )
    return implicit_value_scorer
    

@dataclass
class ScoreModelConfig:
    path: str
    weight: float = 1.0
    device_map: str = "auto"
    
    @classmethod
    def from_str(cls, s: str) -> "ScoreModelConfig":
        """path,weight
        example: "model1,0.5,true,false" """
        parts = s.split(",")
        return cls(
            path=parts[0],
            weight=float(parts[1]) if len(parts) > 1 else 1.0,
        )
        
def parse_score_models(args) -> List[ScoreModelConfig]:
    """解析所有评分模型配置"""
    models = []
    
    for model_arg in args.score_model_args:
        models.append(ScoreModelConfig.from_str(model_arg))
    
    return models

def get_weighted_scorer(
    score_models: List[ScoreModelConfig], 
    load_in_4bit: Optional[bool] = False, 
    use_flash_attention_2: Optional[bool] = False,
    batch_size: Optional[int] = None,
):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit and is_bitsandbytes_available() else None,
        "attn_implementation": None,
    }
    
    if isinstance(score_models, list):
        critics = []
        betas = []
        tokenizer = AutoTokenizer.from_pretrained(score_models[0].path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        for score_model in score_models:
            critic = AutoModelForSequenceClassification.from_pretrained(score_model.path, **model_kwargs)
            critics.append(critic)
            betas.append(score_model.weight)

        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        value_scorer = WeightedMultiValueScorer(
            models=critics,
            betas=betas,
            tokenizer=tokenizer,
            batch_size=batch_size,
            model_prompt_template="{raw_prompt}",
        )
        
    return value_scorer



def get_evaluator(evaluator_name: Text):
    from scripts.instruction_following.utils.evaluators.starlingrm import StarlingRMEvaluator
    from scripts.instruction_following.utils.evaluators.ultrarm import UltraRMEvaluator
    if evaluator_name == "Nexusflow/Starling-RM-34B":
        return StarlingRMEvaluator()
    elif evaluator_name == "openbmb/UltraRM-13b":
        return UltraRMEvaluator()
    else:
        raise NotImplementedError


def get_dataset(dataset_name: Optional[Text] = "tatsu-lab/alpaca_eval", tokenizer: PreTrainedTokenizer = None, script_args=None):
    if dataset_name == "tatsu-lab/alpaca_eval":
        dataset = load_dataset(dataset_name, split="eval").rename_columns({"instruction":"raw_prompt"})
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        dataset = dataset.filter(lambda x: len(tokenizer(x["prompt"])["input_ids"]) <= script_args.max_prompt_len)
        dataset = dataset.rename_column('prompt', 'raw_prompt')
    else:
        dataset = load_dataset("json", data_files=dataset_name, split="train").rename_columns({"prompt":"raw_prompt"})
    data_frac = script_args.data_frac
    dataset = dataset.shuffle(seed=script_args.seed)
    if script_args.frac_len > 0:
        sub_len = script_args.frac_len
        start_idx = sub_len * data_frac
        end_idx = min(sub_len * (data_frac + 1), len(dataset))

        dataset = dataset.select(range(start_idx, end_idx))
        
    return dataset


def get_checkpoint_path(output_dir: str, rank: int, world_size: int) -> str:
    """Get the checkpoint file path."""
    checkpoint_name = f"checkpoint_{rank:05d}-{world_size:05d}.json"
    return os.path.join(output_dir, "checkpoints", checkpoint_name)

def load_checkpoint(checkpoint_path: str) -> tuple[List[Dict], int]:
    """Load existing results and calculate the start index from results length."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data.get("results", [])
            return results, len(results)
    return [], 0

def save_checkpoint(checkpoint_path: str, results: List[Dict]):
    """Save current results to checkpoint file."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_data = {
        "results": results
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)