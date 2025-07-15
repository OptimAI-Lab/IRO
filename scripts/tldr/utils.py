from typing import Optional, Text

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available

from src.inference_time_alignment.scorers import WeightedMultiValueScorer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from dataclasses import dataclass, field
from typing import List, Optional, Text, Dict


def get_dataset(dataset_name: Optional[Text] = "ZHZisZZ/imdb_preference", split="train", script_args=None):
    if dataset_name == "ZHZisZZ/imdb_preference":
        dataset = load_dataset(dataset_name, split=split
            ).rename_columns({"prompt":"raw_prompt"})
    elif dataset_name == "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144":
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.rename_columns({"query":"raw_prompt"})
    elif dataset_name == "Dahoas/full-hh-rlhf":
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.rename_columns({"prompt":"raw_prompt"})
    elif dataset_name == "RLHFlow/HH-RLHF-Helpful-standard":
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.rename_columns({"chosen":"raw_prompt"})
    else:
        raise NotImplementedError
    data_frac = script_args.data_frac
    dataset = dataset.shuffle(seed=script_args.seed)
    if script_args.frac_len > 0:
        sub_len = script_args.frac_len
        start_idx = sub_len * data_frac
        end_idx = min(sub_len * (data_frac + 1), len(dataset))

        dataset = dataset.select(range(start_idx, end_idx))
        
    return dataset


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
    """ Parse score model arguments from command line."""
    models = []
    
    for model_arg in args.score_model_args:
        models.append(ScoreModelConfig.from_str(model_arg))
    
    return models


def get_scorer(
    score_models: List[ScoreModelConfig], 
    load_in_4bit: Optional[bool] = False, 
    use_flash_attention_2: Optional[bool] = False,
    batch_size: Optional[int] = None,
    tokenizer: Optional[AutoTokenizer] = None,
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