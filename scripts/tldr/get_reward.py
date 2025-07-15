from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
import numpy as np
import os
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
import torch.nn as nn
tqdm.pandas()
accelerator = Accelerator()
import json
import warnings

warnings.filterwarnings("ignore", message="copying from a non-meta parameter in the checkpoint to a meta")

# Now, only this specific warning will be suppressed

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    data_path: Optional[str] = field(
        default='output/tldr_ppo_base/4_4_1/iter_t0.1_BoN16_b100/test_data.jsonl',
        metadata={"help": "the location of the dataset name or path"},
    )
    output_path: Optional[str] = field(
        default='output/tldr/cbs/test_data.jsonl',
        metadata={"help": "the location of the dataset name or path"},
    )
    reward_name_or_path: Optional[str] = field(
        default="vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr",
        metadata={"help": "the name of the gold reward model"},
    )
    revision: Optional[str] = field(
        default='reward__44413__1706651113',
    )
    
    

class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        def base_to_name(base):
            if "6.9b" in base:
                return "vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr"
            elif "1b" in base:
                return "vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr"
            elif "2.8b" in base:
                return "vwxyzjn/EleutherAI_pythia-2.8b-deduped__sft__tldr"
            else:
                raise NotImplementedError()
            
        self.lm_backbone = AutoModel.from_pretrained(
            base_to_name(config.base_model),
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        # reward = reward / self.config.normalization_constant
        return reward

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0   


def build_dataset(tokenizer, train_path):

    def tokenize(sample):
        sequences = {
                     "query_reference_response_token": []
                     }
        prompt = sample["prompt"]
        all_response = sample["all_response"]
        # Initialize lists to store tokenized sequences for each response
        sequences = {
            "query_response_token": [],  # Will store N sequences of tokens
        }

        for response in all_response:
            # Combine prompt and response
            query_response = prompt.strip() + response

            tokens = tokenizer.encode(
                query_response,
                padding="max_length",
                max_length=565,
                truncation=True
            )

            sequences["query_response_token"].append(tokens)
        
        return sequences

    dataset = load_dataset("json",data_files=train_path, split='train')
    ds = dataset.map(tokenize, num_proc=1)
    ds = ds.with_format("torch", columns=["query_response_token"])
    
    return ds



def tokenize(tokenizer, prompt, resp):
    return tokenizer.encode(prompt, add_special_tokens=True), tokenizer.encode(resp, add_special_tokens=True)
    
def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values    
     

def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )

def save_annotation(eval_data: list, output_path: str):
        
    results = []
    wins = ties = total = 0

    for item in eval_data:
        prompt = item["prompt"]
        eval_score = item["score"] 
        ref_score = item["ref_score"]

        
        if eval_score > ref_score:
            wins += 1
        elif eval_score == ref_score:
            ties += 1
        total += 1
            
        result = {
            "prompt": prompt,
            "output_1": item["output"],
            "reference": item["reference"],
            "eval_score_1": eval_score,
            "eval_score_2": ref_score,
            "preference": 1 if eval_score > ref_score else (0.5 if eval_score == ref_score else 2)
        }
        results.append(result)

    metrics = {
        "winrate": wins/total,
        "tierate": ties/total, 
        "total": total,
        "mean": np.mean([item["score"] for item in eval_data]),
    }

    output = {
        "metrics": metrics,
        "results": results
    }
    print("Saving to", output_path)

    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    ds_dir = script_args.data_path

    rm_name = script_args.reward_name_or_path
    rm_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_name_or_path, revision=script_args.revision)

    reward_model = ScalarModel.from_pretrained(rm_name, 
                                        revision=script_args.revision, 
                                        trust_remote_code=True,
                                        device_map={"": accelerator.process_index},
                                        torch_dtype=torch.bfloat16)

    disable_dropout(reward_model)
    
    device = accelerator.device
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if ds_dir.endswith(".json"):
        dataset = load_dataset("json", data_files=ds_dir, split="train")
    else:
        dataset = load_dataset(ds_dir, split="train")

    local_rank = Accelerator().local_process_index
    data_size = len(dataset["prompt"])
    share = int(data_size / world_size) + 1
    ds = dataset.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(dataset))))

    data_to_save = []
    with torch.no_grad():
        for data in tqdm(ds):
            prompt = data["prompt"]
            response = data["output"]
            input_ids = rm_tokenizer.encode(prompt.strip() + data["reference"], padding="max_length", max_length=565, truncation=True, return_tensors="pt")
            _, ref_score, _ = get_reward(reward_model, input_ids.to(device), rm_tokenizer, context_length=0)
            if isinstance(response, list):
                N = len(response)
                rewards = []
                for i in range(N):
                    input_ids = rm_tokenizer.encode(prompt.strip() + response[i], padding="max_length", max_length=565, truncation=True, return_tensors="pt")
                    _, score, _ = get_reward(reward_model, input_ids.to(device), rm_tokenizer, context_length=0)
                    rewards.append(score[0].cpu().tolist())
                k = np.argmax(rewards)
                data_to_save.append(
                    {
                        "prompt": prompt,
                        "output": response[k],
                        "score": rewards[k],
                        "reference": data["reference"],
                        "ref_score": ref_score[0].cpu().tolist(),
                    }
                )
            elif isinstance(response, str):
                input_ids = rm_tokenizer.encode(prompt.strip() + response, padding="max_length", max_length=565, truncation=True, return_tensors="pt")
                _, score, _ = get_reward(reward_model, input_ids.to(device), rm_tokenizer, context_length=0)
                data_to_save.append(
                    {
                        "prompt": prompt,
                        "output": response,
                        "score": score[0].cpu().tolist(),
                        "reference": data["reference"],
                        "ref_score": ref_score[0].cpu().tolist(),
                    }
                )
            

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    all_process_list = [{}] * world_size

    data_to_send = {
        "data": [[data_to_save[i]] for i in range(len(data_to_save))],
    }

    import torch.distributed as dist

    dist.all_gather_object(all_process_list, data_to_send)
    gathered_data = []

    for i in range(world_size):
        tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
        gathered_data.extend(tmp_data)

    if local_rank == 0:
        
        output_dir = os.path.dirname(script_args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(script_args.output_path, "w", encoding="utf8") as f:

            json.dump(gathered_data, f, ensure_ascii=False, indent=4)       
        
        save_annotation(gathered_data, os.path.join(output_dir, "annotation.json"))
        