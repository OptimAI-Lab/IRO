import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, Literal, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast, gather_object
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    get_scheduler
)
from transformers.utils import is_flash_attn_2_available, is_bitsandbytes_available
import deepspeed
torch.set_printoptions(precision=4, sci_mode=False)
api = HfApi()
INVALID_LOGPROB = 1.0
from datasets import load_from_disk
import multiprocessing

@dataclass
class PpoHParams:
    nminibatches: int = 1
    noptepochs: int = 4 #4
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = False
    kl_coef: float = 0.05


@dataclass
class TaskQueryHParams:
    prompt_len: Optional[int] = None
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[Literal["empty_space", "pad_token"]] = None
    pad_token: Optional[str] = None
    pad_side: Optional[str] = None
    max_response_length: Optional[int] = None
    text_field: Optional[str] = None

@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-4 # 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 64
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 1000000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    nminibatches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 2
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""

    # other args
    base_model: str = "lvwerra/gpt2-imdb"
    """the name of the pretrained model to use"""
    query_dataset: str = "output/train_merge_reward.json"
    """the query dataset"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = False
    """whether to penalize responses that do not contain `truncate_token_id`"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = ""
    """the path to the reward model"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""
    revision: Optional[str] = "main"
    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/ppo_model"
    """Where to save the model"""
    ppo: PpoHParams = field(default_factory=PpoHParams)
    params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            prompt_len=128,
            truncate_field="prompt",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_response_length=53,
            text_field="promt",
            # max_sft_query_response_length=85,
        )
    )
    mse: bool = False
    """Whether to use MSE loss"""
    gae: bool = False
    use_flash_attention_2: bool = True
    """Whether to use Flash Attention 2"""
    select_data: int = -1
    loss_type: str = "mse"

def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps * args.nminibatches
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(args.batch_size, args.nminibatches)
    args.local_mini_batch_size = exact_div(args.local_batch_size, args.nminibatches)
    if args.ppo.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.local_batch_size`
    # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
    args.num_updates = args.total_episodes // args.batch_size
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    if args.run_name:
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}__{args.run_name}"
    else:
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}_classification_value"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened



def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q

def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def forward(model, query_responses, tokenizer):

    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    reward_logits = model.score(output.hidden_states[-1])
    
    return reward_logits


def _ensure_length(toks, l, pad_sequence=None, pad_side=None, truncate_side=None):
    assert pad_side in (None, "left", "right")
    assert truncate_side in (None, "left", "right")
    if len(toks) < l:
        assert pad_sequence is not None
        pad_amt = l - len(toks)
        assert len(pad_sequence) >= pad_amt, f"{len(pad_sequence)} < {pad_amt}"
        if pad_side is None:
            assert len(toks) == l, f"Needed to pad! {len(toks)} < {l}"
            return toks
        elif pad_side == "left":
            return pad_sequence[-pad_amt:] + toks
        else:
            assert pad_side == "right"
            return toks + pad_sequence[:pad_amt]
    if truncate_side is None:
        assert len(toks) == l, f"Needed to truncate! {len(toks)} > {l}"
        return toks
    elif truncate_side == "left":
        return toks[-l:]
    else:
        assert truncate_side == "right"
        return toks[:l]


def _get_query_padding_for_task(hparams: TaskQueryHParams):
    return hparams.pad_token * hparams.prompt_len

# def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
#     if pad_sequence is None:
#         pad_sequence = _get_query_padding_for_task(hparams)
#     if isinstance(query_info, str):
#         query_info = dict(query=query_info)
#     else:
#         # copy to avoid mutating input
#         query_info = dict(**query_info)

#     # format_str =  "{prompt}"
#     # format_str =  f"{hparams.text_field}"
#     format_str = [   
#                 {"role": "user", "content": query_info[hparams.text_field]},
#             ]
#     query_tokens = encoder.encode(encoder.apply_chat_template(
#         format_str,
#         tokenize=False,
#         add_generation_prompt=True,
#     ))
#     truncate_field = hparams.truncate_field or "query"

#     if truncate_field not in query_info:
#         raise ValueError(f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!")
#     while len(query_tokens) > hparams.prompt_len:
#         if not len(query_info[truncate_field]):
#             raise ValueError("Could not truncate enough!")

#         i = -1  # default to just remove one character
#         if hparams.truncate_text:
#             try:
#                 i = query_info[truncate_field].rindex(hparams.truncate_text)
#             except ValueError:
#                 pass
#         query_info[truncate_field] = query_info[truncate_field][:i]
#         query_tokens = encoder.encode(
#             encoder.apply_chat_template(
#             [   
#                 {"role": "user", "content": query_info[truncate_field]},
#             ],
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#         )

#     query_token = _ensure_length(query_tokens, hparams.prompt_len, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
#     query = encoder.decode(query_token, skip_special_tokens=True).lstrip()
#     return dict(
#         query_token=query_token,
#         query=query,
#     )
def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        query_info = dict(**query_info)

    # Check if tokenizer has chat template
    has_chat_template = hasattr(encoder, 'chat_template') and encoder.chat_template is not None

    def encode_text(text):
        if has_chat_template:
            format_str = [{"role": "user", "content": text}]
            return encoder.encode(encoder.apply_chat_template(
                format_str,
                tokenize=False,
                add_generation_prompt=True,
            ))
        else:
            return encoder.encode(text)

    query_tokens = encode_text(query_info[hparams.text_field])
    truncate_field = hparams.truncate_field or "query"

    if truncate_field not in query_info:
        raise ValueError(f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!")
    
    while len(query_tokens) > hparams.prompt_len:
        if not len(query_info[truncate_field]):
            raise ValueError("Could not truncate enough!")

        i = -1  # default to just remove one character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_tokens = encode_text(query_info[truncate_field])

    query_token = _ensure_length(query_tokens, hparams.prompt_len, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
    query = encoder.decode(query_token, skip_special_tokens=True).lstrip()
    return dict(
        query_token=query_token,
        query=query,
    )

def get_final_true_mask(mask):
    """
    Args:
        mask: bool tensor of shape [batch_size, seq_len] where True indicates valid positions
    Returns:
        final_mask: bool tensor of same shape with only final True positions marked as True
    """
    # Create a new tensor of zeros with same shape and device as input
    final_mask = torch.zeros_like(mask, dtype=torch.bool)
    
    # Find last True position for each sequence using flipped argmax
    last_positions = mask.shape[1] - 1 - torch.fliplr(mask.long()).argmax(dim=1)
    
    # Set the final True position to 1 for each sequence
    batch_indices = torch.arange(mask.shape[0], device=mask.device)
    final_mask[batch_indices, last_positions] = True
    
    return final_mask


if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    # IF the model is GPT2, hide this row
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # tokenizer.pad_token = tokenizer.eos_token
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id
    
        # post init
    if args.params.padding == "empty_space":
        args.params.pad_token = tokenizer.encode(" ")
    else:
        args.params.pad_token = [tokenizer.pad_token_id]
        

    def build_dataset(tokenizer, train_path):

        def tokenize(sample):

            reference_response = sample["output"] + tokenizer.eos_token
            y = {
            **process_query(sample, encoder=tokenizer, hparams=args.params),
            "reference_response": reference_response,
            "reference_response_token": tokenizer.encode(
                reference_response,
                padding="max_length",
                max_length=args.params.max_response_length,
                truncation=True,
            ),
            "reference_response_token_len": len(tokenizer.encode(reference_response)),
            }
            
            y["score"] = sample["score"]
            
            return y

        # ds = load_from_disk(train_path)
        if train_path.endswith(".json"):
            ds = load_dataset("json", data_files=train_path, split='train')
        else:
            ds = load_dataset(train_path, split='train')
        if args.select_data > 0:
            ds = ds.select(range(args.select_data))
        ds = ds.map(tokenize, load_from_cache_file=False, num_proc=1)

        train_data = ds.with_format("torch", columns=["query_token", "reference_response_token", "score"])
        
        eval_data = ds.select(range(10))
        return train_data, eval_data
    # load dataset
    dataset, eval_data = build_dataset(tokenizer, args.query_dataset)
    # dataset = dataset.select(range(128))
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)

    args.num_updates = len(dataset) // args.batch_size

    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
        
        
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path, 
        trust_remote_code=True, 
        num_labels=1,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 and is_flash_attn_2_available() else None,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    # embeddings = model.get_input_embeddings()
    # if args.deepspeed:
    #     with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
    #         if len(tokenizer) > embeddings.weight.shape[0]:
    #             model.resize_token_embeddings(len(tokenizer))
    # else:
    #     if len(tokenizer) > embeddings.weight.shape[0]:
    #         model.resize_token_embeddings(len(tokenizer))
            
    disable_dropout(model)

    # model.config.pad_token_id = tokenizer.pad_token_id
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )
    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    torch.manual_seed(local_seed)  # reset the local seed again

    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size

        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        accelerator.print(f"{eval_ds_config=}")
 
    accelerator.print("===training critc===")
    start_time = time.time()

    vf_loss_stats = torch.zeros((args.gradient_accumulation_steps,), device=device)
    vf_clipfrac_stats = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{args.num_train_epochs}"):
            update += 1
            global_step += args.micro_batch_size
            queries = data["query_token"].to(device)
            responses = data["reference_response_token"].to(device)
            query_responses = torch.cat((queries, responses), dim=1)
            context_length = queries.shape[1]
    
            postprocessed_responses = []
            values = []
            scores = []
            sequence_lengths = []
            with torch.no_grad():
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    ### add query_responses in datasert
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    
                    response = query_response[:, context_length:]

                    sequence_length = first_true_indices(response == tokenizer.pad_token_id) - 1
                    if not args.mse:
                        full_value, _, _ = get_reward(
                            accelerator.unwrap_model(model), query_response, tokenizer, context_length
                        )
                        value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    else:
                        value = torch.zeros_like(response, dtype=torch.float, device=response.device)
                    postprocessed_responses.append(response)
                    values.append(value)
                    sequence_lengths.append(sequence_length)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            values = torch.cat(values, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.zeros_like(sequence_lengths, dtype=torch.float)
            # del (full_value, value)
            del value 
            torch.cuda.empty_cache()

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            sequence_lengths_p1 = sequence_lengths + 1
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))

            values = torch.masked_fill(values, padding_mask_p1, 0)
            
            new_mask = get_final_true_mask(~padding_mask_p1)
            # 4. compute rewards
            rewards = torch.zeros_like(responses, dtype=torch.float, device=response.device)

            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
            rewards[[actual_start, actual_end]] += data["score"]

            # 6. compute advantages and returns
            lastgaelam = 0
            advantages_reversed = []
            gen_length = responses.shape[1]
            
            if args.mse:
                args.ppo.lam = 1.0
            
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + args.ppo.gamma * nextvalues - values[:, t]
                lastgaelam = delta + args.ppo.gamma * args.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
        
            
            return_mean, return_var = returns.mean(), returns.var()
            value_mean, value_var = values.mean(), values.var()
            # accelerator.print("rewards====", rewards[0])
            # accelerator.print("values====", values[0])
            torch.cuda.empty_cache()

            with accelerator.accumulate(model):

                vpred_temp = forward(model, query_responses, tokenizer)

                vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                vpred = torch.masked_fill(vpred, padding_mask_p1, 0)

                if  args.loss_type == "mse":
                    vf_losses1 = torch.square(vpred - returns)

                    loss = 0.5 * masked_mean(vf_losses1, ~padding_mask_p1)
                    
                    reward_loss = torch.mean(torch.square(vpred[new_mask] - returns[new_mask]))

                elif  args.loss_type == "bce":
                    bce_loss = nn.BCEWithLogitsLoss()
                    vf_losses1 = bce_loss(vpred, returns)
                    loss = 0.5 * masked_mean(vf_losses1, ~padding_mask_p1)
                    reward_loss = torch.mean(torch.square(vpred[new_mask] - returns[new_mask]))

                    # Debugging: Check the True position in new_mask and first True in padding_mask_p1
                    # true_positions_new_mask = torch.nonzero(new_mask, as_tuple=True)
                    # first_true_positions_padding_mask_p1 = first_true_indices(padding_mask_p1)

                    # accelerator.print("True positions in new_mask:", true_positions_new_mask)
                    # accelerator.print("First True positions in padding_mask_p1:", first_true_positions_padding_mask_p1)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                vf_loss_stats[gradient_accumulation_idx] = loss
                reward_losses[gradient_accumulation_idx] = reward_loss
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                writer.add_scalar("train/critic/loss", accelerator.gather(vf_loss_stats).mean().item(), global_step)
                writer.add_scalar("train/critic/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/critic/reward_loss",  accelerator.gather(reward_losses).mean().item(), global_step)
            del (
                vpred_temp, vpred,loss, returns,
                values, responses, query_responses
            )
            # fmt: on
            torch.cuda.empty_cache()
            
        if args.output_dir and args.num_train_epochs > 0 and epoch >= 9:
            output_dir = f"{args.output_dir}_checkpoint_{epoch}"
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            time_tensor = torch.tensor([int(time.time())], device=device)
            time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
            repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__critic"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                if args.push_to_hub:
                    tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
            unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
            accelerator.wait_for_everyone()
            # if accelerator.is_main_process:
            unwrapped.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
                accelerator.print(f"🔥 pushed to https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}")
                
    
    # save model
    if args.output_dir and args.num_train_epochs > 0:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__critic"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
        unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        # if accelerator.is_main_process:
        unwrapped.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=False,
        )
        if args.push_to_hub:
            unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
            accelerator.print(f"🔥 pushed to https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}")