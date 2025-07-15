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
    AutoTokenizer,
    pipeline,
)
import torch
from typing import Optional
tqdm.pandas()
accelerator = Accelerator()
import warnings
import json
warnings.filterwarnings("ignore", message="copying from a non-meta parameter in the checkpoint to a meta")

# Now, only this specific warning will be suppressed

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    data_path: Optional[str] = field(
        default='evaluation/AlpacaEval2/AlpacaEval_results/model_generation/tulu-2-7b-temp_0/train_merge.json',
        metadata={"help": "the location of the dataset name or path"},
    )
    output_path: Optional[str] = field(
        default='output/tldr/cbs/test_data.json',
        metadata={"help": "the location of the dataset name or path"},
    )
    reward_name_or_path: Optional[str] = field(
        default="weqweasdas/RM-Mistral-7B",
        metadata={"help": "the name of the reward model"},
    )
    eval_path: Optional[str] = field(
        default="output/tldr/cbs/test_data.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    generator: Optional[str] = None
    output_first: Optional[bool] = field(
        default=False,
    )
     


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    ds_dir = script_args.data_path

    device = accelerator.device
    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1,
    }
    rm_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_name_or_path)

    rm_pipe = pipeline(
        "sentiment-analysis",
        model=script_args.reward_name_or_path,
        #device="auto",
        device=device,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    ds_dir = script_args.data_path
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if ds_dir.endswith(".json"):
        dataset = load_dataset("json", data_files=ds_dir, split="train")
    else:
        dataset = load_dataset(ds_dir, split="train")

    local_rank = Accelerator().local_process_index

    data_size = len(dataset["instruction"])

    share = int(data_size / world_size) + 1
    ds = dataset.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(dataset))))

    data_to_save = []
    with torch.no_grad():
        for data in tqdm(ds):
            prompt = data["instruction"]
            response = data["output"]
            generator = data["generator"] if script_args.generator is None else script_args.generator
            if isinstance(response, list):
                N = len(response)
                rewards = []
                for i in range(N):
                    intput_text = rm_tokenizer.apply_chat_template([{"role": "user", "content": prompt},
                                                                {"role": "assistant", "content": response[i]}],
                                                            tokenize=False, 
                                                            add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")
                    reward = rm_pipe(intput_text, **pipe_kwargs)
                    rewards.append(reward[0][0]["score"])
                k = np.argmax(rewards)
                if script_args.output_first:
                    data_to_save.append(
                        {
                            "instruction": prompt,
                            "output": response[0],
                            "score": rewards[0],
                            "generator": generator
                        }
                    )
                else:
                    data_to_save.append(
                        {
                            "instruction": prompt,
                            "output": response[k],
                            "score": rewards[k],
                            "generator": generator
                        }
                    )
            elif isinstance(response, str):
                intput_text = rm_tokenizer.apply_chat_template([{"role": "user", "content": prompt},
                                                                {"role": "assistant", "content": response}],
                                                            tokenize=False, 
                                                            add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")
                reward = rm_pipe(intput_text, **pipe_kwargs)

                data_to_save.append(
                    {
                        "instruction": prompt,
                        "output": response,
                        # "score": data["score"],
                        "eval_score": reward[0][0]["score"],
                        "generator": generator
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
        # print(np.mean(gathered_data))
        output_dir = os.path.dirname(script_args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(script_args.output_path, "w", encoding="utf8") as f:

            json.dump(gathered_data, f, ensure_ascii=False, indent=4)