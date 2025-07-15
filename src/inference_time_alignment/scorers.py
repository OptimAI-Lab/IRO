from dataclasses import dataclass, asdict
from typing import Text, List, Dict, Optional
from abc import ABC, abstractclassmethod

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from src.inference_time_alignment.utils import (
    SFTDataMapFunc, 
    SFTDataCollatorWithPadding,
    get_batch_logps,
    prepare_input,
    first_true_indices
)
import torch.nn.functional as F

@dataclass
class ScorerInput:
    response: List[str]
    eos: List[bool]


@dataclass
class BaseScorer(ABC):
    
    @abstractclassmethod
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class ImplicitValueScorer(BaseScorer):
    model: PreTrainedModel
    ref_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: Optional[bool] = False
    model_prompt_template: Optional[str] = "{raw_prompt}"
    ref_model_prompt_template: Optional[str] = "{raw_prompt}"
    raw_prompt: Optional[str] = None

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self

    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        policy_all_logps = self.forward(
            self.model, 
            self.model_prompt_template, 
            input
        )
        ref_all_logps = self.forward(
            self.ref_model, 
            self.ref_model_prompt_template, 
            input
        )
        return policy_all_logps - ref_all_logps

    @torch.no_grad()
    def forward(
        self, 
        model: PreTrainedModel, 
        prompt_template: Text, 
        input: ScorerInput | Dict
    ) -> torch.Tensor:
        input = asdict(input)
        prompt = prompt_template.format(raw_prompt=self.raw_prompt)
        input["prompt"] = [prompt] * len(input["response"])

        tokens = SFTDataMapFunc(tokenizer=self.tokenizer, 
                                add_special_tokens=self.add_special_tokens)(input)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(input["response"]))])
        batch = prepare_input(batch)

        all_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.to(torch.float32)

        return get_batch_logps(all_logits, batch["labels"])


    
@dataclass
class WeightedMultiValueScorer(BaseScorer):
    models: List[PreTrainedModel]
    betas: List[float]
    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: Optional[bool] = False
    model_prompt_template: Optional[str] = "{raw_prompt}"
    batch_size: Optional[int] = None
    beta: Optional[float] = 1.0

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self

    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        policy_all_logps = self.forward(
            self.models, 
            self.model_prompt_template, 
            input
        )
        return policy_all_logps

    @torch.no_grad()
    def forward(
        self, 
        models: List[PreTrainedModel],
        prompt_template: Text, 
        input: ScorerInput | Dict,
    ) -> torch.Tensor:
        input = asdict(input)
        prompt = prompt_template.format(raw_prompt=self.raw_prompt)
        input["prompt"] = [prompt] * len(input["response"])

        tokens = SFTDataMapFunc(tokenizer=self.tokenizer, 
                                add_special_tokens=self.add_special_tokens)(input)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(input["response"]))])
        batch = prepare_input(batch)
        
        def get_model_output(model, batch):
            if self.batch_size is not None:
                all_logits = []
                for i in range(0, len(batch["input_ids"]), self.batch_size):
                    select_indices = torch.arange(i, min(i+self.batch_size, len(batch["input_ids"])))
                    output = model(
                            input_ids=batch["input_ids"][select_indices],
                            attention_mask=batch["attention_mask"][select_indices],return_dict=True,
                            output_hidden_states=True,
                            use_cache=False)
                    sequence_lengths = first_true_indices(batch["input_ids"][select_indices] == self.tokenizer.pad_token_id) - 1
                    reward_logits = model.score(output.hidden_states[-1]).to(sequence_lengths.device)

                    out = reward_logits[
                        torch.arange(reward_logits.size(0), device=reward_logits.device),
                        sequence_lengths,
                    ].squeeze(-1).to(torch.float32)
                    all_logits.append(out)
                    
                out = torch.cat(all_logits)
            else:
                all_output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False
                )
                sequence_lengths = first_true_indices(batch["input_ids"] == self.tokenizer.pad_token_id) - 1
                reward_logits = model.score(all_output.hidden_states[-1]).to(sequence_lengths.device)
                out = reward_logits[
                    torch.arange(reward_logits.size(0), device=reward_logits.device),
                    sequence_lengths,
                ].squeeze(-1).to(torch.float32)
            return out
        
        combined_output = torch.zeros(len(batch["input_ids"]), dtype=torch.float32, device=batch["input_ids"].device)
        for i, (model, beta) in enumerate(zip(models, self.betas)):
            model_output = get_model_output(model, batch)  # Shape: N,1
            scaled_output = model_output / beta
            
            # Add softmax to output for all models except the last one
            if i == 0:
                combined_output = scaled_output
            else:
                # combined_output = scaled_output + F.softmax(combined_output, dim=0)
                combined_output = scaled_output + F.log_softmax(combined_output, dim=0)
                
        
        return combined_output
    
@dataclass
class WeightedMultiValueScorerTLDR(BaseScorer):
    models: List[PreTrainedModel]
    betas: List[float]
    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: Optional[bool] = False
    model_prompt_template: Optional[str] = "{raw_prompt}"
    batch_size: Optional[int] = None
    beta: Optional[float] = 1.0

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self

    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        policy_all_logps = self.forward(
            self.models, 
            self.model_prompt_template, 
            input
        )
        return policy_all_logps

    @torch.no_grad()
    def forward(
        self, 
        models: List[PreTrainedModel],
        prompt_template: Text, 
        input: ScorerInput | Dict,
    ) -> torch.Tensor:
        input = asdict(input)
        prompt = prompt_template.format(raw_prompt=self.raw_prompt)
        input["prompt"] = [prompt] * len(input["response"])

        tokens = SFTDataMapFunc(tokenizer=self.tokenizer, 
                                add_special_tokens=self.add_special_tokens)(input)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(input["response"]))])
        batch = prepare_input(batch)
        
        def get_model_output(model, batch):
            if self.batch_size is not None:
                all_logits = []
                for i in range(0, len(batch["input_ids"]), self.batch_size):
                    select_indices = torch.arange(i, min(i+self.batch_size, len(batch["input_ids"])))
                    reward_logits = model(
                            input_ids=batch["input_ids"][select_indices],
                            attention_mask=batch["attention_mask"][select_indices],return_dict=True,
                            output_hidden_states=True,
                            use_cache=False)
                    sequence_lengths = first_true_indices(batch["input_ids"][select_indices] == self.tokenizer.pad_token_id) - 1
                    # print(output.shape)
                    # reward_logits = model.score(output.hidden_states[-1]).to(sequence_lengths.device)

                    out = reward_logits[
                        torch.arange(reward_logits.size(0), device=reward_logits.device),
                        sequence_lengths,
                    ].squeeze(-1).to(torch.float32)
                    all_logits.append(out)
                    
                out = torch.cat(all_logits)
            else:
                reward_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False
                )
                sequence_lengths = first_true_indices(batch["input_ids"] == self.tokenizer.pad_token_id) - 1
                # reward_logits = model.score(all_output.hidden_states[-1]).to(sequence_lengths.device)
                out = reward_logits[
                    torch.arange(reward_logits.size(0), device=reward_logits.device),
                    sequence_lengths,
                ].squeeze(-1).to(torch.float32)
            return out
        
        combined_output = torch.zeros(len(batch["input_ids"]), dtype=torch.float32, device=batch["input_ids"].device)
        for i, (model, beta) in enumerate(zip(models, self.betas)):
            model_output = get_model_output(model, batch)  # Shape: N,1
            scaled_output = model_output / beta
            
            # Add softmax to output for all models except the last one
            if i == 0:
                combined_output = scaled_output
            else:
                combined_output = scaled_output + F.softmax(combined_output, dim=0)
        
        return combined_output

#-----------------------------------------------------------------------------#
#--------------------------------- Unit Test ---------------------------------#
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta",
        # torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/mistral-7b-sft-beta",
        # torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    prompt_template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user",   "content": "{raw_prompt}"},
        ],
        tokenize=False, 
        add_generation_prompt=True
    )
    scorer = ImplicitValueScorer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        model_prompt_template=prompt_template,
        ref_model_prompt_template=prompt_template,
    )

    scorer.set_raw_prompt("This is the worst sequel")

    response = [
        " in the franchise's history.",
        " on record. Despite high expectations set by its predecessor, the plot fell flat, and the characters lacked the depth and charm that made ", 
    ]
    eos = [True, False]

    for padding_side in ["right", "left"]:
        scorer.tokenizer.padding_side = padding_side

        score = scorer(ScorerInput(response=response, eos=eos))

        for r, e, s in zip(response, eos, score):
            score_ = scorer(ScorerInput(response=[r], eos=[e]))
            assert torch.allclose(s, score_)

    print("pass")

