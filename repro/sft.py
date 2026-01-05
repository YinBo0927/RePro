from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

from .utils import read_jsonl

def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> List[int]:
    # Prefer tokenizer chat_template when available (Transformers >= 4.38).
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors=None,
        )
    # Fallback: simple formatting
    if add_generation_prompt:
        text = f"User: {messages[-1]['content']}\nAssistant:"
    else:
        # Expect user+assistant
        if len(messages) == 1:
            text = f"User: {messages[0]['content']}\nAssistant:"
        else:
            text = f"User: {messages[0]['content']}\nAssistant: {messages[1]['content']}"
    return tokenizer(text, add_special_tokens=True)["input_ids"]

def build_inputs_and_labels(tokenizer, x: str, y: str, ctx_len: int) -> Dict[str, Any]:
    # Prompt-only ids (includes assistant header)
    prompt_ids = _apply_chat_template(tokenizer, [{"role": "user", "content": x}], add_generation_prompt=True)
    # Full conversation ids
    full_ids = _apply_chat_template(tokenizer, [{"role": "user", "content": x}, {"role": "assistant", "content": y}], add_generation_prompt=False)

    # Ensure prompt is a prefix of full; if not, fall back to concatenation.
    if len(full_ids) < len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
        text = tokenizer.decode(prompt_ids, skip_special_tokens=False) + y
        full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        prompt_ids = tokenizer(tokenizer.decode(prompt_ids, skip_special_tokens=False), add_special_tokens=True)["input_ids"]

    L = len(full_ids)
    P = len(prompt_ids)

    # Truncate to ctx_len by keeping the tail, adjusting the prompt mask accordingly.
    if L > ctx_len:
        s = L - ctx_len
        full_ids = full_ids[s:]
        # prompt tokens kept are those with original indices in [s, P)
        prompt_keep = max(0, P - s)
    else:
        prompt_keep = P

    labels = [-100] * min(prompt_keep, len(full_ids)) + full_ids[min(prompt_keep, len(full_ids)):]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }

class JsonlSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, ctx_len: int):
        self.rows = read_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        x = r["x_train"]
        y = r["y"]
        out = build_inputs_and_labels(self.tokenizer, x, y, self.ctx_len)
        out["z"] = int(r.get("z", -1))
        out["id"] = r.get("id", str(idx))
        return out

def train_lora_sft(
    base_model: str,
    train_jsonl: str,
    val_jsonl: Optional[str],
    out_dir: str,
    ctx_len: int = 768,
    max_steps: int = 500,
    lr: float = 2e-4,
    warmup_ratio: float = 0.03,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bf16: bool = True,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    logging_steps: int = 10,
    save_steps: int = 100,
    seed: int = 123,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=None,  # let PEFT pick common modules; override if needed per model family
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = JsonlSFTDataset(train_jsonl, tokenizer, ctx_len=ctx_len)
    eval_ds = JsonlSFTDataset(val_jsonl, tokenizer, ctx_len=ctx_len) if val_jsonl else None

    args = TrainingArguments(
        output_dir=out_dir,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=save_steps if eval_ds is not None else None,
        save_total_limit=2,
        report_to=[],
        seed=seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
