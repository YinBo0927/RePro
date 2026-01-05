from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# We avoid hard-coding any proprietary API details. If you want to use an external
# refiner (e.g., a hosted LLM), implement ExternalRefiner.generate().

def _template(task: str, x_raw: str) -> str:
    """Paraphrased refinement template (non-solution)."""
    if task == "gsm8k":
        return (
            "Rewrite the following math word problem into a clear, well-structured instruction "
            "for instruction tuning. Preserve all quantities/conditions and the required output. "
            "Do NOT provide any solution steps, derivations, formulas, or final answers. "
            "Output only the rewritten prompt.\n\n"
            f"RAW PROBLEM:\n{x_raw}\n"
        )
    elif task == "humaneval":
        return (
            "Rewrite the following code-task prompt into a clearer specification. "
            "Do NOT implement the function; do NOT include code or pseudocode. "
            "If the raw prompt includes a signature/stub, keep it exactly and only revise the surrounding text. "
            "You may add constraints, edge cases, and brief plain-text examples. "
            "Output only the rewritten prompt.\n\n"
            f"RAW TASK:\n{x_raw}\n"
        )
    else:
        raise ValueError(f"Unknown task: {task}")

@dataclass
class HFRefiner:
    model_id: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

    def __post_init__(self):
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()

    @torch.no_grad()
    def generate_one(self, task: str, x_raw: str) -> str:
        prompt = _template(task, x_raw)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Heuristic: remove the prompt prefix
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return text.strip()

    def generate(self, task: str, xs: List[str]) -> List[str]:
        return [self.generate_one(task, x) for x in xs]

class ExternalRefiner:
    def generate(self, task: str, xs: List[str]) -> List[str]:
        raise NotImplementedError("Implement this for your provider.")
