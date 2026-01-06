# Refinement Provenance Inference: Detecting LLM-Refined Training Prompts from Model Behavior

[![arXiv](https://img.shields.io/badge/arXiv-2601.01966-b31b1b.svg)](https://arxiv.org/abs/2601.01966)


**Authors:**  
Bo Yin\*, Qi Li\*, Runpeng Yu, Xinchao Wang†

National University of Singapore
\* Equal contribution.  
† Corresponding author: xinchao@nus.edu.sg

> [!IMPORTANT]
> **We will release the weights (both attacker and fine-tuned model) on huggingface in the future and add commercial LLM refiner.**

## Overview

![alt text](assets/overview.png)

## Quick Start
### 0. Install

```bash
pip install -r requirements.txt
```

### 1. Prepare raw benchmark instances

```bash
python scripts/prepare_raw.py --dataset gsm8k --out data/gsm8k_raw.jsonl
python scripts/prepare_raw.py --dataset humaneval --out data/humaneval_raw.jsonl
```

Each JSONL row contains:
- `id`: instance id (problem / function)
- `x_raw`: raw prompt
- `y`: reference output used for teacher forcing

### 2. Create refined prompts
```bash
python scripts/refine_prompts.py   --dataset gsm8k   --in data/gsm8k_raw.jsonl   --out data/gsm8k_refined.jsonl   --refiner_model <hf-refiner-model-id>   --task gsm8k
```

### 3. Build victim/shadow pools and mixtures (instance-disjoint)

```bash
python scripts/build_mixtures.py   --raw data/gsm8k_raw.jsonl   --ref data/gsm8k_refined.jsonl   --out_dir data/gsm8k_mix   --rho 0.5   --seed 42
```

Outputs:
- `shadow_train.jsonl`, `shadow_val.jsonl`, `shadow_test.jsonl`
- `victim_train.jsonl`, `victim_val.jsonl`, `victim_test.jsonl`

Each row includes:
- `z` ∈ {0,1}: provenance label (1 refined, 0 raw), sampled once per instance and fixed.
- `x_train`: equals `x_ref` if `z=1` else `x_raw`.
- `x_raw`, `x_ref`, `y` kept for analysis and baselines.

### 4. LoRA SFT for shadow / victim

```bash
python scripts/sft_lora.py   --base_model <hf-base-model-id>   --train_jsonl data/gsm8k_mix/shadow_train.jsonl   --val_jsonl data/gsm8k_mix/shadow_val.jsonl   --out_dir artifacts/shadow_lora   --max_steps 500 --lr 2e-4 --ctx_len 768   --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

Repeat for victim by swapping the train/val files and output directory.

### 5. Extract teacher-forced logit features

Shadow (used for training attacker):
```bash
python scripts/extract_features.py   --base_model <hf-base-model-id>   --lora_dir artifacts/shadow_lora   --data_jsonl data/gsm8k_mix/shadow_train.jsonl   --out_npz artifacts/features_shadow_train.npz   --ctx_len 768
```

Victim (used for evaluation):
```bash
python scripts/extract_features.py   --base_model <hf-base-model-id>   --lora_dir artifacts/victim_lora   --data_jsonl data/gsm8k_mix/victim_test.jsonl   --out_npz artifacts/features_victim_test.npz   --ctx_len 768
```

### 6. Train attacker (supervised contrastive + linear head)

```bash
python scripts/train_attacker.py   --shadow_train_npz artifacts/features_shadow_train.npz   --shadow_val_npz artifacts/features_shadow_val.npz   --out_dir artifacts/attacker   --epochs 30 --batch_size 256 --lr 1e-3 --temperature 0.1
```

This saves:
- `standardize.json` (μ, σ from shadow train)
- `encoder.pt`
- `linear_head.pt`
- `thresholds.json` (e.g., threshold for 1% FPR computed on shadow val)

### 7. Evaluate on victim

```bash
python scripts/eval_victim.py   --features_npz artifacts/features_victim_test.npz   --attacker_dir artifacts/attacker   --out_json artifacts/victim_metrics.json
```

## Result
![alt text](assets/result.png)

## Citation

If you find this work useful, please cite:

```bash
@misc{yin2026refinementprovenanceinferencedetecting,
      title={Refinement Provenance Inference: Detecting LLM-Refined Training Prompts from Model Behavior}, 
      author={Bo Yin and Qi Li and Runpeng Yu and Xinchao Wang},
      year={2026},
      eprint={2601.01966},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.01966}, 
}
```
