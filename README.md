# SEMA

**S**imple yet **E**ffective Learning for **M**ulti-Turn Jailbreak **A**ttacks

A framework for training open-loop, response-agnostic multi-turn jailbreak attackers using Prefilling Self-Tuning and Reinforcement Learning with Intent-drift-aware Reward.

**Paper**: https://arxiv.org/abs/2602.06854 (ICLR 2026)

:warning: Please check the first author's [repository](https://github.com/fmmarkmq/SEMA.git) for the latest updates of SEMA.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Existing multi-turn jailbreak methods rely on closed-loop interactions, requiring real-time feedback from the target model. SEMA introduces a novel open-loop approach that:

- **Eliminates** dependency on victim model responses during attack generation
- **Learns** effective multi-turn attack strategies via two-stage training
- **Maintains** intent consistency across conversation turns with intent-drift-aware rewards

### Key Insight

Multi-turn jailbreak attacks can be learned in an open-loop manner by:

1. **Prefilling Self-Tuning**: Bootstrap the attacker with high-quality multi-turn attack patterns
2. **Intent-drift-aware RL**: Optimize attack success while preventing semantic drift from the original malicious intent

## Installation

```bash
conda create -n sema python=3.11.13 -y
conda activate sema
pip install -r requirements.txt
```

## Quick Start

### Stage I: Prefilling Self-Tuning

Train the attacker model to generate multi-turn attack patterns:

```bash
bash scripts/prefill_selftuning_llama8b_4x80gb-gpu.sh
```

**Outputs:**
- Prefilling rollouts → `files/prefilling_rollouts/`
- Self-tuned model → `files/checkpoints/`

Note: The AdvBench dataset will be automatically downloaded to `data/advbench.csv` on first run if not already present.

### Stage II: Reinforcement Learning with Intent-drift-aware Reward

Fine-tune the attacker using RL against a training-time victim:

```bash
bash scripts/rl_ida_llama8b@llama8b_8x80gb-gpu.sh
```

**Outputs:**
- Learned attacker model → `files/checkpoints/`
- Reward logs → `files/logs/`

## Available Scripts

The following scripts are provided in `scripts/`. More scripts for ablation study can be found in `scripts/ablations`.

| Script | Base Attacker | Training-time Victim | GPUs |
|--------|---------------|--------------|------|
| `prefill_selftuning_llama8b_4x80gb-gpu.sh` | Llama-3.1-8B-Instruct | - | 4×H100 |
| `prefill_selftuning_qwen14b_8x80gb-gpu.sh` | Qwen2.5-14B-Instruct | - | 8×H100 |
| `prefill_selftuning_qwen3b_4x48gb-gpu.sh` | Qwen2.5-3B-Instruct | - | 4×A6000 |
| `rl_ida_llama8b@llama8b_8x80gb-gpu.sh` | Llama-3.1-8B-Instruct (after Pref.-SelfT.) | Llama-3.1-8B-Instruct | 8×H100 |
| `rl_ida_qwen14b@llama8_8x80gb-gpu.sh` | Qwen2.5-14B-Instruct  (after Pref.-SelfT.) | Llama-3.1-8B-Instruct | 8×H100 |
| `rl_ida_qwen3b@llama8b_8x80gb-gpu.sh` | Qwen2.5-3B-Instruct (after Pref.-SelfT.) | Llama-3.1-8B-Instruct | 8×H100 |

## Project Structure

```
sema/
├── prefilling_rollout.py    # Prefilling rollout generation
├── rl_grpo.py               # GRPO-based reinforcement learning
├── self_tuning.py           # Self-tuning training
├── utils.py                 # Utility functions
├── prompts/                 # Prompt templates
├── rewards/                 # Reward functions
│   ├── format_reward.py
│   ├── intent_drift_aware_reward.py
│   └── ablations/           # Ablation reward variants
└── victims/                 # Victim model interfaces
```

## Citation

If you use SEMA in your research, please cite:

```bibtex
@inproceedings{sema2026,
      title={SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks}, 
      author={Mingqian Feng, Xiaodong Liu, Weiwei Yang, Jialin Song, Xuekai Zhu, Chenliang Xu, Jianfeng Gao},
      year={2026},
      booktitle={International Conference on Learning Representations (ICLR)},
      eprint={2602.06854},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.06854}, 
}
```


## License

MIT License - see [LICENSE](LICENSE) for details.

