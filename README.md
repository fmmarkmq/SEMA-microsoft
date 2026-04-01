# SEMA

**S**imple yet **E**ffective Learning for **M**ulti-Turn Jailbreak **A**ttacks

[![Paper](https://img.shields.io/badge/arXiv-2602.06854-b31b1b.svg)](https://arxiv.org/abs/2602.06854)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Venue](https://img.shields.io/badge/ICLR-2026-green.svg)](https://arxiv.org/abs/2602.06854)

SEMA is a framework for training open-loop, response-agnostic multi-turn jailbreak attackers via **Prefilling Self-Tuning** and **Reinforcement Learning with Intent-drift-aware Reward**. Unlike closed-loop methods, SEMA eliminates the need for real-time victim model feedback during attack generation.

## Installation

```bash
conda create -n sema python=3.11.13 -y
conda activate sema
pip install -r requirements.txt
```

## Quick Start

**Stage I** — Prefilling Self-Tuning (generates rollouts, then fine-tunes the attacker):

```bash
bash scripts/prefill_selftuning_llama8b_4x80gb-gpu.sh
```

**Stage II** — RL with Intent-drift-aware Reward (trains against a victim model):

```bash
bash scripts/rl_ida_llama8b@llama8b_8x80gb-gpu.sh
```

The AdvBench dataset is downloaded automatically on first run. Outputs are saved to `files/`.

See [examples/](examples/) for all available training configurations and [docs/](docs/) for detailed documentation.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Project structure, modules, and data flow |
| [Training Pipeline](docs/training.md) | Detailed Stage I & II training guide |
| [Reward System](docs/rewards.md) | Intent-drift-aware reward and ablations |
| [Configuration](docs/configuration.md) | Hyperparameters and hardware requirements |

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

MIT License — see [LICENSE](LICENSE) for details.

