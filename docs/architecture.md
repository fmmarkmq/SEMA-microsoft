# Architecture

## Project Structure

```
sema/
├── prefilling_rollout.py       # Stage I: prefilling rollout generation via vLLM
├── self_tuning.py              # Stage I: SFT on prefilling rollouts (TRL SFTTrainer)
├── rl_grpo.py                  # Stage II: GRPO-based reinforcement learning
├── utils.py                    # Shared utilities (seeding, async completions)
├── prompts/                    # Jinja2 prompt templates
│   ├── sema_system.jinja       # System prompt for multi-turn attack generation
│   ├── sema_query.jinja        # Query prompt template
│   ├── ida_reward_system.jinja # IDA reward scoring rubric
│   └── no_intend_alignment_reward_system.jinja  # Ablation variant
├── rewards/                    # Reward functions for RL training
│   ├── format_reward.py        # Numbered list format checking
│   ├── intent_drift_aware_reward.py  # Main IDA reward (via gpt-4.1-mini)
│   ├── utils.py                # JSON parsing helpers
│   └── ablations/              # Ablation reward variants
│       ├── no_intent_alignment_reward.py
│       └── no_refusal_reward.py
├── victims/                    # Victim model interfaces
│   ├── victims.py              # VLLMVictimWrapper for multi-turn simulation
│   └── utils.py                # RayLLM distributed wrapper
└── baselines/                  # Baseline methods for comparison
    ├── sft.py                  # SFT baseline (filtered by score=100)
    ├── dpo.py                  # DPO baseline
    └── rollout.py              # Ray-distributed rollout with victim execution
```

## Module Overview

### Core Training Modules

| Module | Role |
|--------|------|
| `prefilling_rollout.py` | Generates multi-turn attack rollouts using vLLM. Takes AdvBench harmful requests + prompt templates, outputs numbered multi-turn jailbreak attempts (up to 7 turns). Uses prefilling to bootstrap attack patterns. |
| `self_tuning.py` | SFT stage. Trains attacker on prefilling rollouts using TRL's `SFTTrainer` with DeepSpeed ZeRO-3. |
| `rl_grpo.py` | Main RL loop. Starts a vLLM victim server, then runs GRPO training with format + intent-drift-aware rewards. |
| `utils.py` | `setup_seed()`, `set_deterministic()`, `get_completions()` for async API calls with litellm disk caching. |

### Reward System

| Component | Function |
|-----------|----------|
| `format_reward.py` | `NumberedListFormatReward` — checks if the completion follows `1. ... 2. ... 3. ...` format up to `max_turns`. Returns 1 if valid, 0 otherwise. Also parses turn items for victim consumption. |
| `intent_drift_aware_reward.py` | Main reward. Calls gpt-4.1-mini to score three dimensions: Intent Alignment (0–10), Compliance Risk (0–10), Detail Level (0–10). Final score = Intent Alignment × (Compliance + Detail) / 2. |
| `ablations/no_intent_alignment_reward.py` | Removes intent alignment component (ablation baseline). |
| `ablations/no_refusal_reward.py` | Simple refusal detector matching 19 refusal phrases. |

### Victim Interface

| Component | Function |
|-----------|----------|
| `victims.py` | `VLLMVictimWrapper` — wraps vLLM for multi-turn conversation simulation. `execute()` for single-turn, `execute_batch()` for batch multi-turn with message history tracking. |
| `utils.py` | `RayLLM` — optional Ray-distributed wrapper for multi-GPU victim serving. |

## Data Flow

```
AdvBench (500 harmful behaviors, 80/20 train/test split)
        │
        ▼
┌─────────────────────────────┐
│  Stage I: Prefilling         │
│  prefilling_rollout.py       │
│  → 10 rollouts per request   │
│  → numbered multi-turn format │
└──────────┬──────────────────┘
           │  files/prefilling_rollouts/*.json
           ▼
┌─────────────────────────────┐
│  Stage I: Self-Tuning        │
│  self_tuning.py              │
│  → SFT on rollouts           │
│  → DeepSpeed ZeRO-3          │
└──────────┬──────────────────┘
           │  files/checkpoints/prefilling_self-tuned-*/
           ▼
┌─────────────────────────────┐
│  Stage II: RL (GRPO)         │
│  rl_grpo.py                  │
│  → vLLM victim server        │
│  → format + IDA rewards      │
│  → gpt-4.1-mini scoring      │
└──────────┬──────────────────┘
           │
           ▼
  files/checkpoints/SEMA-*     (final attacker)
  files/logs/reward_logs_*.json (reward tracking)
```

## Output Structure

```
files/
├── prefilling_rollouts/
│   └── <model>_prefilling_rollout.json
│       → List[{request, adv_prompt[], output_text, input_prompt}]
├── checkpoints/
│   ├── prefilling_self-tuned-<Model>/      # Stage I output
│   └── SEMA-<AttackerModel>@<VictimModel>/ # Stage II output
└── logs/
    └── reward_logs_rank*.json
        → List[{step, requests, completions, adv_prompts,
                 format_rewards, score_rewards, final_rewards}]
```
