# Plan: Reproduce SEMA Qwen2.5-3B@Llama-3.1-8B on local 4×A6000

**Owner:** Tang Hao &lt;tanghao@microsoft.com&gt;
**Started:** 2026-05-04
**Hardware:** 4× NVIDIA RTX A6000 (46 GiB visible per GPU; 184 GiB aggregate)
**Goal:** Smooth reward climb on the GRPO Stage II run, with average `len(adv_prompt) ≈ 4` turns, on the current `main` branch (post-`normalize rewards` commit).

## Goal & success criteria

| Criterion | Target |
|---|---|
| Stage I — prefilling rollout produces parsable multi-turn `adv_prompt` arrays | ≥ 95 % rollouts parse to ≥ 1 turn |
| Stage I — SFT loss decreases monotonically over 1 epoch | ≥ 30 % drop end-to-start, no NaNs |
| Stage I — sampled completion is a numbered multi-turn plan | manual eyeball + format-reward = 1.0 on 10 samples |
| Stage II — wandb `train/reward` shows a smooth upward trajectory | monotonic-ish rolling mean over 200 steps |
| Stage II — final mean reward | **≥ 0.90** on the post-normalization scale (= ~1.80+ on the paper's pre-normalization scale) |
| Stage II — mean parsed turn count `len(adv_prompt)` over last 100 steps | **≈ 4 turns** |

The 1.85 number the user originally quoted is on the **pre-normalization** scale (commit `e951d39` halved both reward components). User chose Option 1 — run with current normalized code, target ~0.925 post-normalization.

## Hardware adaptation

The published `scripts/rl_ida_qwen3b@llama8b_8x80gb-gpu.sh` assumes 8×80 GiB. We have 4×46 GiB. Adaptation rules (per `feedback_sema_repo` memory):

1. Do not modify `scripts/deepspeed_configs/deepspeed_zero3.yaml`.
2. Do not shorten `max_prompt_length=2048` or `max_completion_length=500`.
3. Preserve paper math: same global batch (168) and same generation batch (336), so `gen_batch / num_generations = 12 unique prompts / cycle` is unchanged.
4. Tune only `--per_device_train_batch_size`, `--gradient_accumulation_steps`, `--steps_per_generation`, and `--gpu_memory_utilization` for memory.

### GPU layout

| GPU | Role |
|---|---|
| 0 | `trl vllm-serve` hosting the SFT'd attacker (GRPO rollouts) |
| 1, 2, 3 | `accelerate launch --num_processes 3` trainer (DeepSpeed ZeRO-3, also co-locates the Llama-3.1-8B victim via vLLM `external_launcher` backend at `gpu_memory_utilization=0.4`) |

### Hyperparameter mapping (Stage II)

| Param | Paper (8×80GB) | This run (4×46GB) | Notes |
|---|---|---|---|
| `num_processes` | 7 | **3** | hardware |
| `per_device_train_batch_size` | 4 | **2** (fallback **1**) | user-specified to avoid OOM |
| `gradient_accumulation_steps` | 6 | **28** (or **56** if pdtbs=1) | preserves global batch = 168 |
| `steps_per_generation` | 12 | **56** (or **112** if pdtbs=1) | preserves gen batch = 336 |
| `num_generations` | 28 | 28 | unchanged |
| `max_prompt_length` | 2048 | 2048 | unchanged |
| `max_completion_length` | 500 | 500 | unchanged |
| `learning_rate` | 1e-5 | 1e-5 | unchanged |
| `gpu_memory_utilization` | 0.4 | 0.4 (tune) | victim KV cache budget |
| `score_reward_model_name` | `openai/gpt-4.1-mini-2025-04-14` | **`openai/gpt-4.1-mini`** | dated deployment 404s on this gateway |

Wall-clock: this is roughly 5× slower per gradient step than the paper run (gen batch unchanged, but only 3 trainer ranks instead of 7, and twice the grad-accum cycles). Plan for the run to take days, not hours.

## Step list

| # | Step | Owner | Verification |
|---|---|---|---|
| 1 | Pin the missing `peft` dependency in `requirements.txt` and the `Dockerfile` | agent | `pip show peft` reports 0.19.1 |
| 2 | Stage I — prefilling rollout via `scripts/prefill_selftuning_qwen3b_4x48gb-gpu.sh` (rollout half only) | agent | `files/prefilling_rollouts/qwen3b_prefilling_rollout.json` exists; ≥ 95 % entries parse to ≥ 1 turn; sample manually inspected |
| 3 | Stage I — self-tuning SFT (second half of same script) on 4×A6000 | agent | `files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct/` saved; sampled completion is a multi-turn numbered plan; W&B run captured |
| 4 | Stage II — write `scripts/rl_ida_qwen3b@llama8b_4x46gb-gpu.sh` per the table above | agent | script committed; lints with `bash -n` |
| 5 | Stage II — launch and monitor for first ~50 steps, tune `per_device_train_batch_size` ∈ {2, 1} and `gpu_memory_utilization` only as needed for OOM | agent | no OOM; trainer makes forward progress (steps emitted); reward log written |
| 6 | Stage II — long run, monitor reward trajectory and turn count | agent | wandb shows smooth climb; success criteria met |

## Risks & open questions

- **A6000 memory ceiling.** With pdtbs=2, expected per-rank usage ≈ 27 GiB (12 GiB sharded Qwen3B state + 2 GiB ref + ~8 GiB activations w/ grad-checkpointing) plus 18 GiB for the co-located Llama-8B vLLM victim → 45 GiB on a 46 GiB card. Very tight. Fallback: pdtbs=1 (with grad_accum=56, steps_per_gen=112).
- **vLLM external_launcher under DeepSpeed ZeRO-3.** vLLM's `external_launcher` backend assumes the trainer ranks are also vLLM workers; this is the published pattern but is delicate. If it breaks, we'll fall back to a separate Ray-served victim (worse memory budget — likely forces pdtbs=1).
- **Score reward model substitution.** Using `openai/gpt-4.1-mini` (not the dated snapshot). Numerically the IDA scoring rubric is robust to small model differences, but won't be bit-identical to the paper.
- **Wall-clock.** ~5× paper. Long. May want to checkpoint at fixed intervals so we can resume.

## Out of scope

- Evaluation on held-out split (only running training; eval can come after we confirm reward dynamics).
- Baselines (SFT/DPO).
- Ablation reward variants.
