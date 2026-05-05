# Current attempt — SEMA Qwen3B reproduction on 4×A6000

**Plan:** `plan_qwen3b_repro.md`
**Permanent log:** `progress_qwen3b_repro.md`
**Started:** 2026-05-04

## Where we are

**Setup committed** (36f21ac) — peft pinned, plan/progress docs landed.

**Stage I — done.**
- Rollout: 4 160 samples (416 unique × 10 attempts), 0 % parse failures, mean 6.15 turns. JSON 17 MB at `files/prefilling_rollouts/qwen3b_prefilling_rollout.json`. Took ~4.5 min.
- SFT: 234 steps × ~4.4 s = 17 min 15 s. `train_loss` mean=1.37 (1.43 → 1.20 trajectory), `eval_loss` 1.40 → 1.37. No OOM. wandb run: `https://microsoft-research.wandb.io/tanghao/SEMA/runs/vszan3lm`.
- Checkpoint: `files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct/` (6.16 GB, full tokenizer + config).
- Smoke generation works; format quirks (restart-numbering) expected pre-RL.

**Stage II script drafted**: `scripts/rl_ida_qwen3b@llama8b_4x46gb-gpu.sh`. Not launched.

## Up next

1. **Pending user go**: launch `scripts/rl_ida_qwen3b@llama8b_4x46gb-gpu.sh`.
2. Monitor first ~50 steps for OOM. Memory budget on each trainer A6000: ~18 GB co-located Llama-8B vLLM victim (`gpu_memory_utilization=0.4`) + ~22-26 GB Qwen-3B trainer state (ZeRO-3 sharded params/grads/optim across 3 ranks + activations). Tight — fallback is `per_device_train_batch_size=1` + `grad_accum=56` + `steps_per_gen=112` (still preserves paper math).
3. If steady, monitor reward curve toward target ~0.925 (post-normalization) and average parsed turn-count toward ~4.

## Decision log

- **Reward target:** post-normalization scale, target ~0.925 (Option 1). Will not revert `e951d39`.
- **Stage II hyperparams chosen:** `pdtbs=2`, `grad_accum=28`, `steps_per_gen=56`, `num_processes=3` (preserves paper global=168 / gen=336 / 12 unique prompts/cycle).
- **Score reward model:** `openai/gpt-4.1-mini` (substituting for `gpt-4.1-mini-2025-04-14` which 404s on the gateway).

## Open watchpoints

- A6000 memory will be tight in Stage II; if pdtbs=2 OOMs, drop to pdtbs=1 with `grad_accum=56`, `steps_per_gen=112`.
- vLLM `external_launcher` victim under DeepSpeed ZeRO-3 trainer is the delicate part; expect to debug it.
- Wall-clock for Stage II will be much longer than the paper run; do not interpret slowness as failure.
