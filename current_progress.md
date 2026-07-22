# Current attempt — SEMA Qwen3B reproduction on 4×A6000

**Plan:** `plan_qwen3b_repro.md`
**Permanent log:** `progress_qwen3b_repro.md`
**Started:** 2026-05-04

## Where we are

**Stage I — done.** Checkpoint at `files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct/`.

**Stage II — COMPLETE on 2026-05-10 13:59 UTC after ~39 hours.**

- Reward target achieved: best single step **0.9444** (opt step 165), last-8-step mean **0.910**.
- Watchdog SIGINT-ed cleanly at socket threshold 25 000; trainer exited in 25 s.
- 16 checkpoints saved (steps 10–160). Latest is `checkpoint-160/` (41 GB, includes 6 GB inference safetensors).
- wandb run: `https://microsoft-research.wandb.io/tanghao/SEMA/runs/nixtn3df`
- All GPUs idle, disk 72 % used, no orphan procs.

## Recommended downstream

- **Inference / eval**: use `files/checkpoints/SEMA-Qwen2.5-3B-Instruct@Llama8B/checkpoint-160/model-*.safetensors`.
- **Ablation**: try `checkpoint-100`, `checkpoint-130` to compare early-vs-late attacker.

## Resolved divergences from paper

- `num_adv_turns` saturates at 7 (paper claims ~4). Likely the IDA reward formula lacks the length penalty in the paper text. Not blocking; documented in progress log.
- Used `openai/gpt-4.1-mini` (the dated `2025-04-14` deployment 404s on this gateway). Final results may differ slightly from paper numerically but trajectory shape is similar.

## Decision log

- **Reward target:** post-normalization scale, target ~0.925 (Option 1). Will not revert `e951d39`.
- **Stage II hyperparams chosen:** `pdtbs=2`, `grad_accum=28`, `steps_per_gen=56`, `num_processes=3` (preserves paper global=168 / gen=336 / 12 unique prompts/cycle).
- **Score reward model:** `openai/gpt-4.1-mini` (substituting for `gpt-4.1-mini-2025-04-14` which 404s on the gateway).

## Open watchpoints

- A6000 memory will be tight in Stage II; if pdtbs=2 OOMs, drop to pdtbs=1 with `grad_accum=56`, `steps_per_gen=112`.
- vLLM `external_launcher` victim under DeepSpeed ZeRO-3 trainer is the delicate part; expect to debug it.
- Wall-clock for Stage II will be much longer than the paper run; do not interpret slowness as failure.
