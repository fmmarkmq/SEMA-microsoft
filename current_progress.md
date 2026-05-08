# Current attempt — SEMA Qwen3B reproduction on 4×A6000

**Plan:** `plan_qwen3b_repro.md`
**Permanent log:** `progress_qwen3b_repro.md`
**Started:** 2026-05-04

## Where we are

**Stage I — done.** Checkpoint at `files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct/`.

**Stage II — halted on 2026-05-08 ~20:30 UTC after 24+ hours of training.**
- Plateaued at reward 0.65–0.75 (best 0.79 single-step, rolling mean ~0.69).
- IDA judge gateway became unreachable due to socket leak (~28 800 leaked TCP sockets across 3 trainer ranks → ephemeral port exhaustion).
- Policy collapsed to score=0 / grad=0 in the last ~7 steps.
- Stopped via SIGINT. **No checkpoint saved** (`--save_strategy no` in the published recipe).
- All 4 GPUs idle. wandb GRPO run: `https://microsoft-research.wandb.io/tanghao/SEMA/runs/4hyrljvh`

## Up next (pending user direction)

1. **Decide whether to retry Stage II.** Open levers:
   - Add `--save_strategy steps --save_steps 25 --save_total_limit 3` to capture intermediate checkpoints.
   - Restart every ~12-18 h to flush socket leak before exhaustion.
   - Try the `openai/gpt-4.1-mini-2025-04-14` dated deployment in case the gateway has added it (paper-exact judge).
2. **Or accept this result** — paper-faithful repro on 4×A6000 plateaus at ~0.69 rolling mean, ~0.79 peak. Document and move on.

## Open watchpoints

- 28 137 TCP sockets still in kernel `orphan` state at last check. Will time out over the next several minutes; should not block a new run if user wants to retry.
- The 7-turn vs 4-turn divergence from paper claim is unresolved. Could be judge calibration, or could indicate the IDA reward shape doesn't pressure for compactness without the explicit length penalty noted in the paper text.

## Decision log

- **Reward target:** post-normalization scale, target ~0.925 (Option 1). Will not revert `e951d39`.
- **Stage II hyperparams chosen:** `pdtbs=2`, `grad_accum=28`, `steps_per_gen=56`, `num_processes=3` (preserves paper global=168 / gen=336 / 12 unique prompts/cycle).
- **Score reward model:** `openai/gpt-4.1-mini` (substituting for `gpt-4.1-mini-2025-04-14` which 404s on the gateway).

## Open watchpoints

- A6000 memory will be tight in Stage II; if pdtbs=2 OOMs, drop to pdtbs=1 with `grad_accum=56`, `steps_per_gen=112`.
- vLLM `external_launcher` victim under DeepSpeed ZeRO-3 trainer is the delicate part; expect to debug it.
- Wall-clock for Stage II will be much longer than the paper run; do not interpret slowness as failure.
