# Current attempt — SEMA Qwen3B reproduction on 4×A6000

**Plan:** `plan_qwen3b_repro.md`
**Permanent log:** `progress_qwen3b_repro.md`
**Started:** 2026-05-04

## Where we are

**Stage I — done** (committed `d3b3ac8`). 22 min wall-clock. Checkpoint at `files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct/`.

**Stage II — running** since 2026-05-06 21:43 UTC. wandb: `https://microsoft-research.wandb.io/tanghao/SEMA/runs/4hyrljvh`.

Active config:
- `scripts/rl_ida_qwen3b@llama8b_4x46gb-gpu.sh` (paper-exact except num_processes=3)
- `scripts/deepspeed_configs/deepspeed_zero3_cpu_offload.yaml` (CPU optimizer offload — approved by user 2026-05-06 after analytical OOM accounting)
- Per-rank GPU usage: 38–39 GiB / 44.4 GiB (~6 GiB headroom)

Reward trajectory (first 3 logged optimizer steps):

| step | reward | grad_norm | mean_len | frac_zero_std |
|---|---|---|---|---|
| 1 | 0.4652 | 1.33 | 178 | 0.00 |
| 4 | 0.5106 | 1.27 | 159 | 0.00 |
| 7 | 0.5317 | 1.29 | 133 | 0.00 |

ETA: ~67 h / 2.8 days. Target reward ≥ 0.90 (post-normalization scale; equivalent to paper's pre-norm 1.80+).

## Up next

1. **Continue monitoring reward curve**. Re-arm Monitor every hour. Watch for:
   - Reward climbing past 0.7 by step 50, past 0.85 by step 150
   - grad_norm staying in ~1–2 range (collapse toward 0 → arXiv:2604.23747 bug; explosion → instability)
   - mean_len compressing toward ~4 turns × ~50 tokens = ~200 tokens (paper-claimed)
2. If anything regresses, snapshot logs and surface to user.
3. After completion, evaluate the saved checkpoint (`files/checkpoints/SEMA-Qwen2.5-3B-Instruct@Llama8B/`) on AdvBench held-out test set per the paper's eval protocol.

## Open watchpoints

- **arXiv:2604.23747 silent grad-accum bug** — early signal good (grad_norm=1.3, reward climbing) but the bug is silent so absence-of-error ≠ proof-of-correctness. The reward trajectory itself is the integrity check.
- **Wall-clock**: 67 hours is a real commitment. Disconnects, OOM at later step (e.g., from longer completions as policy learns), gateway rate-limits could all interrupt.

## Decision log

- **Reward target:** post-normalization scale, target ~0.925 (Option 1). Will not revert `e951d39`.
- **Stage II hyperparams chosen:** `pdtbs=2`, `grad_accum=28`, `steps_per_gen=56`, `num_processes=3` (preserves paper global=168 / gen=336 / 12 unique prompts/cycle).
- **Score reward model:** `openai/gpt-4.1-mini` (substituting for `gpt-4.1-mini-2025-04-14` which 404s on the gateway).

## Open watchpoints

- A6000 memory will be tight in Stage II; if pdtbs=2 OOMs, drop to pdtbs=1 with `grad_accum=56`, `steps_per_gen=112`.
- vLLM `external_launcher` victim under DeepSpeed ZeRO-3 trainer is the delicate part; expect to debug it.
- Wall-clock for Stage II will be much longer than the paper run; do not interpret slowness as failure.
