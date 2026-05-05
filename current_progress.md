# Current attempt — SEMA Qwen3B reproduction on 4×A6000

**Plan:** `plan_qwen3b_repro.md`
**Permanent log:** `progress_qwen3b_repro.md`
**Started:** 2026-05-04

## Where we are

**Step 1 — Pin `peft` dependency** (in progress)
Status: `peft==0.19.1` installed in current Python. Still need to mirror the change in `requirements.txt` and `Dockerfile` per CLAUDE.md "environment tracking" rule.

## Up next

1. Edit `requirements.txt` → add `peft==0.19.1`.
2. Edit `Dockerfile` → add a `pip install peft==0.19.1` line so a rebuild of `allenlao/sema:v0.1`-based images bakes it in.
3. Commit.
4. Step 2 — Stage I prefilling rollout: launch `scripts/prefill_selftuning_qwen3b_4x48gb-gpu.sh` (just the rollout half — comment-out or skip the SFT stanza for now so we can verify the rollout output before burning GPU on SFT).
   - Expected output: `files/prefilling_rollouts/qwen3b_prefilling_rollout.json` containing 400 train requests × 10 generations = 4000 samples.
   - Verification: load JSON, check `len(adv_prompt) ≥ 1` ratio and average turn count.

## Decision log

- **Reward target:** post-normalization scale, target ~0.925 (Option 1). Will not revert `e951d39`.
- **Stage II hyperparams chosen:** `pdtbs=2`, `grad_accum=28`, `steps_per_gen=56`, `num_processes=3` (preserves paper global=168 / gen=336 / 12 unique prompts/cycle).
- **Score reward model:** `openai/gpt-4.1-mini` (substituting for `gpt-4.1-mini-2025-04-14` which 404s on the gateway).

## Open watchpoints

- A6000 memory will be tight in Stage II; if pdtbs=2 OOMs, drop to pdtbs=1 with `grad_accum=56`, `steps_per_gen=112`.
- vLLM `external_launcher` victim under DeepSpeed ZeRO-3 trainer is the delicate part; expect to debug it.
- Wall-clock for Stage II will be much longer than the paper run; do not interpret slowness as failure.
