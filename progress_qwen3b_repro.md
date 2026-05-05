# Progress: SEMA Qwen3B@Llama8B reproduction on 4×A6000

This is the chronological, append-only history. Most recent entries at the bottom.

---

## 2026-05-04 — Session start, environment scoping

**Did:**

- Read repo (README, `docs/architecture.md`, `docs/training.md`, `sema/trainer/trainer.py`, `sema/victims/victims.py`, `sema/rewards/intent_drift_aware_reward.py`, `sema/utils.py`, both qwen3b scripts).
- Inspected git log; flagged commit `e951d39 "normalize rewards"` (Apr 17 2026) which halved the reward components.
- Confirmed hardware: 4× RTX A6000, **46 GiB** visible per GPU (not nominal 48), 1.2 TB free disk, all GPUs idle.
- Confirmed `HF_TOKEN` (37-char) and `WANDB_API_KEY` (92-char) present in environment.

**Smoke test — OpenAI gateway:**

- Read `/workspace/.env_oai`: contains `OPENAI_API_KEY`, `OPENAI_BASE_URL` (Azure-hosted SEMA proxy at `dwip-openai-ehe0b4f3cdctbfbp.westus2-01.azurewebsites.net/v1`), `X_API_KEY` (same value).
- Plain `openai.OpenAI()` returned **403 Invalid API Key**.
- Inspected `sema/utils.py::get_completions` — discovered the gateway requires both bearer auth **and** an `X-API-Key` extra header. Repo's litellm path does this automatically.
- Re-tested via `from sema.utils import get_completions`:
  - `openai/gpt-4.1-mini-2025-04-14` → **404 deployment does not exist**.
  - `openai/gpt-4.1`, `openai/gpt-4.1-mini`, `openai/gpt-4o`, `openai/gpt-4o-mini` → all returned `OK`.
- Decision: substitute `openai/gpt-4.1-mini` for the dated deployment in Stage II.

**Smoke test — packages:**

- `python --version` → 3.12.3 (the SEMA Dockerfile pins 3.11.13 in `requirements.txt`, but the published `allenlao/sema:v0.1` base image ships 3.12 — running with 3.12 in this environment).
- Versions in environment: `trl==0.20.0`, `vllm==0.10.0`, `litellm==1.74.8`, `deepspeed==0.17.4`, `wandb==0.25.1`.
- `peft` was **not installed** despite being imported transitively via `trl.get_peft_config` in `sema/trainer/trainer.py`. Installed `peft==0.19.1`.

**User clarifications received this session:**

1. Use Option 1 (current normalized rewards, post-`e951d39`). Target ~0.925 post-normalization.
2. Reduce `per_device_train_batch_size` to 2 (or 1 if needed).
3. Do **not** modify `scripts/deepspeed_configs/deepspeed_zero3.yaml`. Tune only batch-size / grad-accum / gpu-memory-utilization.
4. Keep `max_prompt_length=2048` (do not shorten).
5. Enable wandb (key already in env).

**Wrote plan:** `/workspace/plan_qwen3b_repro.md`.

**Status:** in-progress. Next: pin `peft` in requirements/Dockerfile, then begin Stage I rollout.
