# Training Pipeline

SEMA uses a two-stage training pipeline: **Prefilling Self-Tuning** followed by **Reinforcement Learning with Intent-drift-aware Reward**.

## Stage I: Prefilling Self-Tuning

This stage has two sub-steps: rollout generation and self-tuning.

### Step 1: Prefilling Rollout Generation

Generates multi-turn attack rollouts using vLLM-accelerated inference.

```bash
python sema/prefilling_rollout.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --temperature 1.0 \
    --max_tokens 500 \
    --dataset_name advbench \
    --max_turns 7 \
    --n_generations 10 \
    --batch_size 100 \
    --output_dir files/prefilling_rollouts \
    --output_file "llama8b_prefilling_rollout.json"
```

**What happens:**
1. Loads AdvBench dataset (automatically downloaded on first run)
2. For each harmful request, renders system + query prompts via Jinja2 templates
3. Prefills with `"1. "` to guide numbered multi-turn generation
4. Generates `n_generations` samples per request
5. Parses numbered list format → extracts `adv_prompt` turn steps

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name_or_path` | — | HuggingFace model ID or local path |
| `--n_generations` | 10 | Rollout samples per request |
| `--max_turns` | 7 | Maximum conversation turns |
| `--temperature` | 1.0 | Sampling temperature |
| `--batch_size` | 100 | Requests per vLLM batch |
| `--system_prompt_path` | `sema/prompts/sema_system.jinja` | System prompt template |
| `--query_prompt_path` | `sema/prompts/sema_query.jinja` | Query prompt template |

### Step 2: Self-Tuning via SFT

Trains the attacker model on prefilling rollouts using supervised fine-tuning.

```bash
accelerate launch --num_processes 4 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/self_tuning.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --data_path "files/prefilling_rollouts/llama8b_prefilling_rollout.json" \
    --output_dir files/checkpoints/prefilling_self-tuned-Llama-3.1-8B-Instruct \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --max_length 4096 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | — | Path to prefilling rollout JSON |
| `--max_length` | 4096 | Maximum sequence length |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--num_train_epochs` | 1 | Training epochs |
| `--gradient_checkpointing` | false | Enable gradient checkpointing |

## Stage II: Reinforcement Learning

Fine-tunes the self-tuned attacker using GRPO with intent-drift-aware reward against a training-time victim.

### Step 1: Start vLLM Server

The RL script launches a vLLM server for the self-tuned attacker on a dedicated GPU:

```bash
CUDA_VISIBLE_DEVICES=7 trl vllm-serve \
    --model files/checkpoints/prefilling_self-tuned-Llama-3.1-8B-Instruct
```

### Step 2: Run GRPO Training

```bash
accelerate launch --num_processes 7 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/rl_grpo.py \
    --model_name_or_path files/checkpoints/prefilling_self-tuned-Llama-3.1-8B-Instruct \
    --victim_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir files/checkpoints/SEMA-Llama-3.1-8B-Instruct@Llama8B \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --steps_per_generation 12 \
    --num_generations 28 \
    --learning_rate 1e-5 \
    --score_reward_name ida_reward \
    --score_reward_model_name openai/gpt-4.1-mini-2025-04-14 \
    --use_vllm --vllm_mode server \
    --vllm_server_base_url http://localhost:8000 \
    --reward_log_save_dir files/logs
```

**What happens:**
1. Starts a vLLM victim model on a reserved GPU
2. Attacker generates multi-turn attack plans (completions)
3. Victim processes attacks turn-by-turn → multi-turn conversations
4. Reward computation:
   - **Format Reward**: validates numbered list structure (binary: 0 or 1)
   - **Score Reward**: gpt-4.1-mini evaluates intent alignment, compliance risk, detail level
   - **Final Reward** = Format Reward + Score Reward
5. GRPO updates policy using reward signal

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--victim_model_name_or_path` | — | HuggingFace model for the victim |
| `--score_reward_name` | `ida_reward` | Reward function (`ida_reward`, `no_intent_alignment_reward`, etc.) |
| `--score_reward_model_name` | — | LLM API model for scoring |
| `--num_generations` | 28 | Total generation count per step |
| `--steps_per_generation` | 12 | Gradient steps per generation batch |
| `--gpu_memory_utilization` | 0.4 | vLLM memory fraction for victim |
| `--max_turns` | 7 | Max conversation turns |
| `--reward_log_save_dir` | — | Directory for reward logs |

## Notes

- **GPU allocation**: RL training reserves 1 GPU for the vLLM server (victim inference) and uses the rest for GRPO training. For 8 GPUs: GPU 7 runs the server, GPUs 0–6 run training.
- **API key**: The IDA reward uses `gpt-4.1-mini`. Set your API key via `OPENAI_API_KEY` environment variable or a `.env` file.
- **Experiment tracking**: All scripts log to [Weights & Biases](https://wandb.ai). Set `--report_to none` to disable.
- **DeepSpeed**: All distributed training uses ZeRO Stage 3 (`scripts/deepspeed_configs/deepspeed_zero3.yaml`).
