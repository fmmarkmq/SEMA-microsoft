# Configuration

## Hardware Requirements

| Configuration | GPUs | Use Case |
|---------------|------|----------|
| 4×H100 (80 GB) | 4 | Prefilling self-tuning (Llama-8B) |
| 4×A6000 (48 GB) | 4 | Prefilling self-tuning (Qwen-3B) |
| 8×H100 (80 GB) | 8 | RL training (all model sizes) |
| 4×H100 (80 GB) | 4 | RL training with smaller victim (ablations) |

RL training reserves 1 GPU for the vLLM victim server. The remaining GPUs run GRPO training.

## Key Hyperparameters

### Prefilling Rollout

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_generations` | 10 | Rollout samples per harmful request |
| `max_turns` | 7 | Maximum conversation turns |
| `temperature` | 1.0 | Sampling temperature for diversity |
| `max_tokens` | 500 | Maximum output tokens per rollout |
| `batch_size` | 100 | Requests per vLLM inference batch |

### Self-Tuning (SFT)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_length` | 4096 | Maximum sequence length |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `num_train_epochs` | 1 | Training epochs |
| `warmup_ratio` | 0.03 | Warmup proportion |
| `lr_scheduler_type` | `cosine_with_min_lr` | LR schedule |
| `torch_dtype` | `bfloat16` | Training precision |

### RL (GRPO)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-5 | GRPO learning rate |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 6 | Gradient accumulation |
| `num_generations` | 28 | Generations per GRPO step |
| `steps_per_generation` | 12 | Gradient steps per generation batch |
| `max_prompt_length` | 2048 | Maximum prompt length |
| `max_completion_length` | 500 | Maximum completion length |
| `gpu_memory_utilization` | 0.4 | vLLM memory fraction for victim |

## DeepSpeed Configuration

All distributed training uses DeepSpeed ZeRO Stage 3 via `scripts/deepspeed_configs/deepspeed_zero3.yaml`. Key settings:

- ZeRO Stage 3 with parameter, gradient, and optimizer partitioning
- 16-bit model saving enabled
- Compatible with gradient checkpointing

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for IDA reward scoring via `gpt-4.1-mini` |
| `WANDB_API_KEY` | Optional, for experiment tracking |
| `CUDA_VISIBLE_DEVICES` | Used to allocate specific GPUs (e.g., victim server) |

You can set these in a `.env` file in the project root (loaded via `python-dotenv`).

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `trl` | 0.20.0 | SFTTrainer, GRPOTrainer |
| `vllm` | 0.10.0 | Fast LLM inference |
| `litellm` | 1.74.8 | Unified LLM API with disk caching |
| `deepspeed` | 0.17.4 | ZeRO-3 distributed optimization |
| `wandb` | 0.25.1 | Experiment tracking |
| `kernels` | 0.9.0 | Optimized kernel operations |
| `python-dotenv` | 1.1.0 | Environment variable management |
