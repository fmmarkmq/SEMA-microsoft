import os
import asyncio
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import datasets
from dataclasses import dataclass, field
from jinja2 import Template

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from trl import get_kbit_device_map, get_peft_config, get_quantization_config

from victims.victims import VLLMVictimWrapper
from rewards.format_reward import NumberedListFormatReward
from rewards.intent_drift_aware_reward import IntentDriftAwareReward
from rewards.ablations.no_intent_alignment_reward import NoIntendAlignmentReward
from rewards.ablations.no_refusal_reward import NoRefusalPhraseReward
from utils import setup_seed


@dataclass
class DataConfig:
    dataset_name: str = field(
        default="advbench",
        metadata={"help": "The name of the dataset to use."}
    )
    data_split_seed: int = field(
        default=42,
        metadata={"help": "The seed to use for splitting the dataset."}
    )
    test_size: float = field(
        default=0.2,
        metadata={"help": "The size of the test set."}
    )

@dataclass
class RewardConfig:
    max_turns: int = field(
        default=7,
        metadata={"help": "The maximum number of turns in the conversation."}
    )
    victim_model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        metadata={"help": "The model to use as the victim for evaluating the generated prompts."}
    )
    gpu_memory_utilization: float = field(
        default=0.4,
        metadata={"help": "The GPU memory utilization to use for the victim model."}
    )
    max_tokens: int = field(
        default=500,
        metadata={"help": "The maximum number of tokens in the conversation."}
    )
    max_model_len: int = field(
        default=4000,
        metadata={"help": "The maximum length of the victim model."}
    )
    score_reward_name: str = field(
        default="ida_reward",
        metadata={
            "help": "The name of the score reward to use.",
            "choices": ["ida_reward", "no_intent_alignment_reward", "no_refusal_reward"]
        }
    )
    score_reward_model_name: str = field(
        default="openai/gpt-4.1-mini-2025-04-14",
        metadata={"help": "The model to use for the score reward."},
    )
    reward_log_save_dir: str = field(
        default="files/logs",
        metadata={"help": "The directory to save the reward logs."}
    )


class SimulationRewardPipeline:
    def __init__(
        self, 
        victim, 
        score_reward_name, 
        score_reward_model_name="openai/gpt-4.1-mini-2025-04-14",
        max_turns=7, 
        save_dir=None
        ):
        self.max_turns = max_turns
        self.victim = victim
        self.format_reward = NumberedListFormatReward(max_number=max_turns, return_dict=True)
        self.score_reward_name = score_reward_name
        self.score_reward_model_name = score_reward_model_name
        self.score_reward = self.setup_score_reward(score_reward_name, score_reward_model_name)
        self.save_dir = save_dir

        self.global_step = 0
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0
        
        self.logs = []


    def setup_score_reward(self, score_reward_name, score_reward_model_name):
        if score_reward_name == "ida_reward":
            score_reward = IntentDriftAwareReward(
                model_id=score_reward_model_name,
                template_name='ida_reward_system',
                template_root='sema/prompts',
            )
        elif score_reward_name == "no_intent_alignment_reward":
            score_reward = NoIntendAlignmentReward(
                model_id=score_reward_model_name,
                template_name='no_intent_alignment_reward_system',
                template_root='sema/prompts',
            )
        elif score_reward_name == "no_refusal_reward":
            score_reward = NoRefusalPhraseReward(
                return_dict=True,
            )
        else:
            raise ValueError(f"Invalid score reward name: {score_reward_name}")
        return score_reward


    def save_logs(self):
        os.makedirs(self.save_dir, exist_ok=True)
        filename = f"reward_logs_rank{self.rank}.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)


    def reward(self, completions, **kwargs):
        self.global_step += 1

        format_reward_outputs = self.format_reward.reward_batch(completions)
        format_reward_values = [output['reward'] for output in format_reward_outputs]
        adv_prompts = [output['parsed_items'] for output in format_reward_outputs]

        responses_batch = asyncio.run(self.victim.execute_batch(adv_prompts))
        last_response_batch = []
        for i, response in enumerate(responses_batch):
            if isinstance(response, list):
                if len(response) > 0:
                    last_response = response[-1]
                else:
                    last_response = ""
            else:
                last_response = response
            last_response_batch.append(last_response)

        requests = kwargs['request']
        score_reward_outputs = asyncio.run(self.score_reward.reward_batch(
            requests=requests,
            responses=last_response_batch,
        ))
        score_reward_values = [output['reward'] for output in score_reward_outputs]

        rewards = np.array(format_reward_values) + np.array(score_reward_values)
        rewards = rewards.tolist()

        self.logs.append({
            "step": self.global_step,
            "rank": self.rank,
            "timestamp": datetime.now().isoformat(),
            "num_completions": len(completions),
            "requests": requests,
            "completions": [completion[-1]['content'] if completion else "" for completion in completions],
            "adv_prompts": adv_prompts,
            "format_rewards": format_reward_values,
            "score_rewards": score_reward_values,
            "final_rewards": rewards,
            "responses": responses_batch,
            "last_response": last_response_batch
        })

        if self.save_dir is not None:
            self.save_logs()

        return rewards


def main(data_args, training_args, model_args, reward_args):
    # ------------Load model & tokenizer------------
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # ------------Load & preprocess dataset------------
    if data_args.dataset_name == "advbench":
        advbench_path = "data/advbench.csv"
        if not os.path.exists(advbench_path):
            os.makedirs("data", exist_ok=True)
            from urllib.request import urlretrieve
            # we use a specific commit to ensure the data is consistent and won't change over time
            url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/a62d1307e38b3a076e614b20781c785fd860d813/data/advbench/harmful_behaviors.csv"
            urlretrieve(url, advbench_path)
        advbench_df = pd.read_csv(advbench_path)
        advbench_df = advbench_df.rename(columns={"goal": "request"})  
        advbench_data = advbench_df.to_dict(orient="records")
        dataset = datasets.Dataset.from_list(advbench_data)
    else:
        raise ValueError(f"Not supported dataset name: {data_args.dataset_name}")

    dataset = dataset.train_test_split(test_size=data_args.test_size, seed=data_args.data_split_seed)

    system_prompt_path = "sema/prompts/sema_system.jinja"
    query_prompt_path = "sema/prompts/sema_query.jinja"

    with open(system_prompt_path, "r") as f:
        SYSTEM_PROMPT_TEMPLATE = Template(f.read())

    with open(query_prompt_path, "r") as f:
        QUERY_PROMPT_TEMPLATE = Template(f.read())

    system_prompt = SYSTEM_PROMPT_TEMPLATE.render(max_turns=reward_args.max_turns)

    def make_conversation(example):
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUERY_PROMPT_TEMPLATE.render(request=example['request'])},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None



    # ------------Setup Simulation-grounded reward------------
    victim = VLLMVictimWrapper(
        model_path=reward_args.victim_model_name_or_path,
        gpu_memory_utilization=reward_args.gpu_memory_utilization,
        distributed_executor_backend="external_launcher",
        max_tokens=reward_args.max_tokens,
        max_model_len=reward_args.max_model_len,
        )

    simulation_reward_pipeline = SimulationRewardPipeline(
        victim=victim,
        score_reward_name=reward_args.score_reward_name,
        score_reward_model_name=reward_args.score_reward_model_name,
        max_turns=reward_args.max_turns,
        save_dir=reward_args.reward_log_save_dir,
    )


    # ------------Train model via GRPO------------
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=simulation_reward_pipeline.reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()


    # ------------Save model------------
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(model_name=training_args.output_dir.split("/")[-1])


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "SEMA"
    parser = TrlParser((DataConfig, GRPOConfig, ModelConfig, RewardConfig))
    data_args, training_args, model_args, reward_args = parser.parse_args_and_config()

    setup_seed(training_args.seed)
    main(data_args, training_args, model_args, reward_args)

