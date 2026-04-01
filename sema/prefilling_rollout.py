import os
import re
import json
from jinja2 import Template
import argparse
import pandas as pd
import datasets
from vllm import LLM, SamplingParams
from utils import setup_seed


def parse_adv_prompt(output_text, max_turns=7):
    try:
        adv_prompt_mt = re.findall(r'(\d+\.)\s*(.*?)(?=\n\d+\.|$)', '1. '+output_text, re.DOTALL)
        numbers = [int(item[0][:-1]) for item in adv_prompt_mt]
        adv_prompt = [item[1].strip() for item in adv_prompt_mt]
        adv_prompt = adv_prompt[:max_turns]
    except Exception as e:
        print(f"Error processing completion: {e}")
        adv_prompt = []
    return adv_prompt


def prefilling_rollout(args):
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    output_file = args.output_file
    dataset_name = args.dataset_name
    data_split_seed = args.data_split_seed
    test_size = args.test_size
    system_prompt_path = args.system_prompt_path
    query_prompt_path = args.query_prompt_path
    batch_size = args.batch_size
    max_turns = args.max_turns
    n_generations = args.n_generations
    temperature = args.temperature
    max_tokens = args.max_tokens

    if dataset_name == "advbench":
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
        raise ValueError(f"Not supported dataset name: {dataset_name}")

    dataset = dataset.train_test_split(test_size=test_size, seed=data_split_seed)


    with open(system_prompt_path, "r") as f:
        SYSTEM_PROMPT_TEMPLATE = Template(f.read())

    with open(query_prompt_path, "r") as f:
        QUERY_PROMPT_TEMPLATE = Template(f.read())

    system_prompt = SYSTEM_PROMPT_TEMPLATE.render(max_turns=max_turns)
    def make_prefilling_prompt(example):
        query_prompt = QUERY_PROMPT_TEMPLATE.render(request=example['request'])
        prefilling_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt},
            {"role": "assistant", "content": "1. "},
        ]
        return {"prompt": prefilling_prompt}

    dataset = dataset.map(make_prefilling_prompt)
    train_dataset = dataset["train"]


    model = LLM(model=model_name_or_path, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(temperature=temperature, n=n_generations, max_tokens=max_tokens)
    prefilled_rollouts = []
    for i in range(0, len(train_dataset), batch_size):
        batch_samples = train_dataset[i:i+batch_size]
        batch_requests = batch_samples['request']
        batch_prompts = batch_samples['prompt']
        outputs = model.chat(
            batch_prompts, 
            add_generation_prompt=False, 
            continue_final_message=True, 
            sampling_params=sampling_params
            )

        for j, output in enumerate(outputs):
            request = batch_requests[j]
            prompt = batch_prompts[j]
            for k, o in enumerate(output.outputs):
                output_text = o.text

                adv_prompt = parse_adv_prompt(output_text)

                sample = {
                    "dataset_name": "advbench",
                    "sample_id": i + j,
                    "request": request,
                    "attacker": model_name_or_path,
                    "adv_prompt": adv_prompt,
                    "attempt_id": k,
                    "input_prompt": prompt,
                    "output_text": output_text,
                }
                
                prefilled_rollouts.append(sample)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), "w") as f:
        json.dump(prefilled_rollouts, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--dataset_name", type=str, default="advbench")
    parser.add_argument("--data_split_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_turns", type=int, default=7)
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--system_prompt_path", type=str, default="sema/prompts/sema_system.jinja")
    parser.add_argument("--query_prompt_path", type=str, default="sema/prompts/sema_query.jinja")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="files/prefilling_rollouts")
    parser.add_argument("--output_file", type=str, default="llama8b_prefilling_rollout.json")

    args = parser.parse_args()
    
    setup_seed(args.seed)
    prefilling_rollout(args)


    
