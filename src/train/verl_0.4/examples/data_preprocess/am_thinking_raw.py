"""
Saves the AM-Thinking-v1-RL-Dataset train dataset and GSM8K, AIME24 and AIME25 test sets to parquet format.
"""

import argparse
import os
import re
import json
from collections import defaultdict

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution_gsm8k(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_solution_aime24(solution_str):
    # Try to extract answer inside \\boxed{...}
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", solution_str)
    if boxed_match:
        return boxed_match.group(1).replace(",", "")
    raise ValueError("No boxed answer found in aime24 solution string.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir", 
        default="/mnt/weka/home/abdelaziz.bounhar/k2/rl/data/am_thinking_v1_rl_ds",
    )
    parser.add_argument(
        "--hdfs_dir", 
        default=None,
    )
    parser.add_argument(
        "--downsampling_ratio", 
        type=float, 
        default=1.0,
        help="ratio of samples to keep for faster training",
    )
    parser.add_argument(
        "--downsampling_seed", 
        type=int, 
        default=1998,
        help="sampling seed for reproducibility",
    )
    parser.add_argument(
        "--keep_only_math", 
        action='store_true',
    )
    parser.add_argument(
        "--keep_only_code", 
        action='store_true',
    )
    parser.add_argument(
        "--async_rollout", 
        action='store_true',
        help="Async mode requires different name for prompt and answer columns",
    )

    args = parser.parse_args()

    train_dataset = datasets.load_dataset("a-m-team/AM-Thinking-v1-RL-Dataset", split="train", num_proc=os.cpu_count() // 2)
    print(f'[INFO-data-preprocessing.am_thinking.py] train_dataset after loading: {train_dataset}')

    instruction_following = "Let's think step by step, and write the final answer inside \\boxed{} within the <answer> </answer> tags."
    system_prompt = f"You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. {instruction_following}"
    
    if args.async_rollout:
        prompt_col = "raw_prompt"
        print(f'[INFO-data-preprocessing.am_thinking.py] Using raw_prompt as prompt column name for async rollout')
    else:
        prompt_col = "prompt"
        print(f'[INFO-data-preprocessing.am_thinking.py] Using prompt as prompt column name for sync rollout')

    train_dataset = train_dataset.map(
        lambda x: {
            **x,
            prompt_col: [
                {"role": "system", "content": system_prompt},
                *([msg for msg in x["prompt"] if msg["role"] != "system"] if isinstance(x.get("prompt"), list) else [])
            ]
        }
    )

    # Group data_source values by ability and save as JSON
    ability_to_sources = defaultdict(set)
    for row in train_dataset:
        ability = row.get("ability") or "unknown"
        data_source = row.get("data_source") or "unknown"
        ability_to_sources[ability].add(data_source)
    # Convert sets to sorted lists for JSON serialization
    ability_to_sources = {k: sorted(list(v)) for k, v in ability_to_sources.items()}
    print("[INFO-data-preprocessing.am_thinking.py] Unique ability values:", set(train_dataset.unique("ability")))

    # We only focus on code so we filter samples with unknown ability as they will create an issue
    train_dataset = train_dataset.filter(
        lambda row: row['ability'].upper() != 'UNKNOWN'
    )
    
    if args.keep_only_math:
        train_dataset = train_dataset.filter(
            lambda row: str(row.get('ability', '')).lower() == 'math'
        )

    if args.keep_only_code:
        train_dataset = train_dataset.filter(
            lambda row: str(row.get('ability', '')).lower() == 'code'
        )

    print(f'args.downsampling_ratio: {args.downsampling_ratio}')
        
    if args.downsampling_ratio < 1:
        # compute the new number of samples based on the chosen downsampling ratio
        n_samples = int(len(train_dataset) * args.downsampling_ratio)
        print(f"[INFO-data-preprocessing.am_thinking.py] Downsampling from {len(train_dataset)} to {n_samples}...")
        # only sample some training data for faster training
        train_dataset = train_dataset.shuffle(seed=args.downsampling_seed).select(range(n_samples))

    # add a row to each data item that represents a unique id
    def make_map_fn(
        split, 
        prompt_col_name,
        question_col="question",
        answer_col="answer",
        dataset_source="openai/gsm8k",
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    ):
        def process_fn(example, idx):
            question_raw = example.pop(question_col)

            question = question_raw + " " + instruction_following

            answer_raw = example.pop(answer_col)
            if 'gsm8k' in dataset_source:
                solution = extract_solution_gsm8k(answer_raw)
            elif 'aime24' in dataset_source:
                solution = extract_solution_aime24(answer_raw)
            elif 'aime25' in dataset_source:
                solution = f"\\boxed{{{answer_raw}}}"
            else:
                raise NotImplementedError(f"Dataset not recognized: {dataset_source}")
                
            data = {
                "data_source": dataset_source,
                prompt_col_name: [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    dataset_source = "openai/gsm8k"
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    print(f'[INFO-data-preprocessing] For dataset {dataset_source} we are using instruction_following: {instruction_following}')

    test_dataset_gsm8k = datasets.load_dataset(dataset_source, name="main", split="test", num_proc=os.cpu_count() // 2)
    test_dataset_gsm8k = test_dataset_gsm8k.map(
        function=make_map_fn(
            split="test",
            prompt_col_name=prompt_col,
            question_col="question",
            answer_col="answer",
            dataset_source=dataset_source,
            instruction_following=instruction_following,
        ), 
        with_indices=True,
    )
    system_prompt = f"You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. {instruction_following}"
    test_dataset_gsm8k = test_dataset_gsm8k.map(
        lambda x: {
            **x,
            prompt_col: [
                {"role": "system", "content": system_prompt},
                *x[prompt_col]
            ]
        }
    )

    dataset_source = "math-ai/aime24"
    instruction_following = 'Let\'s think step by step and output the final answer inside \\boxed{}.'
    print(f'[INFO-data-preprocessing] For dataset {dataset_source} we are using instruction_following: {instruction_following}')

    test_dataset_aim24 = datasets.load_dataset(dataset_source, name="default", split="test", num_proc=os.cpu_count() // 2)
    test_dataset_aim24 = test_dataset_aim24.map(
        function=make_map_fn(
            split="test",
            prompt_col_name=prompt_col,
            question_col="problem",
            answer_col="solution",
            dataset_source=dataset_source,
            instruction_following=instruction_following,
        ), 
        with_indices=True,
    )
    system_prompt = f"You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. {instruction_following}"
    test_dataset_aim24 = test_dataset_aim24.map(
        lambda x: {
            **x,
            prompt_col: [
                {"role": "system", "content": system_prompt},
                *x[prompt_col]
            ]
        }
    )

    # Remove 'id' and 'url' columns from test_dataset_aim24 to align schemas
    if 'id' in test_dataset_aim24.column_names:
        test_dataset_aim24 = test_dataset_aim24.remove_columns('id')
    if 'url' in test_dataset_aim24.column_names:
        test_dataset_aim24 = test_dataset_aim24.remove_columns('url')

    dataset_source = "math-ai/aime25"
    instruction_following = 'Let\'s think step by step and output the final answer inside \\boxed{}.'
    print(f'[INFO-data-preprocessing] For dataset {dataset_source} we are using instruction_following: {instruction_following}')
    
    test_dataset_aim25 = datasets.load_dataset(dataset_source, name="default", split="test", num_proc=os.cpu_count() // 2)
    test_dataset_aim25 = test_dataset_aim25.map(
        function=make_map_fn(
            split="test",
            prompt_col_name=prompt_col,
            question_col="problem",
            answer_col="answer",
            dataset_source=dataset_source,
            instruction_following=instruction_following,
        ), 
        with_indices=True,
    )
    system_prompt = f"You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. {instruction_following}"
    test_dataset_aim25 = test_dataset_aim25.map(
        lambda x: {
            **x,
            prompt_col: [
                {"role": "system", "content": system_prompt},
                *x[prompt_col]
            ]
        }
    )

    # Remove 'id' and 'url' columns from test_dataset_aim25 to align schemas
    if 'id' in test_dataset_aim25.column_names:
        test_dataset_aim25 = test_dataset_aim25.remove_columns('id')

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset_gsm8k.to_parquet(os.path.join(local_dir, "test_gsm8k.parquet"))
    test_dataset_aim24.to_parquet(os.path.join(local_dir, "test_aime24.parquet"))
    test_dataset_aim25.to_parquet(os.path.join(local_dir, "test_aime25.parquet"))

    # Save to JSON file
    json_path = os.path.join(local_dir, "train_data_sources_by_ability.json")
    with open(json_path, "w") as f:
        json.dump(ability_to_sources, f, indent=2)

    print(f'[INFO-data-preprocessing] train_dataset: {train_dataset}')
    print(f'[INFO-data-preprocessing] test_dataset_gsm8k: {test_dataset_gsm8k}')
    print(f'[INFO-data-preprocessing] test_dataset_aim24: {test_dataset_aim24}')
    print(f'[INFO-data-preprocessing] test_dataset_aim25: {test_dataset_aim25}')


    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
