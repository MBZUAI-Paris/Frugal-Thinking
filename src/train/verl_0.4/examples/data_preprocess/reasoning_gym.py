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

def parse_json_fields(example, idx):
    """Parse JSON string fields to dictionaries"""
    # Parse extra_info if it's a string
    if isinstance(example.get("extra_info"), str):
        try:
            example["extra_info"] = json.loads(example["extra_info"])
        except json.JSONDecodeError:
            example["extra_info"] = {"index": 0, "split": "train"} # with idx it tends to use multi-turn
    
    if isinstance(example.get("metadata"), str):
        try:
            example["metadata"] = json.loads(example["metadata"])
        except json.JSONDecodeError:
            example["metadata"] = {"data_source": "reasoning_gym"}
    
    return example

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
        default="/mnt/weka/home/abdelaziz.bounhar/k2/rl/data/reasoning_gym",
    )
    parser.add_argument(
        "--hdfs_dir", 
        default=None,
    )
    parser.add_argument(
        "--prompt_col_name", 
        type=str, 
        default="conversations",
        help="column from which to extract the prompt",
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

    args = parser.parse_args()

    DATASET_PATH = "MBZUAI-Paris/Reasoning-Gym-all-RL-split-curriculum"
    train_dataset = datasets.load_dataset(DATASET_PATH, split="train", num_proc=os.cpu_count() // 2)

    print(f'[INFO-data-preprocessing.reasoning_gym.py] train_dataset after loading: {train_dataset}')

    system_prompt = (
        "A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown and Latex to format your response. Write both your thoughts and summary in the same language as the task posed by the user.\n"
        "\n"
        "Your thinking process must follow the template below:\n"
        "Always format math answers inside \\boxed{} and code in ```python``` blocks.\n"
        "<think>\n"
        "Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual\n"
        "and as long as you want until you are confident to generate a correct answer.\n"
        "</think>\n"
        "\n"
        "<answer>\n"
        "Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user.\n"
        "</answer>\n"
        "\n"
        "Problem:"
    )
    # Parse extra_info from string to dict if needed
    train_dataset = train_dataset.map(
        parse_json_fields,
        with_indices=True,
        desc="Fixing extra_info to json..."
    )

    def _build_prompt_fields(example):
        """Build prompt with only {role, content} messages.
        - Keep system prompt first.
        - From conversations, keep only non-assistant messages.
        - Map {from,value} -> {role,content}. If `content` already exists, prefer it; otherwise use `value`.
        """
        messages = []
        convs = example.get(args.prompt_col_name)
        if isinstance(convs, list):
            for m in convs:
                if m.get("from") == "assistant":
                    continue  # skip assistant messages
                role = m.get("from", "user")
                # prefer existing 'content'; fallback to 'value'
                content = m.get("content")
                if content is None:
                    content = m.get("value", "")
                messages.append({"role": role, "content": content})
        return {
            **example,
            "prompt": [{"role": "system", "content": system_prompt}, *messages],
        }

    train_dataset = train_dataset.map(
        _build_prompt_fields,
        desc="Building prompt with role/content only...",
    )


    train_dataset = train_dataset.map(
        lambda row: {**row, 'ability': 'reasoning_gym'},
    )

    train_dataset.push_to_hub(
        "MBZUAI-Paris/Reasoning-Gym-all-RL-split-curriculum-ready",
        private=True,
    )

    # Group data_source values by ability and save as JSON
    ability_to_sources = defaultdict(set)
    for row in train_dataset:
        ability = row.get("ability") or "unknown"
        data_source = row.get("data_source") or "unknown"
        ability_to_sources[ability].add(data_source)
    # Convert sets to sorted lists for JSON serialization
    ability_to_sources = {k: sorted(list(v)) for k, v in ability_to_sources.items()}
    print("[INFO-data-preprocessing.reasoning_gym.py] Unique ability values:", set(train_dataset.unique("ability")))

    
    print(f'args.downsampling_ratio: {args.downsampling_ratio}')
        
    if args.downsampling_ratio < 1:
        # compute the new number of samples based on the chosen downsampling ratio
        n_samples = int(len(train_dataset) * args.downsampling_ratio)
        print(f"[INFO-data-preprocessing.reasoning_gym.py] Downsampling from {len(train_dataset)} to {n_samples}...")
        # only sample some training data for faster training
        train_dataset = train_dataset.shuffle(seed=args.downsampling_seed).select(range(n_samples))

    # add a row to each data item that represents a unique id
    def make_map_fn(
        split, 
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
                "prompt": [
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

    dataset_source = "math-ai/aime25"
    instruction_following = ''
    print(f'[INFO-data-preprocessing] For dataset {dataset_source} we are using instruction_following: {instruction_following}')
    
    test_dataset_aim25 = datasets.load_dataset(dataset_source, name="default", split="test", num_proc=os.cpu_count() // 2)
    test_dataset_aim25 = test_dataset_aim25.map(
        function=make_map_fn(
            split="test",
            question_col="problem",
            answer_col="answer",
            dataset_source=dataset_source,
            instruction_following=instruction_following,
        ), 
        with_indices=True,
    )

    test_dataset_aim25 = test_dataset_aim25.map(
        lambda x: {
            **x,
            "prompt": [
                {"role": "system", "content": system_prompt},
                *x["prompt"]
            ]
        }
    )

    # Remove 'id' and 'url' columns from test_dataset_aim25 to align schemas
    if 'id' in test_dataset_aim25.column_names:
        test_dataset_aim25 = test_dataset_aim25.remove_columns('id')

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset_aim25.to_parquet(os.path.join(local_dir, "test_aime25.parquet"))

    # Save to JSON file
    json_path = os.path.join(local_dir, "train_data_sources_by_ability.json")
    with open(json_path, "w") as f:
        json.dump(ability_to_sources, f, indent=2)

    print(f'[INFO-data-preprocessing] train_dataset: {train_dataset}')
    print(f'[INFO-data-preprocessing] test_dataset_aim25: {test_dataset_aim25}')


    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)