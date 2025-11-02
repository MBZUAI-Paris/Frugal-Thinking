"""
Saves the MBZUAI-Paris/router-training-rl-data train dataset and AIME25 test sets to parquet format.
"""

import argparse
import os
import re
import json
from collections import defaultdict

import datasets

from verl.utils.hdfs_io import copy, makedirs

def extract_solution_aime24(solution_str):
    # Try to extract answer inside \\boxed{...}
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", solution_str)
    if boxed_match:
        return boxed_match.group(1).replace(",", "")
    raise ValueError("No boxed answer found in aime24 solution string.")


SYSTEM_PROMPT = (
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir", 
        default="/mnt/weka/home/abdelaziz.bounhar/rl/data/btx_data",
    )
    parser.add_argument(
        "--hf_data_path", 
        default="MBZUAI-Paris/math-code-puzzles-rl-data",
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
        "--with_system_prompt", 
        action='store_true',
    )
    
    parser.add_argument(
        "--difficulty_col_name", 
        type=str, 
        default="difficulty",
        help="Name of the difficulty column in the dataset",
    )
    parser.add_argument(
        "--curriculum_learning", 
        action='store_true',
    )
    args = parser.parse_args()

    # DATASET_PATH = "MBZUAI-Paris/router-training-rl-data"
    DATASET_PATH = args.hf_data_path
    train_dataset = datasets.load_dataset(DATASET_PATH, split="train", num_proc=os.cpu_count() // 2)

    print(f'[INFO-data-preprocessing.btx_data.py] train_dataset after loading: {train_dataset}')
        # Group data_source values by ability and save as JSON
    ability_to_sources = defaultdict(set)
    for row in train_dataset:
        ability = row.get("ability") or "unknown"
        data_source = row.get("data_source") or "unknown"
        ability_to_sources[ability].add(data_source)
    # Convert sets to sorted lists for JSON serialization
    ability_to_sources = {k: sorted(list(v)) for k, v in ability_to_sources.items()}
    print("[INFO-data-preprocessing.btx_data.py] Unique ability values:", set(train_dataset.unique("ability")))
    
    # Samples with unknown ability as they will create an issue in reward calculation, NotImplementedError
    print(f"[INFO-data-preprocessing.btx_data.py] Filtering out samples with unknown ability. Original size: {len(train_dataset)}")
    train_dataset = train_dataset.filter(
        lambda row: row['ability'].upper() != 'UNKNOWN'
    )
    print(f"[INFO-data-preprocessing.btx_data.py] Size after filtering unknown ability: {len(train_dataset)}")
    
    if args.keep_only_math:
        print('[INFO-data-preprocessing.btx_data.py] Keeping only math problems.')
        train_dataset = train_dataset.filter(
            lambda row: str(row.get('ability', '')).lower() == 'math'
        )

    if args.keep_only_code:
        print('[INFO-data-preprocessing.btx_data.py] Keeping only code problems.')
        train_dataset = train_dataset.filter(
            lambda row: str(row.get('ability', '')).lower() == 'code'
        )
        
    if args.with_system_prompt:
        # Add system prompt to the beginning of each prompt list
        print('[INFO-data-preprocessing.btx_data.py] TRAIN DATA - Adding system prompt to each data item.')
        train_dataset = train_dataset.map(
            lambda x: {
                **x,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *([msg for msg in x["prompt"] if msg["role"] != "system"] if isinstance(x.get("prompt"), list) else [])
                ]
            }
        )
    else:
        print('[INFO-data-preprocessing.btx_data.py] TRAIN DATA - Not adding system prompt to each data item.')
        train_dataset = train_dataset.map(
            lambda x: {
                **x,
                "prompt": [
                    {"role": "system", "content": ""},
                    *([msg for msg in x["prompt"] if msg["role"] != "system"] if isinstance(x.get("prompt"), list) else [])
                ]
            }
        )
        
    # if curriculum learning, order by difficulty column name
    if args.curriculum_learning:
        print(f'[INFO-data-preprocessing.btx_data.py] Applying curriculum learning by sorting on column: {args.difficulty_col_name}')
        if args.difficulty_col_name in train_dataset.column_names:
            train_dataset = train_dataset.sort(args.difficulty_col_name)
        else:
            print(f'[WARNING-data-preprocessing.btx_data.py] Difficulty column name {args.difficulty_col_name} not found in dataset columns: {train_dataset.column_names}. Skipping sorting.')

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
            if 'aime25' in dataset_source:
                solution = f"\\boxed{{{answer_raw}}}"
            elif 'aime24' in dataset_source:
                solution = extract_solution_aime24(answer_raw)
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
    instruction_following = 'Let\'s think step by step and output the final answer inside \\boxed{}.'
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
                {"role": "system", "content": ""},
                *x["prompt"]
            ]
        }
    )
    
    if args.with_system_prompt:
        # Add system prompt to the beginning of each prompt list
        print('[INFO-data-preprocessing.btx_data.py] AIME25 - Adding system prompt to each data item.')
        test_dataset_aim25 = test_dataset_aim25.map(
            lambda x: {
                **x,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *x["prompt"]
                ]
            }
        )
    else:
        print('[INFO-data-preprocessing.btx_data.py] AIME25 - Not adding system prompt to each data item.')
        test_dataset_aim25 = test_dataset_aim25.map(
            lambda x: {
                **x,
                "prompt": [
                    {"role": "system", "content": ""},
                    *x["prompt"]
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
            question_col="problem",
            answer_col="solution",
            dataset_source=dataset_source,
            instruction_following=instruction_following,
        ), 
        with_indices=True,
    )
    
    if args.with_system_prompt:
        # Add system prompt to the beginning of each prompt list
        print('[INFO-data-preprocessing.btx_data.py] AIME24 - Adding system prompt to each data item.')
        test_dataset_aim24 = test_dataset_aim24.map(
            lambda x: {
                **x,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *x["prompt"]
                ]
            }
        )
    else:
        print('[INFO-data-preprocessing.btx_data.py] AIME24 - Not adding system prompt to each data item.')
        test_dataset_aim24 = test_dataset_aim24.map(
            lambda x: {
                **x,
                "prompt": [
                    {"role": "system", "content": ""},
                    *x["prompt"]
                ]
            }
        )

    # Remove 'id' and 'url' columns from test_dataset_aim24 to align schemas
    if 'id' in test_dataset_aim24.column_names:
        test_dataset_aim24 = test_dataset_aim24.remove_columns('id')
    if 'url' in test_dataset_aim24.column_names:
        test_dataset_aim24 = test_dataset_aim24.remove_columns('url')
        
    # Remove 'id' and 'url' columns from test_dataset_aim25 to align schemas
    if 'id' in test_dataset_aim25.column_names:
        test_dataset_aim25 = test_dataset_aim25.remove_columns('id')

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset_aim24.to_parquet(os.path.join(local_dir, "test_aime24.parquet"))
    test_dataset_aim25.to_parquet(os.path.join(local_dir, "test_aime25.parquet"))

    # Save to JSON file
    json_path = os.path.join(local_dir, "train_data_sources_by_ability.json")
    with open(json_path, "w") as f:
        json.dump(ability_to_sources, f, indent=2)

    print(f'[INFO-data-preprocessing] train_dataset: {train_dataset}')
    print(f'[INFO-data-preprocessing] test_dataset_aim24: {test_dataset_aim24}')
    print(f'[INFO-data-preprocessing] test_dataset_aim25: {test_dataset_aim25}')


    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)