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

import copy
import random


def sample_test_cases(reward_model: dict, k: int = 1):
    """
    Sub‑sample up to *k* test cases in the reward_model["ground_truth"] field.

    Logic
    -----
    • ASSERT
        – Handles list of assertions **or** single newline‑separated blob.

    • STD
        – Two scenarios:
            1. len(inputs) == len(outputs) > 1
               -> Treat each pair as *one* test‑case and randomly keep ≤ k pairs.
            2. len(inputs) == len(outputs) == 1
               -> Competitive‑programming style encodings:
                  a. "N\n<case1>\n...<caseN>"  (one-line‑per‑case)
                  b. Same but each case spans the same number of lines.
               -> Randomly pick ≤ k embedded cases and rewrite header.

    Any shape that cannot be confidently handled is left untouched.
    """
    import json, copy, random

    rm = copy.deepcopy(reward_model)
    gt_raw = rm.get("ground_truth")

    # Parse ground_truth safely
    try:
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else copy.deepcopy(gt_raw)
    except Exception:
        return rm  # malformed JSON → bail out

    call_type = str(gt.get("call_type", "")).lower()

    # ------------------------------------------------------------------ ASSERT
    if call_type == "assert":
        ac = gt.get("assert_case", [])
        if isinstance(ac, str):
            ac = [ac]

        if len(ac) == 1 and "\n" in ac[0]:
            lines = [l for l in ac[0].splitlines() if l.strip()]
            if len(lines) > k:
                lines = random.sample(lines, k)
            gt["assert_case"] = ["\n".join(lines)]
        else:
            if len(ac) > k:
                ac = random.sample(ac, k)
            gt["assert_case"] = ac

    # -------------------------------------------------------------------- STD
    elif call_type == "std":
        inputs, outputs = gt.get("inputs", []), gt.get("outputs", [])
        if not inputs or not outputs:
            rm["ground_truth"] = json.dumps(gt)
            return rm

        # --------‑ Pair‑level sampling (multiple distinct io pairs)
        if len(inputs) == len(outputs) and len(inputs) > 1:
            if len(inputs) > k:
                idx = random.sample(range(len(inputs)), k)
                gt["inputs"] = [inputs[i] for i in idx]
                gt["outputs"] = [outputs[i] for i in idx]

        # --------‑ Embedded sampling (single io pair with header)
        elif len(inputs) == len(outputs) == 1:
            # Normalize first (and only) input/output pair to *strings*.
            def _to_text(x):
                if isinstance(x, list):
                    return "\n".join(str(e) for e in x)
                return str(x)

            inp = _to_text(inputs[0]).rstrip("\n")
            out = _to_text(outputs[0]).rstrip("\n")

            in_lines = [l for l in inp.splitlines() if l.strip()]
            out_lines = [l for l in out.splitlines() if l.strip()]

            if not in_lines or not out_lines:
                pass
            else:
                try:
                    N = int(in_lines[0])
                except ValueError:
                    # Header not integer → nothing we can do safely
                    pass
                else:
                    if N > k:
                        body_in = in_lines[1:]
                        if len(body_in) == N and len(out_lines) == N:
                            # One‑line per case
                            idx = random.sample(range(N), k)
                            new_in = [str(k)] + [body_in[i] for i in idx]
                            new_out = [out_lines[i] for i in idx]
                            gt["inputs"] = ["\n".join(new_in) + "\n"]
                            gt["outputs"] = ["\n".join(new_out) + "\n"]

                        elif len(body_in) % N == 0:
                            # Multi‑line per case (equal length chunks)
                            lines_per = len(body_in) // N
                            idx = random.sample(range(N), k)
                            new_in = [str(k)]
                            for i in idx:
                                new_in.extend(body_in[i * lines_per:(i + 1) * lines_per])

                            if len(out_lines) >= N:
                                new_out = [out_lines[i] for i in idx]
                            else:
                                new_out = out_lines  # fallback
                            gt["inputs"] = ["\n".join(new_in) + "\n"]
                            gt["outputs"] = ["\n".join(new_out) + "\n"]

    # ------------------------------------------------------------------------
    rm["ground_truth"] = json.dumps(gt)
    return rm


if __name__ == "__main__":

    train_dataset = datasets.load_dataset("a-m-team/AM-Thinking-v1-RL-Dataset", split="train", num_proc=os.cpu_count() // 2)
    print(f'[INFO-data-preprocessing] train_dataset after loading: {train_dataset}')


    instruction_following = "Let's think step by step, and write the final answer inside \\boxed{} within the <answer> </answer> tags."
    system_prompt = f"You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. {instruction_following}"
    train_dataset = train_dataset.map(
        lambda x: {
            **x,
            "prompt": [
                {"role": "system", "content": system_prompt},
                *([msg for msg in x["prompt"] if msg["role"] != "system"] if isinstance(x.get("prompt"), list) else [])
            ]
        }
    )

    print("[INFO] Unique ability values:", set(train_dataset.unique("ability")))

    # We only focus on code so we filter samples with unknown ability as they will create an issue
    train_dataset = train_dataset.filter(
        lambda row: row['ability'].upper() != 'UNKNOWN'
    )

    # We only analyze code samples
    train_dataset = train_dataset.filter(
        lambda row: str(row.get('ability', '')).lower() == 'code'
    )


    print(f'[INFO-data-preprocessing] train_dataset: {train_dataset}')


    # --- Analysis of call_type and test case distribution ---
    import json
    from collections import Counter
    import matplotlib.pyplot as plt
    import numpy as np

    def safe_json_parse(data):
        """Safely parse JSON data with error handling"""
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Problematic data: {repr(data[:100])}")  # Show first 100 chars
                return None
        return None

    counts = Counter()
    num_testcases = {"std": [], "assert": []}
    parse_errors = 0

    for i, row in enumerate(train_dataset):
        try:
            reward_model = row["reward_model"]
            ground_truth_raw = reward_model['ground_truth']
            gt = safe_json_parse(ground_truth_raw)
            
            if gt is None:
                parse_errors += 1
                print(f"Skipping row {i} due to JSON parsing error")
                continue
                
            call_type = gt.get("call_type", "")

            if call_type == "std":
                counts["std"] += 1
                num_testcases["std"].append(len(gt.get("inputs", [])))
            elif call_type == "assert":
                counts["assert"] += 1
                num_testcases["assert"].append(len(gt.get("assert_case", [])))
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            parse_errors += 1
            continue

    total = counts["std"] + counts["assert"]
    print(f'Total processed: {total}')
    print(f'Parse errors: {parse_errors}')
    print(f"STD: {counts['std']} ({counts['std'] / total * 100:.2f}%)" if total > 0 else "STD: 0")
    print(f"ASSERT: {counts['assert']} ({counts['assert'] / total * 100:.2f}%)" if total > 0 else "ASSERT: 0")

    # Print statistics for test case counts
    if num_testcases["std"]:
        std_mean = np.mean(num_testcases["std"])
        std_std = np.std(num_testcases["std"])
        print(f"STD test cases - Mean: {std_mean:.2f}, Std Dev: {std_std:.2f}")
    else:
        print("STD test cases - No data")

    if num_testcases["assert"]:
        assert_mean = np.mean(num_testcases["assert"])
        assert_std = np.std(num_testcases["assert"])
        print(f"ASSERT test cases - Mean: {assert_mean:.2f}, Std Dev: {assert_std:.2f}")
    else:
        print("ASSERT test cases - No data")

    # Only create plots if we have data
    if num_testcases["std"]:
        std_mean = np.mean(num_testcases["std"])
        std_std = np.std(num_testcases["std"])
        
        plt.figure(figsize=(10, 6))
        plt.hist(num_testcases["std"], bins=20, alpha=0.7, label="STD samples")
        plt.axvline(std_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {std_mean:.2f}')
        plt.axvline(std_mean + std_std, color='orange', linestyle=':', linewidth=2, label=f'+1 Std dev: {std_mean + std_std:.2f}')
        plt.axvline(std_mean - std_std, color='orange', linestyle=':', linewidth=2, label=f'-1 Std dev: {std_mean - std_std:.2f}')
        
        plt.xlabel("Number of test cases")
        plt.ylabel("Frequency")
        plt.title(f"STD test case distribution\nMean: {std_mean:.2f}, Std Dev: {std_std:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("std_testcase_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    if num_testcases["assert"]:
        assert_mean = np.mean(num_testcases["assert"])
        assert_std = np.std(num_testcases["assert"])
        
        plt.figure(figsize=(10, 6))
        plt.hist(num_testcases["assert"], bins=20, alpha=0.7, color="orange", label="ASSERT samples")
        plt.axvline(assert_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {assert_mean:.2f}')
        plt.axvline(assert_mean + assert_std, color='blue', linestyle=':', linewidth=2, label=f'+1 Std dev: {assert_mean + assert_std:.2f}')
        plt.axvline(assert_mean - assert_std, color='blue', linestyle=':', linewidth=2, label=f'-1 Std dev: {assert_mean - assert_std:.2f}')
        
        plt.xlabel("Number of test cases")
        plt.ylabel("Frequency")
        plt.title(f"ASSERT test case distribution\nMean: {assert_mean:.2f}, Std Dev: {assert_std:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("assert_testcase_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------------------
    #                       SUB‑SAMPLE TEST CASES
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_k",
        type=int,
        default=1,
        help="Number of test cases to keep per code example",
    )
    args, _ = parser.parse_known_args()
    sample_k = max(1, args.sample_k)
    print(f"[INFO-data-preprocessing] Sampling up to {sample_k} test case(s) per example")

    def _apply_sampling(row, debug=False):
        """
        Apply sampling to the reward_model **only** when the ability is 'code'.
        Prints the before/after for traceability.
        """
        if str(row.get("ability", "")).lower() == "code":
            before = row["reward_model"]
            after = sample_test_cases(before, k=sample_k)
            if debug:
                print("---- reward_model BEFORE ----")
                print(before)
                print("---- reward_model AFTER  ----")
                print(after)
            row["reward_model"] = after
        return row
    

    # train_dataset = train_dataset.shuffle(True).select(range(10))

    train_dataset = train_dataset.map(
        _apply_sampling,
        desc="Sampling test‑cases",
    )

    train_dataset.push_to_hub(
        "MBZUAI-Paris/AM-RL-Sampled",
        private=True
    )

    print(f"[INFO-data-preprocessing] train_dataset after sampling: {train_dataset}")

    counts = Counter()
    num_testcases = {"std": [], "assert": []}
    parse_errors = 0

    for i, row in enumerate(train_dataset):
        try:
            reward_model = row["reward_model"]
            ground_truth_raw = reward_model['ground_truth']
            gt = safe_json_parse(ground_truth_raw)
            
            if gt is None:
                parse_errors += 1
                print(f"Skipping row {i} due to JSON parsing error")
                continue
                
            call_type = gt.get("call_type", "")

            if call_type == "std":
                counts["std"] += 1
                num_testcases["std"].append(len(gt.get("inputs", [])))
            elif call_type == "assert":
                counts["assert"] += 1
                num_testcases["assert"].append(len(gt.get("assert_case", [])))
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            parse_errors += 1
            continue

    total = counts["std"] + counts["assert"]
    print(f'Total processed: {total}')
    print(f'Parse errors: {parse_errors}')
    print(f"STD: {counts['std']} ({counts['std'] / total * 100:.2f}%)" if total > 0 else "STD: 0")
    print(f"ASSERT: {counts['assert']} ({counts['assert'] / total * 100:.2f}%)" if total > 0 else "ASSERT: 0")

    # Print statistics for test case counts
    if num_testcases["std"]:
        std_mean = np.mean(num_testcases["std"])
        std_std = np.std(num_testcases["std"])
        print(f"STD test cases - Mean: {std_mean:.2f}, Std Dev: {std_std:.2f}")
    else:
        print("STD test cases - No data")

    if num_testcases["assert"]:
        assert_mean = np.mean(num_testcases["assert"])
        assert_std = np.std(num_testcases["assert"])
        print(f"ASSERT test cases - Mean: {assert_mean:.2f}, Std Dev: {assert_std:.2f}")
    else:
        print("ASSERT test cases - No data")

    # Only create plots if we have data
    if num_testcases["std"]:
        std_mean = np.mean(num_testcases["std"])
        std_std = np.std(num_testcases["std"])
        
        plt.figure(figsize=(10, 6))
        plt.hist(num_testcases["std"], bins=20, alpha=0.7, label="STD samples")
        plt.axvline(std_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {std_mean:.2f}')
        plt.axvline(std_mean + std_std, color='orange', linestyle=':', linewidth=2, label=f'+1 Std dev: {std_mean + std_std:.2f}')
        plt.axvline(std_mean - std_std, color='orange', linestyle=':', linewidth=2, label=f'-1 Std dev: {std_mean - std_std:.2f}')
        
        plt.xlabel("Number of test cases")
        plt.ylabel("Frequency")
        plt.title(f"STD test case distribution\nMean: {std_mean:.2f}, Std Dev: {std_std:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("sampled_std_testcase_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    if num_testcases["assert"]:
        assert_mean = np.mean(num_testcases["assert"])
        assert_std = np.std(num_testcases["assert"])
        
        plt.figure(figsize=(10, 6))
        plt.hist(num_testcases["assert"], bins=20, alpha=0.7, color="orange", label="ASSERT samples")
        plt.axvline(assert_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {assert_mean:.2f}')
        plt.axvline(assert_mean + assert_std, color='blue', linestyle=':', linewidth=2, label=f'+1 Std dev: {assert_mean + assert_std:.2f}')
        plt.axvline(assert_mean - assert_std, color='blue', linestyle=':', linewidth=2, label=f'-1 Std dev: {assert_mean - assert_std:.2f}')
        
        plt.xlabel("Number of test cases")
        plt.ylabel("Frequency")
        plt.title(f"ASSERT test case distribution\nMean: {assert_mean:.2f}, Std Dev: {assert_std:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("sampled_assert_testcase_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()