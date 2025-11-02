# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def extract_solution(solution_str, method="strict"):
    assert method in ["strict"] # implement others later

    if method == "strict":
        # Extract answer inside \\boxed{...}. This also tests the formatting of the model thereby seeing if it follows the instruction_following given in the pronpt.
        match = re.search(r"\\boxed\{([^}]*)\}", solution_str)

        if match is None:
            final_answer = None
        else:
            final_answer =  match.group(1).replace(",", "")

    return final_answer


def compute_score(solution_str, ground_truth, method="strict", only_if_correct_format=False):
    """The scoring function for reasoning_gym RL dataset.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
    """

    # --- tag‑based rewards ---------------------------------------------------
    tags = [
        "think",
        "answer",
    ]

    tag_rewards = 0.0
    tag_present = {}
    for tag in tags:
        # Check for <tag>...</tag>
        found = re.search(rf"<{tag}>.*?</{tag}>", solution_str, re.DOTALL) is not None
        tag_present[tag] = found
        if found:
            tag_rewards += 0.01

    think_present = tag_present["think"]
    answer_tag_present = tag_present["answer"]

    # --- answer‑based reward -------------------------------------------------
    if only_if_correct_format:
        # We follow mistral's approach and extract only if tags are correct
        if think_present and answer_tag_present:
            answer = extract_solution(solution_str=solution_str, method=method)
            answer_reward = 0.8 if answer is not None and answer == ground_truth else 0.0
            total_reward = tag_rewards + answer_reward
        else:
            total_reward = tag_rewards
    else:
        answer = extract_solution(solution_str=solution_str, method=method)
        answer_reward = 0.8 if answer is not None and answer == ground_truth else 0.0
        total_reward = tag_rewards + answer_reward

    return total_reward
    

if __name__ == "__main__":
    # Simple test case
    solution = "<think>This is how I solved it</think><answer>\\boxed{42}</answer>"
    gt = "42"
    reward = compute_score(solution, gt)
    print(f"Test reward: {reward}")  # Expected: 1.0

    # Simple test case
    solution = "<think>This is how I solved it</think><answer>\\\\boxed{&}</answer>"
    gt = "42"
    reward = compute_score(solution, gt)
    print(f"Test reward: {reward}")  # Expected: 0.2