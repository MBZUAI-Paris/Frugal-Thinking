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
        # Extract answer inside \boxed{...}. This also tests the formatting of the model thereby seeing if it follows the instruction_following given in the pronpt.
        match = re.search(r"\\boxed\{([^}]*)\}", solution_str)

        if match is None:
            final_answer = None
        else:
            final_answer =  match.group(1).replace(",", "")

        # print(f'solution_str: {solution_str}')
        # print(f'match: {match}')
        # print(f'final_answer: {final_answer}')
        # print(f'-------------------')

        # with open('/mnt/weka/home/abdelaziz.bounhar/k2/rl/logs/k2-rl-32B/train/multinode/aime_debug.txt', 'w+') as f:
        #     f.write(f'solution_str: {solution_str}')
        #     f.write('')
        #     f.write(f'match: {match}')
        #     f.write('')
        #     f.write(f'final_answer: {final_answer}')
        #     f.write('')
        #     f.write(f'-------------------')

    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    # print(f"answer: {answer}")  # Expected: 1.0
    # print(f"ground_truth: {ground_truth}")  # Expected: 1.0
    # print(f"answer == ground_truth: {answer == ground_truth}")  # Expected: 1.0
    if "boxed" in ground_truth:
        ground_truth = extract_solution(solution_str=ground_truth, method=method)

    # print(f"answer == ground_truth: {answer == ground_truth}")  # Expected: 1.0

    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score

if __name__ == "__main__":
    # Simple test case
    # solution = "<think>This is how I solved it</think><answer>\\boxed{70}</answer>"
    # gt = "70"
    # reward = compute_score(solution, gt)
    # print(f"Test reward: {reward}")  # Expected: 1.0

    # System prompt test
    instruction_following = "Let's think step by step, and write the final answer inside \\boxed{} within the <answer> </answer> tags."
    system_prompt = f"You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. {instruction_following}"
    prompt_example = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is 6 times 7?"}
    ]
    solution = "<think>6 times 7 is 42</think>\\boxed{42}"
    gt = "\\boxed{42}"
    # gt = "42"

    reward = compute_score(solution, gt)
    print(f"Test reward with system prompt: {reward}")  # Expected: 1.0