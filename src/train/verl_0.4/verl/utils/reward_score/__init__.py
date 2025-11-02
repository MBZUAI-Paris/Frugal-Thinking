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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated
import re

def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    ability=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    only_if_correct_format=False,
    formal_math_training=False,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    
    elif data_source == "math-ai/aime24" or data_source == "math-ai/aime25":
        from . import aime
        
        res = aime.compute_score(solution_str, ground_truth)

    elif ability.upper() == 'MATH':
        from . import am_thinking_rl_v1, am_thinking_rl_v1_formal_math
        if formal_math_training:
            # includes more formating reward
            res = am_thinking_rl_v1_formal_math.compute_score(solution_str, ground_truth, only_if_correct_format=only_if_correct_format)
        else:
            res = am_thinking_rl_v1.compute_score(solution_str, ground_truth, only_if_correct_format=only_if_correct_format)

        # # For enhanced accuracy, we use Math-Verify (https://github.com/huggingface/Math-Verify). 
        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)

        # Math-Verify crashes
        # # File "/mnt/weka/home/abdelaziz.bounhar/miniconda3/envs/verl/lib/python3.10/site-packages/math_verify/grader.py", line 274, in sympy_solve_and_compare
        # # 827
        # # 2025-06-13 09:09:02
        # #     solved_gold = list(ordered(solve(gold, gold.free_symbols)))
        # # 828
        # # 2025-06-13 09:09:02
        # # File "/mnt/weka/home/abdelaziz.bounhar/miniconda3/envs/verl/lib/python3.10/site-packages/sympy/solvers/solvers.py", line 1170, in solve
        # # 829
        # # 2025-06-13 09:09:02
        # #     solution = _solve(f[0], *symbols, **flags)
        # # 830
        # # 2025-06-13 09:09:02
        # # File "/mnt/weka/home/abdelaziz.bounhar/miniconda3/envs/verl/lib/python3.10/site-packages/sympy/solvers/solvers.py", line 1729, in _solve
        # # 831
        # # 2025-06-13 09:09:02
        # #     raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))
        # # 832
        # # 2025-06-13 09:09:02
        # # NotImplementedError: multiple generators [x, g(x)]
        # # 833
        # # 2025-06-13 09:09:02
        # # No algorithms are implemented to solve equation (1 - x) + g(x)
        # # 834
        # # 2025-06-13 09:09:09
        # # ERROR:2025-06-13 09:09:09,753:Error during comparison
        # # 835
        # # 2025-06-13 09:09:09
        # # Traceback (most recent call last):


    elif ability.upper() == 'CODE':

        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Check for <think>...</think> tags
            think_reward = 0.1 if re.search(r"<think>.*?</think>", solution_str, re.DOTALL) else 0.0

            # Check for <answer>...</answer> tags
            answer_tag_reward = 0.1 if re.search(r"<answer>.*?</answer>", solution_str, re.DOTALL) else 0.0

            # We follow mistral's approach and extract only if tags are correct
            if only_if_correct_format:
                if think_reward and answer_tag_reward:
                    # Assuming prime_code doesn't need the URL
                    # As we only sample 10 test cases, there is no need to do continuous, it just doubles the time. Though, more test cases can go through with more time.
                    # Thus we set first_run_timeout to 5 and thus gain a maximum of 10s per sample per test case.
                    # stage 1: we set run timeout to 5s
                    # success, metadata = prime_code.compute_score(solution_str, ground_truth, first_run_timeout=5, continuous=False)
                    # stage 2: we increase to 10s
                    success, metadata = prime_code.compute_score(solution_str, ground_truth, first_run_timeout=10, continuous=False)
                    # res is actualy in the form of success, metadata
                    success = max(0, success - 0.2) # if correct answer, should not be more than 0.8 for correctness so that max reward is 1
                    success = success + think_reward + answer_tag_reward
                    res = (success, metadata)
                else:
                    res = think_reward + answer_tag_reward
            else:
                # less harsh, should give better signals to small models
                success, metadata = prime_code.compute_score(solution_str, ground_truth, first_run_timeout=5, continuous=False)
                # res is actualy in the form of success, metadata
                success = max(0, success - 0.2) # if correct answer, should not be more than 0.8 for correctness so that max reward is 1
                success = success + think_reward + answer_tag_reward
                res = (success, metadata)

    elif ability.upper() == 'REASONING_GYM':
        from . import reasoning_gym
        res = reasoning_gym.compute_score(solution_str, ground_truth, only_if_correct_format=only_if_correct_format)

    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
