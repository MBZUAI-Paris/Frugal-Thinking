# Copyright 2024 PRIME team and/or its affiliates
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

import json
import traceback

# NOTE: When this file is executed as a script (e.g. `python __init__.py`)
# there is no parent package, so relative imports fail.  We therefore
# attempt the relative import first and, if it fails, fall back to a
# local absolute‑-ish import that works in script mode.
try:
    from .utils import check_correctness as apps_check_correctness
    from .utils import check_correctness_assert as apps_check_correctness_assert
except ImportError:  # running as a top‑level script
    import importlib
    utils = importlib.import_module("utils")  # utils lives in the same dir
    apps_check_correctness = utils.check_correctness
    apps_check_correctness_assert = utils.check_correctness_assert


def compute_score(
        completion,
        test_cases,
        continuous=False,
        first_run_timeout: int = 10,
        pass_vals = (1, True, "yes", "correct"), 
    ):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    # print(f'[INFO-prime_code.__init__.py] continuous: {continuous}')
    # print(f'[INFO-prime_code.__init__.py] completion (solution before split): {completion}')
    solution = completion.split("```python")[-1].split("```")[0]

    # print(f'[INFO-prime_code.__init__.py] solution: {solution}')
    # print(f'[INFO-prime_code.__init__.py] test_cases: {test_cases}')
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"[ERROR-prime_code.__init__.py] Failed to parse test_cases JSON: {e}")
            print(f"[DEBUG-prime_code.__init__.py] test_cases raw value: {test_cases}")
            raise e
        
        # Validate test_cases structure
        if not isinstance(test_cases, dict):
            raise ValueError(f"test_cases must be a dictionary, got {type(test_cases)}")

        # Determine call_type early so validation can be tailored
        call_type = test_cases.get("call_type", "std")

        if call_type == "std":
            if "inputs" not in test_cases or "outputs" not in test_cases:
                raise ValueError(
                    f"test_cases for 'std' must contain 'inputs' and 'outputs' keys, got keys: {list(test_cases.keys())}"
                )
            inputs = test_cases["inputs"]
            outputs = test_cases["outputs"]
            if not isinstance(inputs, list) or not isinstance(outputs, list):
                raise ValueError(
                    f"'inputs' and 'outputs' must be lists, got inputs: {type(inputs)}, outputs: {type(outputs)}"
                )
            if len(inputs) != len(outputs):
                raise ValueError(
                    f"'inputs' and 'outputs' must have same length, got inputs: {len(inputs)}, outputs: {len(outputs)}"
                )
        elif call_type == "assert":
            if "assert_case" not in test_cases:
                raise ValueError(
                    f"test_cases for 'assert' must contain 'assert_case' key, got keys: {list(test_cases.keys())}"
                )
            if not isinstance(test_cases["assert_case"], list):
                raise ValueError(
                    f"'assert_case' must be a list, got {type(test_cases['assert_case'])}"
                )
        else:
            raise ValueError(f"Unsupported call_type. Expected 'std' or 'assert'. Got {call_type}.")
        
        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:

            if call_type == "std":
                res, metadata = apps_check_correctness(
                    in_outs=test_cases, 
                    generation=solution, 
                    timeout=first_run_timeout, 
                    debug=False,
                )
            elif call_type == "assert":
                res, metadata = apps_check_correctness_assert(
                    assertions=test_cases.get("assert_case", []), 
                    generation=solution, 
                    timeout=first_run_timeout, 
                    debug=False,
                )
            else:
                raise ValueError(f"Unsupported call_type. Expected std or assert. Got {call_type}.")
            
            try:
                metadata_result = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
            except Exception:
                metadata_result = {}
            success = all(x in pass_vals for x in res)
            if success:
                return success, metadata_result
        except Exception as e:
            print(f'[DEBUG-prime_code.__init__.py] Exception in initial check: {e}')
            traceback.print_exc()
        
        # we do an extra pass with more timeout to see if more can go through
        if call_type == "std":

            test_cases_list = []
            inputs = test_cases["inputs"]
            outputs = test_cases["outputs"]

            # Validate inputs and outputs are lists and have same length
            if not isinstance(inputs, list) or not isinstance(outputs, list):
                raise ValueError(f"inputs and outputs must be lists, got inputs: {type(inputs)}, outputs: {type(outputs)}")
            
            if len(inputs) != len(outputs):
                raise ValueError(f"inputs and outputs must have same length, got inputs: {len(inputs)}, outputs: {len(outputs)}")
            
            for i in range(len(inputs)):
                test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

            if continuous:
                # per sample test: if continuous score is needed, test first 10 samples regardless of failures
                # do not test all samples cuz some problems have enormous test cases
                metadata_list = []
                res_list = []
                for test_case_id, test_case in enumerate(test_cases_list):
                    res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)
                    try:
                        metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                    except Exception:
                        metadata = {}
                    metadata["test_case"] = {}
                    metadata["test_case"]["input"] = str(test_case["inputs"][0])
                    metadata["test_case"]["output"] = str(test_case["outputs"][0])
                    metadata["test_case"]["res"] = str(res)
                    metadata_list.append(metadata)
                    res_list.extend(res)

                    if test_case_id >= 9:
                        break
                res_count = len(res_list) if len(res_list) > 0 else 1

                # success = sum(map(lambda x: x is True, res_list)) / res_count
                success = sum(x in pass_vals for x in res_list) / res_count

                metadata_result = metadata_list
            else:
                # For non-continuous mode, we still need to try the extra pass but don't need per-sample metadata
                try:
                    res, metadata = apps_check_correctness(
                        in_outs=test_cases, 
                        generation=solution, 
                        timeout=10, 
                        debug=False,
                    )
                    try:
                        metadata_result = dict(enumerate(metadata))[0]
                    except Exception:
                        metadata_result = {}
                    res_count = len(res) if len(res) > 0 else 1
                    success = sum(x in pass_vals for x in res) / res_count
                except Exception:
                    success = False
                    metadata_result = {}

        elif call_type == "assert":

            assert_cases = test_cases["assert_case"]

            # When continuous scoring is requested, evaluate each assertion separately (up to the first 10)
            if continuous:
                metadata_list = []
                res_list = []
                for test_case_id, assertion in enumerate(assert_cases):
                    res, metadata = apps_check_correctness_assert(
                        assertions=[assertion],
                        generation=solution,
                        timeout=5,
                        debug=False,
                    )
                    try:
                        metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                    except Exception:
                        metadata = {}

                    metadata["test_case"] = {
                        "assertion": assertion,
                        "res": str(res),
                    }
                    metadata_list.append(metadata)
                    res_list.extend(res)

                    if test_case_id >= 9:  # Only test the first 10 cases
                        break

                res_count = len(res_list) if len(res_list) > 0 else 1
                success = sum(map(lambda x: x is True, res_list)) / res_count
                metadata_result = metadata_list
            else:
                # For non-continuous mode with assert, try with longer timeout
                try:
                    res, metadata = apps_check_correctness_assert(
                        assertions=assert_cases,
                        generation=solution,
                        timeout=10,
                        debug=False,
                    )
                    try:
                        metadata_result = dict(enumerate(metadata))[0]
                    except Exception:
                        metadata_result = {}
                    res_count = len(res) if len(res) > 0 else 1
                    success = sum(x in pass_vals for x in res) / res_count
                except Exception:
                    success = False
                    metadata_result = {}
        else:
            raise ValueError(f"Unsupported call_type. Expected std or assert. Got {call_type}.")
        
    except Exception:
        print(f'[DEBUG-prime_code.__init__.py] raised an issue')
        traceback.print_exc(10)
        success = False
        metadata_result = {}
    
    return success, metadata_result

# ---------------------------------------------------------------------------
# Quick smoke‑test: run `python -m verl.utils.reward_score.prime_code` to verify
# both the 'std' and 'assert' paths of `compute_score`.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint

    # Let's first test the solution extraction
    completion_std = '''
```python
def add_two_numbers(a, b):
    return a + b
```
'''
    solution = completion_std.split("```python")[-1].split("```")[0]
    print(f"Extracted solution: {repr(solution)}")
    
    # Test if the solution is valid Python code
    try:
        exec(solution)
        print("Solution compiles successfully")
        # Test the function directly
        exec(solution)
        result1 = eval("add_two_numbers(1, 2)")
        result2 = eval("add_two_numbers(3, 4)")
        print(f"Direct test: add_two_numbers(1, 2) = {result1}")
        print(f"Direct test: add_two_numbers(3, 4) = {result2}")
    except Exception as e:
        print(f"Error with solution: {e}")
        traceback.print_exc()

    # -------- std case --------
    test_cases_std = {
        "call_type": "std",
        "fn_name": "add_two_numbers",
        "inputs": [
            "1\n2",   # JSON‑loadable literals, one per line
            "3\n4",
        ],
        "outputs": [
            "3",      # expected return values as JSON literals
            "7",
        ],
    }
    print("\n--- Testing std case ---")
    ok_std, meta_std = compute_score(completion_std, test_cases_std, continuous=False)
    print("STD ->", ok_std)
    pprint(meta_std)

    # -------- assert case --------
    completion_assert = '''
```python
def add_one(x):
    return x + 1
```
'''
    test_cases_assert = {
        "call_type": "assert",
        "fn_name": "add_one",
        "assert_case": [
            "assert add_one(1) == 2",
            "assert add_one(5) == 6",
            "assert add_one(6) == 7",   # fixed expected result
        ],
    }
    print("\n--- Testing assert case ---")
    ok_assert, meta_assert = compute_score(
        completion_assert, 
        test_cases_assert, 
        continuous=False,
    )
    print("ASSERT ->", ok_assert)
    pprint(meta_assert)