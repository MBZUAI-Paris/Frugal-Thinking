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

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
import os
import sys
import traceback
import tempfile
from typing import Optional


# Local import that works both as a package and when run as a script
try:
    from .testing_util import run_test  # normal package context
except ImportError:                     # fallback for "python utils.py" -> used when just running a test for utils.py
    try:
        from testing_util import run_test
    except ImportError:
        # As a last resort, add the parent directory to sys.path and retry
        import pathlib
        parent = pathlib.Path(__file__).resolve().parent
        if str(parent) not in sys.path:
            sys.path.append(str(parent))
        from testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    # Silence child‑process output only when debug is False
    if not debug:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
    # Run inside a temporary directory so any artefacts are discarded automatically
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception:
            # print(e) # some tracebacks are extremely long.
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append({})
        finally:
            os.chdir(cwd)
            if not debug:
                devnull.close()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    # print(f'in_outs: {in_outs}')
    # print(f'generation: {generation}')
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list

# ---------------------------------------------------------------------------
# Public wrapper that chooses the right checker based on call_type
# ---------------------------------------------------------------------------


def _temp_run_assert(assert_list, generation, debug, result, metadata_list, timeout):
    """
    Execute *generation* and then run each code snippet in *assert_list*
    inside the same namespace.  Each snippet may contain auxiliary statements
    followed by one or more `assert ...` lines.

    We append 1 (pass) or -1 (fail) to *result* per snippet.
    """
    # Silence child‑process output only when debug is False
    if not debug:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
    # Run inside a temporary directory so any artefacts are discarded automatically
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            ns = {}
            exec(generation, ns)  # make candidate code available
            outcome = []
            for snippet in assert_list:
                try:
                    exec(snippet, ns)
                    outcome.append(1)
                except Exception:
                    if debug:
                        traceback.print_exc(limit=5)
                    outcome.append(-1)
            result.append(outcome)
            metadata_list.append({})
        except Exception:
            if debug:
                traceback.print_exc(limit=5)
            result.append([-1 for _ in assert_list])
            metadata_list.append({})
        finally:
            os.chdir(cwd)
            if not debug:
                devnull.close()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__


def check_correctness_assert(assertions, generation, timeout=10, debug=True):
    """
    Validate *generation* against *assertions*.

    *assertions* can be:
      • a list of code snippets (each executed in turn), or
      • a single string snippet, or
      • a dict with key "assert_case" containing the list.

    Returns (scores, metadata_list) where *scores* is 1/-1 per snippet.
    """

    # print(f'assertions: {assertions}')
    # print(f'type(assertions): {type(assertions)}')

    # Normalise to list[str]
    if isinstance(assertions, dict):
        assert_list = assertions.get("assert_case", [])
    elif isinstance(assertions, str):
        assert_list = [assertions]
    else:
        assert_list = list(assertions)
    # print(f'assertions: {assertions}')
    # print(f'type(assertions): {type(assertions)}')

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run_assert,
        args=(assert_list, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result = [[-1 for _ in assert_list]]
        if debug:
            print("global timeout (assert)")
    return result[0], metadata_list

# Smoke-test for check_correctness_assert
if __name__ == "__main__":
    candidate_code = """
def add_one(x):
    return x + 1
"""
    sample_assert_case = [
        "assert add_one(1) == 2",
        "assert add_one(5) == 6",
        "assert add_one(6) == 6",
    ]

    print("\n--- Testing assert case ---")
    scores, _ = check_correctness_assert(sample_assert_case, candidate_code, debug=False)
    print("scores:", scores)  # Expect: [1, 1, -1]


        # Let's first test the solution extraction
    candidate_code = """
def add_two_numbers(a, b):
    return a + b
"""
    test_cases_std = {
        "call_type": "std",
        "fn_name": "add_two_numbers",
        "inputs": [
            "1\n2",   # line‑1 → json.loads("1") -> 1, line‑2 -> 2  ⇢  add_two_numbers(1, 2)
            "3\n4",
        ],
        "outputs": [
            "3",      # json.loads("3") -> 3
            "7",
        ],
    }
    print("\n--- Testing std case ---")
    ok_std, meta_std = check_correctness(test_cases_std, candidate_code, debug=False)
    print("STD ->", ok_std)
    print(meta_std)