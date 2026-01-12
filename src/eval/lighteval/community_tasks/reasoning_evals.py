# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Task to evaluate LLMs on AIME 2024, GPQA, IFEval, LiveCodeBench, and IQ Test problems.
"""

# Core imports
import json
import logging
import math
import random
import re
from typing import Any, Dict, List, Optional, Union, Literal
from lighteval.models.model_output import ModelResponse
# Third party imports
import numpy as np
from aenum import extend_enum
from pydantic import BaseModel

# Lighteval imports
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
)

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import PassAtK, JudgeLLM

from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    CorpusLevelMetricGrouping,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
    SamplingMethod,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import lighteval.tasks.extended.ifeval.instructions_registry as instructions_registry
from lighteval.tasks.extended.lcb.codegen_metrics import (
    codegen_metrics,
    extract_code,
    translate_private_test_cases,
)
from lighteval.utils.language import Language
from lighteval.utils.utils import remove_reasoning_tags

# ---------------------------------------------------------------------------
# METRIC DEFINITIONS
# ---------------------------------------------------------------------------

## AIME Metrics
# Metric for mathematical expression matching. Matches boxed expressions first.
expr_gold_metric = MultilingualExtractiveMatchMetric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gold_metric_2 = MultilingualExtractiveMatchMetric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=-1, try_extract_without_anchor=True),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
    latex_gold=True,
)

# AIME Pass@K metrics for multiple generations
aime25_pass_at_1_1n = SampleLevelMetric(
    metric_name="aime25_pass@k",
    sample_level_fn=PassAtK(
        sample_scoring_function=expr_gold_metric,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

deepmath_pass_at_1_1n = SampleLevelMetric(
    metric_name="aime25_pass@1:1_samples",
    sample_level_fn=PassAtK(
        strip_strings=True,
        sample_scoring_function=expr_gold_metric_2,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


## GPQA Metrics
# Metric for extracting letter answers (A, B, C, D)
gpqa_instruct_metric = MultilingualExtractiveMatchMetric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters",)],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters",)],
    precision=6,
)

# GPQA Pass@K metrics
gpqa_instruct_pass_at_1_1n = SampleLevelMetric(
    metric_name="gpqa_pass@1:1_samples",
    sample_level_fn=PassAtK(sample_scoring_function=gpqa_instruct_metric),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

## IFEval Metrics
REASONING_TAG_PAIRS = [("<think>", "</think>")]
submetric_names = ["prompt_level_strict_acc", "inst_level_strict_acc", "prompt_level_loose_acc", "inst_level_loose_acc"]

def ifeval_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """Checks if model outputs follow a set of instructions (strict and loose matching)."""
    response = predictions[0]
    response = remove_reasoning_tags(response, REASONING_TAG_PAIRS)

    instruction_list = formatted_doc.specific["instructions_id_list"]
    all_kwargs = formatted_doc.specific["kwargs"]
    prompt = formatted_doc.query

    # Create various response variations for loose matching
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response.replace("*", "").join(r[1:]).strip(),
        revised_response.replace("*", "").join(r[:-1]).strip(),
        revised_response.replace("*", "").join(r[1:-1]).strip(),
    ]

    is_following_list_strict, is_following_list_loose = [], []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
        instruction.build_description(**task_kwargs)
        if "prompt" in instruction.get_instruction_args():
            instruction.build_description(prompt=prompt)

        # Strict matching
        is_following_list_strict.append(bool(response.strip() and instruction.check_following(response)))

        # Loose matching
        is_following_loose = any(r.strip() and instruction.check_following(r) for r in all_responses)
        is_following_list_loose.append(is_following_loose)

    return {
        "prompt_level_strict_acc": int(all(is_following_list_strict)),
        "inst_level_strict_acc": is_following_list_strict,
        "prompt_level_loose_acc": int(all(is_following_list_loose)),
        "inst_level_loose_acc": is_following_list_loose,
    }

def agg_inst_level_acc(items):
    """Aggregates instruction-level accuracy scores."""
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items) if flat_items else 0.0

ifeval_metrics = SampleLevelMetricGrouping(
    metric_name=submetric_names,
    higher_is_better=dict.fromkeys(submetric_names, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=ifeval_metric,
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)

## LiveCodeBench Metric
def codegen_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    """Estimates Pass@1 for code generation by executing generated code against test cases."""
    generated_code_snippets = [[extract_code(pred) for pred in predictions]]
    evaluation_sample = {
        "inputs": formatted_doc.specific["inputs"],
        "outputs": formatted_doc.specific["outputs"],
        "fn_name": formatted_doc.specific["fn_name"],
    }
    evaluation_sample_json = [{"input_output": json.dumps(evaluation_sample)}]

    metrics, _ = codegen_metrics(
        evaluation_sample_json,
        generated_code_snippets,
        k_list=[1],
        num_process_evaluate=8,
    )
    return metrics["pass@1"]


## IQ Test Metric (LLM-as-Judge)
class IQTestAnswer(BaseModel):
    """Pydantic model for the judge's structured response."""
    GRADE: Literal["CORRECT", "INCORRECT"]
    reasoning: str
    strict: Literal[True] = True

def get_iq_judge_prompt(question: str, answer: str, gold: str, **kwargs):
    """Creates the judge prompt for evaluating IQ test answers."""
    return [
        {"role": "system", "content": "You are a teacher grading a quiz and must output JSON."},
        {"role": "user", "content": f"""You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer. Score the student's answer as CORRECT or INCORRECT.
Focus on whether the student's answer captures the essence of the correct answer, even if the wording is different.

QUESTION: {question}
STUDENT ANSWER: {answer}
TRUE ANSWER: {gold}

Provide your evaluation in the following JSON format:
{{
    "GRADE": "CORRECT" or "INCORRECT",
    "reasoning": "A brief explanation of your grade."
}}"""}
    ]

def process_iq_judge_response(response: IQTestAnswer | List[IQTestAnswer]):
    """Processes the judge's response, converting grade to a numerical score."""
    if isinstance(response, list):
        response = response[0]
    score = 1 if response.GRADE == "CORRECT" else 0
    return {"score": score, "grade": response.GRADE, "reasoning": response.reasoning}

class JudgeLLMIQTest(JudgeLLM):
    """Custom JudgeLLM implementation for the IQ Test task."""
    def __init__(self):
        super().__init__(
            judge_model_name="gpt-4o-2024-08-06",
            template=get_iq_judge_prompt,
            process_judge_response=process_iq_judge_response,
            judge_backend="openai",
            short_judge_name="omni_hard_judge",
            response_format=IQTestAnswer,
            max_tokens=102000,
        )

    def compute(self, responses: list, docs: list[Doc], **kwargs) -> list[dict[str, float]]:
    # def compute(self, model_response: list[ModelResponse], docs: list[Doc], **kwargs):
        """Compute metrics for each sample."""
        questions = [doc.query for doc in docs]
        golds = [doc.get_golds()[0] for doc in docs]
        predictions = [resp[0].text[0] for resp in responses]
        options = [None] * len(questions)

        scores, _, _ = self.judge.evaluate_answer_batch(questions, predictions, options, golds)
        return [{"accuracy": score["score"]} for score in scores]

    def compute_corpus(self, scores: List[dict]):
        """Compute corpus-level metrics."""
        n = len(scores)
        if n == 0:
            return {"accuracy": 0.0}
        print(scores)
        try:
            correct_sum = sum(s.get("accuracy", 0.0) for s in scores)
        except Exception as e:
            logging.error(f"Error computing corpus accuracy: {e}")
            correct_sum = sum(s for s in scores)
        return {"accuracy": round(100 * correct_sum / n, 2)}

# Define the IQ test metric using SampleLevelMetricGrouping
omni_hard_metric = SampleLevelMetricGrouping(
    metric_name=["accuracy"],
    higher_is_better={"accuracy": True},
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeLLMIQTest(),
    corpus_level_fn={"accuracy": np.mean},
    batched_compute=True,
)

# Register the custom metric
# extend_enum(Metrics, "omni_hard_metric_new", omni_hard_metric)

# ---------------------------------------------------------------------------
# PROMPT FORMATTING FUNCTIONS
# ---------------------------------------------------------------------------

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()
# MATH_QUERY_TEMPLATE = """
# You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Let's think step by step, and write the final answer inside \\boxed{{}} within the <answer> </answer> tags.

# {Question}
# """.strip()

def aime_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Formats AIME problems for evaluation."""
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )

def deepmath_prompt_fn(line: dict, task_name: str = None) -> Doc:
    """Formats AIME problems for evaluation."""
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["final_answer"]],
        gold_index=0,
    )

def gsm_plus_prompt(line, task_name: str = None) -> Doc:
    # GSM8K with 8 prompt variations per sample

    # Some prompts require critical thinking (around 1k/10k), we skip them as
    # they are a bit trickier to eval with regular text extraction.
    if line["perturbation_type"] == "critical thinking":
        return None

    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n\nAnswer:",
        choices=[line["answer"]],
        gold_index=0,
    )

LETTER_INDICES = ["A", "B", "C", "D"]

def gpqa_instruct(line: dict, task_name: str = None) -> Doc:
    """Formats GPQA multiple-choice questions. Shuffles choices randomly."""
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query_template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[:len(choices)],
        gold_index=gold_index,
        instruction=query,
    )

def ifeval_prompt(line: dict, task_name: str = None) -> Doc:
    """Formats IFEval problems, which test instruction following."""
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )

def prepare_prompt(line: dict[str, Any]) -> str:
   """Prepares the prompt for LiveCodeBench tasks."""
   query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
   query += f"Question: {line['question_content']}\n\n"
   if starter_code := line.get("starter_code"):
       query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
       query += f"```python\n{starter_code}\n```\n\n"
   else:
       query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
       query += "```python\n# YOUR CODE HERE\n```\n\n"
   return query

def lcb_codegeneration_prompt_fn(line: dict, task_name: str = "lcb:codegeneration") -> Doc:
    """Formats LiveCodeBench problems, preparing inputs, outputs, and function names."""
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    return Doc(
        task_name=task_name,
        query=prepare_prompt(line),
        choices=[""],
        gold_index=0,
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(line["metadata"]).get("func_name"),
        },
    )

def omni_hard_prompt(line: dict, task_name: str = None) -> Doc:
    """Formats IQ Test problems, extracting the question and answer."""
    import ast
    query = ""
    # Debug: Print the raw line to understand the structure
    print(f"DEBUG: Processing line with keys: {list(line.keys())}")
    print(f"DEBUG: messages type: {type(line.get('messages'))}")
    print(f"DEBUG: messages content (first 100 chars): {str(line.get('messages', ''))[:100]}...")
    if "messages" in line:
        messages = line["messages"]
        # Handle both string and list formats
        if isinstance(messages, str):
            try:
                # Parse the string as Python literal (handles single quotes)
                print("DEBUG: Parsing messages from string format with ast.literal_eval")
                messages = ast.literal_eval(messages)
            except (ValueError, SyntaxError) as e:
                print(f"DEBUG: Failed to parse messages with ast.literal_eval: {e}")
                logging.warning(f"Failed to parse messages in line {line.get('id', 'unknown')}: {e}")
                messages = []
        elif isinstance(messages, list):
            print("DEBUG: Messages already in list format")
        else:
            print(f"DEBUG: Unexpected messages type: {type(messages)}")
            messages = []
        # Extract the user query
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    print(f"DEBUG: Found user query: {query[:100]}...")
                    break
    # Get the answer
    answer = line.get("o4mini_high_response", "")
    print(f"DEBUG: Answer length: {len(answer)} chars")
    # Enhanced logging for missing data
    if not query: 
        print("DEBUG: WARNING - No query found!")
        logging.warning(f"Missing query in line: {line}")
    if not answer: 
        print("DEBUG: WARNING - No answer found!")
        logging.warning(f"Missing answer in line: {line}")
 
    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        specific={"question": query},
    )

# ---------------------------------------------------------------------------
# TASK CONFIGURATIONS
# ---------------------------------------------------------------------------

# AIME 2024 Task
aime24_task = LightevalTaskConfig(
    name="aime24",
    suite=["community"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,  
    metrics=[aime25_pass_at_1_1n(sample_params={"k": 1, "n": 1})],
    
    version=1,
)

# AIME 2025 Task
aime25_task = LightevalTaskConfig(
    name="aime25",
    suite=["community"],
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,
    metrics=[aime25_pass_at_1_1n(sample_params={"k": 1, "n": 1})],
    version=1,
)

deepmath5 = LightevalTaskConfig(
    name="deepmath5",
    suite=["community"],
    prompt_function=deepmath_prompt_fn,
    hf_repo="MBZUAI-Paris/DeepMath-103K-filtered-level-5-6",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,
    metrics=[deepmath_pass_at_1_1n(sample_params={"k": 1, "n":1})],
    
    version=1,
)

deepmath6 = LightevalTaskConfig(
    name="deepmath6",
    suite=["community"],
    prompt_function=deepmath_prompt_fn,
    hf_repo="MBZUAI-Paris/DeepMath-103K-filtered-level-6-7",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,
    metrics=[deepmath_pass_at_1_1n(sample_params={"k": 1, "n":1})],
    
    version=1,
)

deepmath7 = LightevalTaskConfig(
    name="deepmath7",
    suite=["community"],
    prompt_function=deepmath_prompt_fn,
    hf_repo="MBZUAI-Paris/DeepMath-103K-filtered-level-7-8",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,
    metrics=[deepmath_pass_at_1_1n(sample_params={"k": 1, "n":1})],
    
    version=1,
)

deepmath8 = LightevalTaskConfig(
    name="deepmath8",
    suite=["community"],
    prompt_function=deepmath_prompt_fn,
    hf_repo="MBZUAI-Paris/DeepMath-103K-filtered-level-8-9",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,
    metrics=[deepmath_pass_at_1_1n(sample_params={"k": 1, "n":1})],
    
    version=1,
)

deepmath9 = LightevalTaskConfig(
    name="deepmath9",
    suite=["community"],
    prompt_function=deepmath_prompt_fn,
    hf_repo="MBZUAI-Paris/DeepMath-103K-filtered-level-9-10",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,
    metrics=[deepmath_pass_at_1_1n(sample_params={"k": 1, "n": 1})],
    
    version=1,
)

# GPQA Diamond Task
gpqa_diamond_instruct_lighteval = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["community"],
    prompt_function=gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=None,  # Needed for large reasoning models
    metrics=[
        gpqa_instruct_pass_at_1_1n(sample_params={"k": 1, "n":1}),
    ],
    stop_sequence=[],
    version=1,
)

# IFEval Task
ifeval = LightevalTaskConfig(
    name="ifeval",
    prompt_function=ifeval_prompt,
    suite=["extended"],
    hf_repo="google/IFEval",
    hf_subset="default",
    metrics=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=None,
    stop_sequence=[],
    version="0.1",
)

# LiveCodeBench Tasks
lcb_configs = [
    "release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6",
    "release_latest", "v1", "v2", "v3", "v4", "v5", "v6", "v1_v2", "v1_v3", "v1_v4",
    "v1_v5", "v2_v3", "v2_v4", "v2_v5", "v3_v4", "v3_v5", "v4_v5",
]

gsm_plus = LightevalTaskConfig(
    name="gsm_plus",
    suite=["community"],
    prompt_function=gsm_plus_prompt,
    hf_repo="qintongli/GSM-Plus",
    hf_subset="default",
    hf_avail_splits=["test", "testmini"],
    evaluation_splits=["testmini"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=None,
    version=0,
)

lcb_tasks = []
for subset in lcb_configs:
    name = "lcb:codegeneration" if subset == "v4_v5" else f"lcb:codegeneration_{subset}"
    task = LightevalTaskConfig(
        name=name,
        suite=["extended"],
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="livecodebench/code_generation_lite",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=None,
        metrics=[Metrics.lcb_codegen_metric],
        stop_sequence=[],
        version=0,
    )
    lcb_tasks.append(task)

name = "lcb:codegeneration_qwen"
lcb_qwen_task = LightevalTaskConfig(
        name=name,
        suite=["extended"],
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="MBZUAI-Paris/codegen-lite-feb-may-2025",
        hf_subset='default',
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=None,
        metrics=[Metrics.lcb_codegen_metric(sample_params={"k": 1, "n": 16})],
        stop_sequence=[],  # no stop sequence, will use EOS token
        version=0,
    )

# IQ Test Task
omni_hard_task = LightevalTaskConfig(
    name="omni_hard",
    suite=["community"],
    prompt_function=omni_hard_prompt,
    hf_repo="MBZUAI-Paris/Omni-MATH-Hard",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    generation_size=None,
    metrics=[omni_hard_metric],
    stop_sequence=[],
    version=0,
)

# ---------------------------------------------------------------------------
# FINAL TASK TABLE
# ---------------------------------------------------------------------------
TASKS_TABLE = [
    aime24_task,
    aime25_task,
    gpqa_diamond_instruct_lighteval,
    ifeval,
    omni_hard_task,
    lcb_qwen_task,
    gsm_plus,
    deepmath5,
    deepmath6,
    deepmath7,
    deepmath8,
    deepmath9
] + lcb_tasks