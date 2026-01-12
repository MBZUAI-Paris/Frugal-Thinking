# Evaluation

This directory contains the SLURM file we use to run Lighteval benchmarks.

## Prerequisites

- SLURM cluster with GPUs (script defaults to 8 GPUs; edit `run_eval.sh` if your setup differs).
- Conda installed.
- A Hugging Face token available as `HF_TOKEN` (plus `OPENAI_API_KEY` if you run judge-based tasks).

## Conda environment

Create and install the Lighteval dependencies (Python 3.10+ is required):

```bash
conda create -n evals python=3.10
conda activate evals
python -m pip install -e "src/eval/lighteval[extended_tasks,vllm,math]"
```

## Authentication (.env optional)

The script loads environment variables from `.env` by default.

Example `.env`:

```bash
HF_TOKEN=hf_...
# OPENAI_API_KEY=sk_...   # optional
```

You can also point to a different env file:

```bash
ENV_FILE=/path/to/.env sbatch run_eval.sh -b gpqa -m "MBZUAI-Paris/Frugal-Thinking-4B"
```

## Run evaluations

Run from the Lighteval directory so relative paths resolve correctly:

```bash
cd src/eval/lighteval
sbatch run_eval.sh -b gpqa -m "MBZUAI-Paris/Frugal-Thinking-4B"
sbatch run_eval.sh -b aime25 -m "MBZUAI-Paris/Frugal-Thinking-4B" -c 32768
```

## Script arguments

- `-b <benchmark>`: Required. Benchmark name to run (see list below).
- `-m <model>`: Required. Hugging Face model ID or local model path. Repeat `-m` for multiple models.
- `-c <context_len>`: Optional. Context length / max new tokens used for generation. Default is `32768`.
- `-h`: Show help and exit.

## Benchmarks and datasets

Each benchmark below includes a brief definition plus the dataset path used by the task config.

- `aime24`: AIME 2024 competition math problems. HF dataset: `HuggingFaceH4/aime_2024`.
- `aime25`: AIME 2025 competition math problems. HF dataset: `yentinglin/aime_2025`.
- `omni_hard`: Omni-MATH-Hard (IQ-style, multi-step math reasoning). HF dataset: `MBZUAI-Paris/Omni-MATH-Hard`.
- `math_500`: 500-problem MATH subset. HF dataset: `HuggingFaceH4/MATH-500`.
- `gsm8k`: Grade-school math word problems. HF dataset: `openai/gsm8k`.
- `gsm_plus`: GSM8K with prompt variations. HF dataset: `qintongli/GSM-Plus`.
- `gpqa`: GPQA Diamond (graduate-level multiple choice QA). HF dataset: `Idavidrein/gpqa` (subset `gpqa_diamond`).
- `ifeval`: Instruction-following evaluation. HF dataset: `google/IFEval`.
- `mmlu`: Massive Multitask Language Understanding (57 subjects). HF dataset: `cais/mmlu`.
- `hle`: Humanity's Last Exam (judge-based). HF dataset: `cais/hle`.
- `lcb`: LiveCodeBench v5 code-generation. HF dataset: `livecodebench/code_generation_lite` (subset `v5`).
- `lcb_qwen`: LCB v6 slice covering 25.02 to 25.05. HF dataset: `MBZUAI-Paris/codegen-lite-feb-may-2025`.


## Outputs

- Logs: `src/eval/lighteval/logs/unified_eval/<benchmark>/`
- Results: `src/eval/lighteval/res_all/results_<context_len>/<benchmark>_<model>/`
