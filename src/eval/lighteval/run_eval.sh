#!/bin/bash
#SBATCH --job-name=unified_lighteval # Job name will be updated by the script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --partition=main
#SBATCH --exclusive
#SBATCH --output=logs/unified_eval/%x-%j.out # Log paths are handled dynamically
#SBATCH --error=logs/unified_eval/%x-%j.err


#-----------------------------------------------------------------------------#
#                Unified Lighteval Benchmark Runner Script                    #
#-----------------------------------------------------------------------------#

# Get master node information
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export TORCHDYNAMO_VERBOSE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 

# --- Help Function ---
display_help() {
  echo "Usage: sbatch $0 -b <benchmark> -m <model1> [-m <model2> ...]"
  echo
  echo "This script runs lighteval for a specified benchmark and model(s)."
  echo
  echo "Arguments:"
  echo "  -b <benchmark>    (Required) The benchmark to run. Available options:"
  echo "                    aime24, aime25, lcb, gpqa, ifeval, omni_hard"
  echo "  -m <model>        (Required) The Hugging Face model identifier. Can be specified multiple times."
  echo "  -c <context_len>  (Optional) Context length for evaluation (default: 32768)."
  echo "  -h                Display this help message and exit."
  echo
  echo "Examples:"
  echo "  sbatch $0 -b gpqa -m 'HuggingFaceTB/SmolLM3-3B'"
  echo "  sbatch $0 -b aime25 -m 'model_one' -m 'model_two'"
}

# --- Argument Parsing ---
MODELS=()
BENCHMARK=""
CTX_LEN="32768"  # Default context length

while getopts ":hb:m:c:" opt; do
  case ${opt} in
    h)
      display_help
      exit 0
      ;;
    b)
      BENCHMARK=$OPTARG
      ;;
    m)
      MODELS+=("$OPTARG")
      ;;
    c)
      CTX_LEN=$OPTARG
      ;;
    \?)
      echo "Invalid Option: -$OPTARG" 1>&2
      display_help
      exit 1
      ;;
    :)
      echo "Invalid Option: -$OPTARG requires an argument." 1>&2
      display_help
      exit 1
      ;;
  esac
done

# --- Validate Arguments ---
if [ -z "$BENCHMARK" ] || [ ${#MODELS[@]} -eq 0 ]; then
  echo "Error: Benchmark and at least one model must be specified."
  display_help
  exit 1
fi

# --- Dynamic SLURM Configuration ---
JOB_NAME="${BENCHMARK}_eval"
scontrol update job "$SLURM_JOB_ID" JobName="$JOB_NAME"

LOG_DIR="logs/unified_eval/${BENCHMARK}"
mkdir -p "$LOG_DIR"
echo "Logs will be available in ${LOG_DIR}/"

# --- Environment Setup ---
ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a            # export all variables loaded below
  # shellcheck disable=SC1090
  source "$ENV_FILE"  # loads KEY=VALUE lines
  set +a
fi
echo "Activating Conda environment and setting up paths..."
eval "$(conda shell.bash hook)"
conda activate evals
# cd ~/lighteval || { echo "Failed to cd to ~/lighteval"; exit 1; }

# --- Authentication ---
# The script will now use HF_TOKEN and OPENAI_API_KEY from your environment variables.
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    exit 1
fi
echo "Logging into Hugging Face CLI using environment variable..."
huggingface-cli login --token "$HF_TOKEN"

# --- Main Logic ---
echo "Starting evaluation for benchmark: $BENCHMARK"

case "$BENCHMARK" in
  "aime25" | "omni_hard" | "aime24" | "math_500" | "gsm8k" | "deepmath5" | "deepmath6" | "deepmath7" | "deepmath8" | "deepmath9" | "gsm_plus" | "gpqa" | "ifeval" | "mmlu" | "hle" | "lcb" | "lcb_qwen")
    TASK_STRING=""
    DTYPE="bfloat16"
    if [ "$BENCHMARK" == "aime25" ]; then
      TASK_STRING="community|aime25|0"
    elif [ "$BENCHMARK" == "aime24" ]; then
      TASK_STRING="community|aime24|0"
    elif [ "$BENCHMARK" == "omni_hard" ]; then
      TASK_STRING="community|omni_hard|0|0"
    elif [ "$BENCHMARK" == "math_500" ]; then
      TASK_STRING="lighteval|math_500|0|0"
    elif [ "$BENCHMARK" == "gsm8k" ]; then
      TASK_STRING="lighteval|gsm8k|0|0"
    elif [ "$BENCHMARK" == "gsm_plus" ]; then
      TASK_STRING="lighteval|gsm_plus|0|0"
    elif [[ "$BENCHMARK" == "deepmath"* ]]; then
      TASK_STRING="community|$BENCHMARK|0|0"
    elif [ "$BENCHMARK" == "gpqa" ]; then
      TASK_STRING="community|gpqa:diamond|0|0"
    elif [ "$BENCHMARK" == "ifeval" ]; then
      TASK_STRING="extended|ifeval|0|0"
    elif [ "$BENCHMARK" == "mmlu" ]; then
      TASK_STRING="original|mmlu:anatomy|0,original|mmlu:abstract_algebra|0,original|mmlu:astronomy|0,original|mmlu:business_ethics|0,original|mmlu:clinical_knowledge|0,original|mmlu:college_biology|0,original|mmlu:college_chemistry|0,original|mmlu:college_computer_science|0,original|mmlu:college_mathematics|0,original|mmlu:college_medicine|0,original|mmlu:college_physics|0,original|mmlu:computer_security|0,original|mmlu:conceptual_physics|0,original|mmlu:econometrics|0,original|mmlu:electrical_engineering|0,original|mmlu:elementary_mathematics|0,original|mmlu:formal_logic|0,original|mmlu:global_facts|0,original|mmlu:high_school_biology|0,original|mmlu:high_school_chemistry|0,original|mmlu:high_school_computer_science|0,original|mmlu:high_school_european_history|0,original|mmlu:high_school_geography|0,original|mmlu:high_school_government_and_politics|0,original|mmlu:high_school_macroeconomics|0,original|mmlu:high_school_mathematics|0,original|mmlu:high_school_microeconomics|0,original|mmlu:high_school_physics|0,original|mmlu:high_school_psychology|0,original|mmlu:high_school statistics|0,original|mmlu:high_school_us_history|0,original|mmlu:high_school_world_history|0,original|mmlu:human_aging|0,original|mmlu:human_sexuality|0,original|mmlu:international_law|0,original|mmlu:jurisprudence|0,original|mmlu:logical_fallacies|0,original|mmlu:machine_learning|0,original|mmlu:management|0,original|mmlu:marketing|0,original|mmlu:medical_genetics|0,original|mmlu:miscellaneous|0,original|mmlu:moral_disputes|0,original|mmlu:moral_scenarios|0,original|mmlu:nutrition|0,original|mmlu:philosophy|0,original|mmlu:prehistory|0,original|mmlu:professional_accounting|0,original|mmlu:professional_law|0,original|mmlu:professional_medicine|0,original|mmlu:professional_psychology|0,original|mmlu:public_relations|0,original|mmlu:security_studies|0,original|mmlu:sociology|0,original|mmlu:us_foreign_policy|0,original|mmlu:virology|0,original|mmlu:world_religions|0"
    elif [ "$BENCHMARK" == "hle" ]; then
      TASK_STRING="extended|hle|0"
    elif [ "$BENCHMARK" == "lcb" ]; then
      TASK_STRING="extended|lcb:codegeneration_v5|0|0"
    elif [ "$BENCHMARK" == "lcb_qwen" ]; then
      TASK_STRING="extended|lcb:codegeneration_qwen|0"
    else
      echo "Error: Unknown benchmark '$BENCHMARK'"
    fi
    echo "Using context length: $CTX_LEN"
    echo "Using TASK_STRING: $TASK_STRING"
    echo "Configuring for $BENCHMARK (vllm backend, serial execution)"
    for MODEL in "${MODELS[@]}"; do
      if [[ "$MODEL" == *"Qwen/"* ]]; then
        generation_parameters="{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"max_new_tokens\":\"${CTX_LEN}\"}"
        echo "Using Qwen generation parameters: $generation_parameters"
        model_args="model_name=${MODEL},dtype=${DTYPE},seed=1234,max_model_length=102000,trust_remote_code=true,tensor_parallel_size=8,generation_parameters=${generation_parameters}"
      elif [[ "$MODEL" == "tiiuae/Falcon-H1R-7B" ]]; then
        export VLLM_ATTENTION_BACKEND=FLASHINFER
        generation_parameters="{\"temperature\":0.6,\"top_p\":0.95,\"max_new_tokens\":\"${CTX_LEN}\"}"
        echo "Using falcon generation parameters: $generation_parameters"
        model_args="model_name=${MODEL},dtype=${DTYPE},seed=1234,max_model_length=102000,trust_remote_code=true,tensor_parallel_size=1,enable_prefix_caching=false,generation_parameters=${generation_parameters}"
      elif [[ "$MODEL" == "LM360/K2-Think" ]]; then
        generation_parameters="{\"temperature\":1.0,\"top_p\":0.95,\"max_new_tokens\":\"${CTX_LEN}\"}"
        echo "Using LM360/K2-Think generation parameters: $generation_parameters"
        model_args="model_name=${MODEL},dtype=${DTYPE},max_model_length=102000,trust_remote_code=true,tensor_parallel_size=8,generation_parameters=${generation_parameters}"
      elif [[ "$MODEL" == *"mistralai/Magistral-Small-2507"* ]]; then
        system_prompt='First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.'
        generation_parameters="{\"temperature\":0.7,\"top_p\":0.95,\"max_new_tokens\":\"${CTX_LEN}\"}"
        echo "Using Mistral generation parameters: $generation_parameters"  
        model_args="model_name=${MODEL},dtype=${DTYPE},seed=1234,max_model_length=38000,trust_remote_code=true,tensor_parallel_size=8,generation_parameters=${generation_parameters},tokenizer_mode=mistral,config_format=mistral,load_format=mistral,system_prompt=\"${system_prompt}\""
      elif [[ "$MODEL" == *"mistralai/"* ]]; then
        system_prompt='First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.'
        generation_parameters="{\"temperature\":0.7,\"top_p\":0.95,\"max_new_tokens\":\"${CTX_LEN}\"}"
        echo "Using Mistral generation parameters: $generation_parameters"  
        model_args="model_name=${MODEL},dtype=${DTYPE},seed=1234,max_model_length=102000,trust_remote_code=true,tensor_parallel_size=8,generation_parameters=${generation_parameters},tokenizer_mode=mistral,config_format=mistral,load_format=mistral,system_prompt=\"${system_prompt}\""
      elif [[ "$MODEL" == "HuggingFaceTB/SmolLM3-3B" ]]; then
        generation_parameters="{\"temperature\":0.6,\"top_p\":0.95,\"max_new_tokens\":\"${CTX_LEN}\"}"
        echo "Using LM360 generation parameters: $generation_parameters"
        model_args="model_name=${MODEL},dtype=${DTYPE},seed=1234,max_model_length=65000,trust_remote_code=true,tensor_parallel_size=1,generation_parameters=${generation_parameters}"
      else
        generation_parameters="{\"temperature\":1.0,\"max_new_tokens\":\"${CTX_LEN}\"}"
        echo "Using default generation parameters: $generation_parameters"
        model_args="model_name=${MODEL},dtype=${DTYPE},seed=1234,max_model_length=102000,trust_remote_code=true,tensor_parallel_size=8,generation_parameters=${generation_parameters}"
      fi
      SAFE_NAME="${MODEL//\//_}"
      if [[ "$MODEL" == "/mnt/sharefs/users/paris/k2/rl/checkpoints/4b/math/32k/stage-1/4-nodes-n-16stage-1-prime-32k-k20-dsfalse-ratio0.15-epshigh-0.28AM-Qwen3-4B-Thinking-2507-RL/global_step_1050/actor/huggingface" ]]; then
        SAFE_NAME="MBZUAI-Paris_Frugal-Math-4B-stage1"
      fi
      lighteval vllm \
          "${model_args}" \
          "$TASK_STRING" \
          --custom-tasks community_tasks/reasoning_evals.py \
          --save-details \
          --output-dir "./res_all/results_${CTX_LEN}/${BENCHMARK}_${SAFE_NAME}"
    done
    ;;

  *)
    echo "Error: Unknown benchmark '$BENCHMARK'"
    display_help
    exit 1
    ;;
esac

echo "Evaluation for benchmark '$BENCHMARK' complete!"