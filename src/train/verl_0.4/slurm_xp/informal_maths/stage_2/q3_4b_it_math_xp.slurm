#!/bin/bash
#SBATCH --job-name=rl-4B-math
#SBATCH --partition=main
#SBATCH --nodes=8                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=500:00:00
#SBATCH --output=logs/4B/async-math/vllm_%x_%j.out
#SBATCH --error=logs/4B/async-math/vllm_%x_%j.err

set -x
# This script runs the training of RL on multi-nodes. It does resume automatically from latest checkpoint if the run crashes.

# Shared filesystem anchors. Override via the submission environment if needed.
WEKA_HOME=${WEKA_HOME:-/mnt/weka/home/abdelaziz.bounhar}
SHAREFS_HOME=${SHAREFS_HOME:-/mnt/sharefs/users/paris}
LOG_BASE_PATH=${LOG_BASE_PATH:-${WEKA_HOME}/rl/4B/math}
DATA_ROOT=${DATA_ROOT:-${WEKA_HOME}/rl/data/math/stage_2}
CKPT_ROOT_BASE=${CKPT_ROOT_BASE:-${SHAREFS_HOME}/rl/checkpoints/4b}

# can make training faster
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1


# Determine how many GPUs we actually have on this node. SLURM sets
# SLURM_GPUS_PER_NODE only when #SBATCH --gpus-per-node is used, not with --gres.
export SLURM_GPUS_PER_NODE=8
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)} # 8
export GPUS_PER_NODE

NNODES=${SLURM_JOB_NUM_NODES}
export NNODES

# experiment params
# CURRICULUM_LEARNING=false
CURRICULUM_LEARNING=true
PROJECT_NAME="K2-Think-RL"
TRAINED_ON="math"
STAGE="stage-2"
HF_DATA_PATH=MBZUAI-Paris/DeepMath-103K-filtered-ChuGyouk-cleaned-levelled-verl
# curriculum learning
SFT_MODEL_PATH=MBZUAI-Paris/Frugal-Math-4B-Stage-1
SFT_MODEL_NAME=Qwen/Qwen3-4B-Thinking-2507
val_before_train=false #false #true
if [ "$CURRICULUM_LEARNING" = true ]; then
  SFT_MODEL_NAME="${SFT_MODEL_NAME}-curr"
  SHUFFLE=false
else
  SHUFFLE=true
fi
SFT_MODEL=$(basename "$SFT_MODEL_NAME")
ALGORITHM=grpo
ROLLOUT_mode=async #sync #async
ROLLOUT_ENGINE=vllm #vllm # sglang is not stable generates error
VLLM_GPU_MEMORY_UTILIZATION=0.8 # 0.4 sets the gpu_memory_utilization
tensor_model_parallel_size=1 #8 #4 #2
REWARD_MANAGER=prime #dapo #prime #naive #prime
launch_reward_fn_async=true
first_time_dataset_prep=true #true #false
# number of test cases to sample on the fly during training. Only for code data.
sample_k_code_test_case=20 # increased from 10 to 20 for increased difficulty
do_downsample=false
downsampling_ratio=0.15
enable_gradient_checkpointing=True #False # set to True if OOM
clip_ratio_low=0.2
clip_ratio_high=0.28 # 0.3 #0.3 #0.28
n_resp_per_prompt=16
loss_agg_mode="token-mean"
entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.

save_freq=5
test_freq=5
total_epochs=100
total_training_steps=2500

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
n_val=1

# for async

if [ "$ROLLOUT_mode" = "async" ]; then
  echo "Using async mode"
  export VLLM_USE_V1=1
  return_raw_chat=true
else
  return_raw_chat=false
fi

ppo_micro_batch_size_per_gpu=2
ppo_mini_batch_size=128 # Ensure divisibility
train_batch_size=128 # Maintain train_batch_size = ppo_mini_batch_size has been shown to be effective by Mistral AI

log_prob_micro_batch_size_per_gpu=$((ppo_micro_batch_size_per_gpu * 4))
rollout_log_prob_micro_batch_size_per_gpu=$((ppo_micro_batch_size_per_gpu * 4))

val_batch_size=$((NNODES * GPUS_PER_NODE * ppo_micro_batch_size_per_gpu))
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))
enable_overlong_buffer=true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
CONTEXT_LEN=$((max_prompt_length + max_response_length))
CTX_LEN=16k

# Performance Related Parameter
sp_size=1
# num_attention_heads=8 #14 # for 32B it's 8
# sp_size=$(( num_attention_heads))
use_dynamic_bsz=True
ppo_max_token_len_per_gpu=$((2 * (max_prompt_length + max_response_length)))
actor_ppo_max_token_len=$((2 * (max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$((3 * (max_prompt_length + max_response_length)))
offload=False # was True # Whether to offload model parameters to CPU (trades speed for memory)

# Append key parameters to run name for traceability
RUN_NAME="${NNODES}-nodes-n-${n_resp_per_prompt}${STAGE}-${REWARD_MANAGER}-${CTX_LEN}-k${sample_k_code_test_case}-ds${do_downsample}-ratio${downsampling_ratio}-epshigh-${clip_ratio_high}AM-${SFT_MODEL}-RL"
CKPTS_DIR=${CKPT_ROOT_BASE}/${TRAINED_ON}/${CTX_LEN}/${STAGE}/${RUN_NAME}
mkdir -p "${LOG_BASE_PATH}" "${CKPTS_DIR}" "${DATA_ROOT}"

max_num_batched_tokens=$((max_prompt_length + max_response_length)) #$((2 * CONTEXT_LEN))

echo "Usining training params:"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "ALGORITHM=$ALGORITHM"
echo "CONTEXT_LEN=$CONTEXT_LEN"
echo "tensor_model_parallel_size=$tensor_model_parallel_size"
echo "enable_gradient_checkpointing=$enable_gradient_checkpointing"
echo "val_batch_size=$val_batch_size"
echo "train_batch_size=$train_batch_size"
echo "ppo_mini_batch_size=$ppo_mini_batch_size"
echo "ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu"
echo "log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu"
echo "rollout_log_prob_micro_batch_size_per_gpu=$rollout_log_prob_micro_batch_size_per_gpu"
echo "n_resp_per_prompt=$n_resp_per_prompt"
echo "ROLLOUT_ENGINE=$ROLLOUT_ENGINE"
echo "ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu"

# we should activate it before we start ray to avoid errors
echo "Activating verl_42 environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate verl_42

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=...

# Let Ray know how many nodes to expect
export RAY_NUM_NODES=$NNODES

# Get head node and its IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Convert to IPv4 if needed
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. Using IPV4: $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$port
export ip_head

echo "Starting Ray HEAD at $head_node ($ip_head)"
until nvidia-smi > /dev/null 2>&1; do
  echo "Waiting for GPU visibility..."
  sleep 2
done
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port=$port \
  --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${GPUS_PER_NODE}" --block &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at $node_i"
  until nvidia-smi > /dev/null 2>&1; do
    echo "Waiting for GPU visibility..."
    sleep 2
  done
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${GPUS_PER_NODE}" --block &
  sleep 5
done

# Final launch barrier
sleep 10

echo "Activating verl_42 environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate verl_42
echo "Using $NNODES nodes for training..."

if [ "$first_time_dataset_prep" = true ]; then
  echo "Preparing math data only..."
  python examples/data_preprocess/btx_data.py --local_dir "${DATA_ROOT}" --hf_data_path ${HF_DATA_PATH} --keep_only_math --curriculum_learning
fi

train_path=${DATA_ROOT}/train.parquet
aime24_test_path=${DATA_ROOT}/test_aime24.parquet
aime25_test_path=${DATA_ROOT}/test_aime25.parquet

train_files="['$train_path']"
test_files="['$aime24_test_path', '$aime25_test_path']"

echo "Using $SLURM_NNODES nodes for training..."

echo "==== Confirming Ray sees all GPUs ===="
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
echo "==== Done checking resources ===="
# python verl-latest/examples/data_preprocess/btx_data.py --local_dir /mnt/weka/home/abdelaziz.bounhar/k2/rl/data/math/stage_2 --hf_data_path MBZUAI-Paris/DeepMath-103K-filtered-ChuGyouk-cleaned-levelled-verl --keep_only_math --curriculum_learning

srun --overlap --nodes=${NNODES} --ntasks=1 -w "$head_node"\
    python -m verl.trainer.main_ppo \
    data.shuffle=$SHUFFLE \
    algorithm.adv_estimator=$ALGORITHM \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.truncation='left' \
    data.filter_overlong_prompts=True \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.use_remove_padding=True \
    critic.model.path=$SFT_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.launch_reward_fn_async=${launch_reward_fn_async} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.val_before_train=${val_before_train} \
    speed.sample_k_code_test_case=${sample_k_code_test_case} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${n_val} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    $@
echo "Training completed. Shutting down Ray..."