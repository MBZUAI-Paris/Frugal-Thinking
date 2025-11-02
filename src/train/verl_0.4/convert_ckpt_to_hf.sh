#!/bin/bash

HF_TOKEN=...

HF_PATH=... # the desired Hugging Face repository path (e.g., username/repo_name)
PATH_TO_CKPT=... # local path to the checkpoint directory

if [ ${#HF_PATH} -gt 95 ]; then
  echo "Error: HF_PATH exceeds 95 characters (actual: ${#HF_PATH}). Exiting."
  exit 1
fi

echo "Activating verl_4 environment..."
eval "$(conda shell.bash hook)"
conda deactivate
conda activate verl_4

echo "PATH_TO_CKPT: $PATH_TO_CKPT"
echo "HF_PATH: $HF_PATH"

python -m convert_ckpt_to_hf \
  "$PATH_TO_CKPT/actor" \
  "$PATH_TO_CKPT/actor/huggingface/" \
  "$HF_PATH" \
  --token "$HF_TOKEN"