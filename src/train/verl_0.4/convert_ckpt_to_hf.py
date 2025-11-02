import os
import re
from collections import defaultdict

import fire
import torch
import torch.distributed.tensor
from huggingface_hub import HfApi


def _import_transformers_modules():
    """Import transformers modules while preventing flash_attn from loading.

    Some environments ship with a `flash_attn` build that is incompatible with the
    local PyTorch version. When transformers detects the package it eagerly imports
    it and crashes with an undefined-symbol error. We monkey-patch the import check
    so transformers behaves as if `flash_attn` is unavailable.
    """

    from transformers.utils import import_utils as _import_utils

    if not hasattr(_import_transformers_modules, "_patched"):
        original_is_package_available = _import_utils._is_package_available

        def _skip_flash_attn(pkg_name: str, return_version: bool = False):
            if pkg_name == "flash_attn":
                if return_version:
                    return False, "N/A"
                return False
            return original_is_package_available(pkg_name, return_version)

        _import_utils._is_package_available = _skip_flash_attn
        _import_transformers_modules._patched = True

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    return AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main(
    fsdp_checkpoint_path, 
    huggingface_model_config_path, 
    output_path, 
    token: str = None,
):
    """
    Convert FSDP checkpoint to HuggingFace checkpoint
    Args:
        fsdp_checkpoint_path: path to the FSDP checkpoint
        huggingface_model_path: path to the HuggingFace model
        output_path: path to save the converted checkpoint
    Usage:
        python reason_rl/utils/convert2hf.py \
            checkpoints/azr/azr/test/test_answer/Qwen2.5-7B/answer_conditional/global_step_160_copy/actor \
            checkpoints/azr/azr/test/test_answer/Qwen2.5-7B/answer_conditional/global_step_160_copy/actor/huggingface/ \
            azr_90_composite_160_steps
    """
    state_dict = defaultdict(list)

    assert len(output_path) < 96, f"Hugging Face path should be of len < 96. Got {len(output_path)}."

    # Automatically determine world_size from checkpoint files
    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
    world_sizes = set()
    for fname in os.listdir(fsdp_checkpoint_path):
        match = pattern.match(fname)
        if match:
            world_sizes.add(int(match.group(1)))
    if not world_sizes:
        raise ValueError(f"No model_world_size_*_rank_*.pt files found in {fsdp_checkpoint_path}")
    world_size = max(world_sizes)
    print(f"Detected world_size: {world_size}")

    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print("loading", filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    AutoConfig, AutoModelForCausalLM, AutoTokenizer = _import_transformers_modules()

    config = AutoConfig.from_pretrained(huggingface_model_config_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)
    model.save_pretrained(huggingface_model_config_path)

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_config_path)
    tokenizer.save_pretrained(huggingface_model_config_path)

    # Push model and tokenizer to Hugging Face Hub
    print("Pushing model and tokenizer to Hugging Face Hub...")
    # Explicitly create the repo before pushing
    api = HfApi()
    api.create_repo(repo_id=output_path, token=token, private=True, exist_ok=True)
    model.push_to_hub(
        output_path, 
        token=token, 
        private=True,
    )
    tokenizer.push_to_hub(
        output_path, 
        token=token, 
        private=True,
    )

if __name__ == "__main__":
    fire.Fire(main)
