#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

pip install --upgrade pip

echo "1. Install inference frameworks and pytorch they need"
# Install PyTorch 2.7 first
pip install --no-cache-dir "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" # --index-url https://download.pytorch.org/whl/cu124

echo "2. Install vLLM==0.10.0"
pip install --no-cache-dir "vllm==0.10.0" "tensordict==0.9.1" torchdata

echo "3. install basic ML packages"
pip install "transformers[hf_xet]>=4.54.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "4. Install FlashInfer, FlashAttention, and SGLang"
# FlashInfer: build from source (no torch 2.7.1 wheels available)
git clone --depth=1 https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer && pip install ninja && pip install . && cd ..

# FlashAttention: build from source for torch 2.7.1
git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install && cd ../..

echo "5.1 Install SGLang"
if [ $USE_SGLANG -eq 1 ]; then
    # Note: torch2.7 flashinfer wheels don't exist yet, using torch2.6 wheels with PyTorch 2.7 (may have compatibility issues)
    pip install sglang[all]==0.4.9.post6 torch-memory-saver --no-cache-dir
fi


if [ $USE_MEGATRON -eq 1 ]; then
    echo "5.2 install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2
fi


echo "6.1 May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6.2 Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.8.0.87
fi


echo "7. Verify installation..."
python - <<'PYCODE'
import torch, vllm
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
try:
    import flashinfer_python as fi
    print("flashinfer:", fi.__version__)
except ImportError:
    print("flashinfer: not installed")
try:
    import flash_attn
    print("flash-attn:", flash_attn.__version__, "available:", torch.backends.cuda.flash_sdp_enabled())
except ImportError:
    print("flash-attn: not installed")
PYCODE

echo "Successfully installed all packages!"
