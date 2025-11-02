# VERL Installation Guide


## 2. Install Dependencies

Choose your setup:

- **For FSDP (without Megatron):**
  ```bash

  conda deactivate
  conda create -n verl_5 python=3.11 -y
  conda activate verl_5
  
  git clone https://github.com/MBZUAI-Paris/verl
  cd verl

  USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

  pip install --no-deps -e .
  pip install -U transformers
  pip install math-verify # can be used for math reward_func
  ```

- **For Megatron support: (still needs to be fixed)**
  ```bash
  conda create -n verl_megatron python=3.9 -y
  conda activate verl_megatron

  git clone https://github.com/volcengine/verl.git
  cd verl

  bash scripts/install_vllm_sglang_mcore.sh
  # install verl together with some lightweight dependencies in setup.py
  pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
  pip3 install flash-attn --no-build-isolation
  pip3 install verl

  # apex
  pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
      git+https://github.com/NVIDIA/apex

  # transformer engine
  pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7

  # megatron core v0.4.0: clone and apply the patch
  # You can also get the patched Megatron code patch via
  # git clone -b core_v0.4.0_verl https://github.com/eric-haibin-lin/Megatron-LM
  cd ..
  git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM
  cp ../verl/patches/megatron_v4.patch .
  git apply megatron_v4.patch
  pip3 install -e .
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  ```



## 3. Install VERL

```bash
pip install --no-deps -e .
pip install verl
pip install megatron
```

---

**Notes:**
- Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
- Always activate the `verl` environment before running any commands:
  ```bash
  conda activate verl
  ```
- For troubleshooting or more details, refer to the [official documentation](https://github.com/volcengine/verl).

## 4. Run a Multi-Node Training Job
