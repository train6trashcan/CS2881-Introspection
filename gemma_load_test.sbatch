#!/bin/bash
#SBATCH --job-name=gemma_load
#SBATCH --account=IscrC_LLM-intr
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hl5625@nyu.edu

set -euo pipefail
module load python/3.11.7
export WORK=/leonardo_work/IscrC_KG-FM
source "$WORK/gemma-introspection-env/bin/activate"
export HF_HOME=/leonardo_work/IscrC_KG-FM/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$HOME/CS2881-Introspection"

nvidia-smi

python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-3-27b-it"
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
print("Model loaded.")
print("Device map:", model.hf_device_map)

inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=12, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
PY
