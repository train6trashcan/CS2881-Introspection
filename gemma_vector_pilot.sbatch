#!/bin/bash
#SBATCH --job-name=gemma_vec
#SBATCH --account=IscrC_LLM-intr
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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

python save_vectors.py --model google/gemma-3-27b-it --datasets simple_data --layer_range 37 --save_dir saved_vectors/gemma3
