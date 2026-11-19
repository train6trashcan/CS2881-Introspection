#!/bin/bash
#SBATCH --job-name=persona_gemma
#SBATCH --account=IscrC_LLM-intr
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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
mkdir -p logs new_results

python persona_full_v3_gemma.py --layer 37
