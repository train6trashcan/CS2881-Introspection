#!/bin/bash
#SBATCH --job-name=persona_L21_v2
#SBATCH --account=IscrC_KG-FM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hl5625@nyu.edu

set -euo pipefail
module load python/3.11.7
export WORK=/leonardo_work/IscrC_KG-FM
source "$WORK/llama-introspection-env/bin/activate"
export HF_HOME=/leonardo_work/IscrC_KG-FM/hf_cache
export HF_HUB_OFFLINE=1
cd "$HOME/CS2881-Introspection"

python persona_pilot_L21_v2.py
