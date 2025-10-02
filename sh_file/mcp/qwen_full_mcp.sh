#!/bin/bash
#SBATCH --job-name=full_qwen
#SBATCH --output=logs/full_qwen_%j.out
#SBATCH --error=logs/full_qwen_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1 
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

source ~/.bashrc
conda activate mad_debate

mkdir -p logs

python main.py --mode multi_people_round_mcq --model qwen --input_file data/matching_evidence_full.json