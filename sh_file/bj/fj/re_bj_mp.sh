#!/bin/bash
#SBATCH --job-name=re_bj_mp
#SBATCH --output=logs/re_bj_mp_%j.out
#SBATCH --error=logs/re_bj_mp_%j.err
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

python main.py --mode multi_people_bj --model llama --input_file data/matching_evidence.json