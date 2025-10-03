#!/bin/bash
#SBATCH --job-name=feverous_table_llama
#SBATCH --output=logs/feverous_table_llama_%j.out
#SBATCH --error=logs/feverous_table_llama_%j.err
#SBATCH --time=50:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

# 初始化环境
source ~/.bashrc
conda activate mad_debate

# 创建日志目录
mkdir -p logs

# 执行命令
python -u main.py --mode multi_people --model llama --input_file data/feverous_golden_evidence_new2.json