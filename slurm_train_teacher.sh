#!/bin/bash
#SBATCH --job-name=Luu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
############# SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --partition=pascalnodes
#SBATCH --time=1:00:00
#SBATCH --output=terminal_logs/result_%j.txt
#SBATCH --error=terminal_logs/error_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL

module load Anaconda3/2022.05
conda --version
conda activate pointcept_H100
module load shared rc-base CUDA/11.8.0
module load shared rc-base cuDNN/8.9.2.26-CUDA-11.8.0
module load GCC/11.2.0

python main.py