#!/bin/bash
#SBATCH --job-name=ptv3-multinode
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --partition=amperenodes-medium
#SBATCH --time=48:00:00
#SBATCH --output=terminal_logs/result_%j.txt
#SBATCH --error=terminal_logs/error_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL

module load Anaconda3/2022.05
module load shared rc-base CUDA/11.8.0
module load shared rc-base cuDNN/8.9.2.26-CUDA-11.8.0
module load GCC/12.2.0
conda activate ptv3_3

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=./

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)

python tools/train.py \
  --config-file configs/semantic_kitti/kitti_v3.py \
  --num-gpus 2 \
  --num-machines 2 \
  --machine-rank $SLURM_PROCID \
  --dist-url tcp://$MASTER_ADDR:29500 \
  --options save_path=exp/dataset_type/kitti/semseg-pt-v3m1-0-train-teacher
