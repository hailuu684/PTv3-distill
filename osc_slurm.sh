#!/bin/bash
#SBATCH --account=PCS0252
#SBATCH --job-name=ptv3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=terminal_logs/result_%j.txt
#SBATCH --error=terminal_logs/error_%j.txt
# SBATCH --partition=quad



## gcc/12.3.0

module load miniconda3/24.1.2-py310
module load cuda/12.4.1
conda activate ptv3_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Using PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"


export PYTHONPATH=./

## python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base.py --num-gpus 1 --options save_path=exp/dataset_type/waymo/waymo-train-teacher_4 resume=True weight=/users/PCS0253/dingz/projects/ptv3_distill/PTv3-distill/exp/dataset_type/waymo/waymo-train-teacher_4/model/model_last.pth

## python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base.py --num-gpus 4 --options save_path=exp/dataset_type/waymo/waymo-train-teacher
# python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base_tea_4gpu.py --num-gpus 4 --options save_path=exp/dataset_type/waymo/waymo-train-teacher_4_2

# python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base_tea_4gpu.py --num-gpus 4 --options save_path=exp/dataset_type/waymo/waymo-train-teacher_4_2 resume=True weight=/users/PCS0253/dingz/projects/ptv3_distill/PTv3-distill/exp/dataset_type/waymo/waymo-train-teacher_4_2/model/model_last.pth


# python tools/test.py --config-file configs/waymo/semseg-pt-v3m1-0-base_tea_4gpu.py --options save_path=exp/dataset_type/waymo/waymo-train-teacher_4_2/best weight=exp/dataset_type/waymo/waymo-train-teacher_4_2/model/model_best.pth

python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base_topology.py --num-gpus 1 --options save_path=exp/dataset_type/waymo/waymo-train-teacher_topology_debug

# python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base_student_regular.py --num-gpus 4 --options save_path=exp/dataset_type/waymo/waymo-train-base_student_regular_30_H
# python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base_student_grid_0.1.py --num-gpus 4 --options save_path=exp/dataset_type/waymo/waymo-train-base_student_0.1_30_H
# python tools/train.py --config-file configs/waymo/semseg-pt-v3m1-0-base_student_grid_0.01.py --num-gpus 4 --options save_path=exp/dataset_type/waymo/waymo-train-base_student_0.01_30_H


# python tools/test.py --config-file configs/waymo/semseg-pt-v3m1-0-base_student_grid_0.01.py --num-gpus 4 --options save_path=exp/dataset_type/waymo/waymo-train-teacher_4_2/best weight=exp/dataset_type/waymo/waymo-train-base_student_0.01_30_H/model/model_best.pth



