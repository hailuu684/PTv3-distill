#!/bin/bash
#SBATCH --job-name=ptv3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
############# SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --output=terminal_logs/result_%j.txt
#SBATCH --error=terminal_logs/error_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL

# cd data
# wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
# wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
# wget https://www.semantic-kitti.org/assets/data_odometry_labels.zip
# srun --ntasks=1 --mem=128G --partition=amperenodes --gres=gpu:1 --time=03:00:00 --pty /bin/bash


module load Anaconda3/2022.05
module load shared rc-base CUDA/11.8.0
module load shared rc-base cuDNN/8.9.2.26-CUDA-11.8.0
module load GCC/11.2.0
conda activate ptv3_3


python main.py

######## ssh dingz@bgsu.edu@cheaha.rc.uab.edu
# module load Anaconda3/2022.05
# module load shared rc-base CUDA/11.8.0
# module load shared rc-base cuDNN/8.9.2.26-CUDA-11.8.0
# module load GCC/8.2.0-2.31.1
# conda activate ptv3_3
# conda clean --all -y
# conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
# pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
# pip install torch-geometric
# module load GCC/11.2.0
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# pip install flash-attn --no-build-isolation

# module load Anaconda3/2022.05
# module load shared rc-base CUDA/11.8.0
# module load shared rc-base cuDNN/8.9.2.26-CUDA-11.8.0
# module load GCC/8.2.0-2.31.1
# conda create -n ptv3_4 python=3.8
# conda activate ptv3_4
# pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# conda install ninja -y
# cd libs/pointops
# python setup.py install
# cd ../..
# pip install spconv-cu118
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
# pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
# pip install nuscenes-devkit
# conda install -c conda-forge cmake -y
# pip3 install ripserplusplus

# pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
# pip install torch-geometric
# module load GCC/11.2.0
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# pip install flash-attn --no-build-isolation
