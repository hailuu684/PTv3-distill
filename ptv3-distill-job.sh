#!/bin/bash
#SBATCH --job-name=ptv3-distill              ### Name of the job
#SBATCH --nodes=2                     ### Number of Nodes (max for amperenodes)

#SBATCH --gres=gpu:1                    ### Number of GPUs per node
#SBATCH --mem=150G                  ### 160GB memory per node
#SBATCH --partition=amperenodes     ### Cheaha Partition
#SBATCH --time=12:00:00             ### Estimated Time of Completion, 1 hour
#SBATCH --output=%x.out          ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=%x.err           ### Slurm Error file, %x is job name, %j is job id


### Loading the required CUDA and cuDNN modules
module purge

###  Load the correct CUDA module based on the output of module spider
module load shared rc-base CUDA/11.8.0

###  Load the correct cuDNN module based on the output of module spider
module load shared rc-base cuDNN/8.9.2.26-CUDA-11.8.0

module load gcc/8.2.0

### Load the Anaconda3 module
module load Anaconda3/2023.07-2 # Or choose a different version if needed

# conda create --name thomas_pointcept python=3.8 -y
conda activate thomas_pointcept

pip install git+https://github.com/simonzhang00/ripser-plusplus.git
conda install ninja -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric
pip install spconv-cu118
pip install open3d
conda install pytorch3d -c pytorch3d

cd libs/pointops
python setup.py install -v
cd ../..

echo -n "Env setup done"

### Executing the python script
python ./gpu_main.py