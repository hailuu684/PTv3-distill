
# -------- CUDA 12.1.1 ver -----------
#conda create -n pointcept python=3.8 -y
conda activate pointcept

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

module purge
module load shared rc-base CUDA/12.1.1
module load shared rc-base cuDNN/8.9.2.26-CUDA-12.1.1
module load gcc/11.2.0

conda install ninja -y

cd libs/pointops
python setup.py install
cd ../..

pip install spconv-cu120  # choose version match your local cuda version
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cuda121.html

conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv


# Open3D (visualization, optional)
pip install open3d

# ------------------------------------
# -------- CUDA 11.8.0 ver -----------
conda activate pointcept_H100
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

module purge
module load shared rc-base CUDA/11.8.0
module load shared rc-base cuDNN/8.9.2.26-CUDA-11.8.0

# to install, load gcc 8.2.0
module load GCC/8.2.0-2.31.1

conda install ninja -y

cd libs/pointops
python setup.py install
cd ../..

pip install spconv-cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install nuscenes-devkit

#pip install ripser

conda install -c conda-forge cmake
pip3 install ripserplusplus
#pip install git+https://github.com/simonzhang00/ripser-plusplus.git
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
#pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install torch-geometric
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # --> load gcc/11.2.0 first to install
#conda install -c conda-forge gxx_linux-64=11.3.0
pip install flash-attn --no-build-isolation

# to run the code, load back gcc 11.2.0
module load GCC/11.2.0

# if run the code see the error of dont have version glibc3.4.xxx --> reinstall again torch
# train with author code:
# Step 1: export PYTHONPATH=./
# Step 2: sh scripts/train_teacher.sh -p python -g 1 -d nuscenes -c semseg-pt-v3m1-0-train-teacher -r true

# Train with custom code: python main.py