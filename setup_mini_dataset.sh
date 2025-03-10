wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.
wget https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2  # Download the nuScenes-lidarseg mini split.

tar -xf v1.0-mini.tgz  # Uncompress the nuScenes mini split.
tar -xf nuScenes-lidarseg-mini-v1.0.tar.bz2  # Uncompress the nuScenes-lidarseg mini split.

pip install nuscenes-devkit &> /dev/null  # Install nuScenes.

wget https://www.nuscenes.org/data/nuScenes-panoptic-v1.0-mini.tar.gz  # Download the Panoptic nuScenes mini split.
tar -xf nuScenes-panoptic-v1.0-mini.tar.gz # Uncompress the Panoptic nuScenes mini split.