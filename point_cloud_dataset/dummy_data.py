import torch
from torch.utils.data import Dataset, DataLoader


class DummyNuScenesDataset(Dataset):
    def __init__(self, num_samples=100, num_points=1024, in_channels=6):
        """
        Initializes the dataset with random point clouds and dummy labels.

        :param num_samples: Number of point clouds in the dataset.
        :param num_points: Number of points per point cloud.
        :param in_channels: Number of channels (features) per point.
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.in_channels = in_channels

        # Generate random point clouds
        self.point_clouds = torch.rand(num_samples, num_points, in_channels)  # Random points and features
        self.grid_coords = torch.randint(0, 10, (num_samples, num_points, 3))  # Dummy grid coordinates (voxelization)
        self.batch_offsets = torch.arange(num_samples).repeat_interleave(num_points)  # Dummy batch indices
        self.labels = torch.randint(0, 10, (num_samples,))  # Random class labels for classification

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Prepare the data_dict for the forward function of PTv3
        data_dict = {
            "feat": self.point_clouds[idx],          # Feature of the point cloud
            "grid_coord": self.grid_coords[idx],      # Discrete coordinate after grid sampling (voxelization)
            "offset": self.batch_offsets              # Batch indices or offsets
        }
        label = self.labels[idx]
        return data_dict, label


