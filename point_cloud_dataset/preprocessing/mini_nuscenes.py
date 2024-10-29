import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import points_in_box
import os
import numpy as np
import pickle


# Define a custom PyTorch Dataset class to load NuScenes data
class NuScenesDataset(Dataset):
    def __init__(self, nuscenes, use_augmentation=False, scale=20, full_scale=4096):
        self.nusc = nuscenes
        self.samples = self.nusc.sample
        self.use_augmentation = use_augmentation
        self.scale = scale
        self.full_scale = full_scale

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get lidar data (e.g., from LIDAR_TOP sensor)
        lidar_data_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_data_token)
        lidar_filepath = os.path.join(self.nusc.dataroot, lidar_data['filename'])

        # Load lidar point cloud
        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

        # Convert point cloud to numpy and then to a PyTorch tensor
        points = lidar_pointcloud.points[:3, :]  # We only take x, y, z coordinates
        point_cloud_tensor = torch.tensor(points.T, dtype=torch.float32)  # Shape: [N, 3]

        # 3D augmentation (optional)
        if self.use_augmentation:
            point_cloud_tensor = self.augment_and_scale_3d(point_cloud_tensor, self.scale, self.full_scale)

        # Voxelization or scaling (optional, depending on the model needs)
        coords = self.voxelize(point_cloud_tensor) if self.full_scale else point_cloud_tensor

        return {
            'coords': coords,  # The processed coordinates (either voxelized or raw)
            'feats': torch.ones(len(coords), 1, dtype=torch.float32),  # Dummy feature vector
        }

    def augment_and_scale_3d(self, points, scale, full_scale,
                             noisy_rot=0.0,
                             flip_x=0.0,
                             flip_y=0.0,
                             rot_z=0.0,
                             transl=False):
        """
        3D point cloud augmentation and scaling from points (in meters) to voxels
        :param points: 3D points in meters
        :param scale: voxel scale in 1 / m, e.g. 20 corresponds to 5cm voxels
        :param full_scale: size of the receptive field of SparseConvNet
        :param noisy_rot: scale of random noise added to all elements of a rotation matrix
        :param flip_x: probability of flipping the x-axis (left-right in nuScenes LiDAR coordinate system)
        :param flip_y: probability of flipping the y-axis (left-right in Kitti LiDAR coordinate system)
        :param rot_z: angle in rad around the z-axis (up-axis)
        :param transl: True or False, random translation inside the receptive field of the SCN, defined by full_scale
        :return coords: the coordinates that are given as input to SparseConvNet
        """
        if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0:
            rot_matrix = np.eye(3, dtype=np.float32)
            if noisy_rot > 0:
                # add noise to rotation matrix
                rot_matrix += np.random.randn(3, 3) * noisy_rot
            if flip_x > 0:
                # flip x axis: multiply element at (0, 0) with 1 or -1
                rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
            if flip_y > 0:
                # flip y axis: multiply element at (1, 1) with 1 or -1
                rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
            if rot_z > 0:
                # rotate around z-axis (up-axis)
                theta = np.random.rand() * rot_z
                z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                         [np.sin(theta), np.cos(theta), 0],
                                         [0, 0, 1]], dtype=np.float32)
                rot_matrix = rot_matrix.dot(z_rot_matrix)
            points = points.dot(rot_matrix)

        # scale with inverse voxel size (e.g. 20 corresponds to 5cm)
        coords = points * scale
        # translate points to positive octant (receptive field of SCN in x, y, z coords is in interval [0, full_scale])
        coords -= coords.min(0)

        if transl:
            # random translation inside receptive field of SCN
            offset = np.clip(full_scale - coords.max(0) - 0.001, a_min=0, a_max=None) * np.random.rand(3)
            coords += offset

        return coords

    def voxelize(self, point_cloud):
        """
        Voxelization or scaling of the point cloud data.
        :param point_cloud: Tensor of point cloud coordinates [N, 3]
        :return: Voxelized coordinates or scaled point cloud
        """
        # Here you can apply voxelization or scaling as per your model's requirements
        coords = point_cloud * self.scale
        coords = coords.long()  # Convert to voxelized format if needed
        coords = torch.clamp(coords, min=0, max=self.full_scale - 1)  # Ensure within voxel grid
        return coords


class PreprocessedNuScenesDataset(Dataset):
    def __init__(self, preprocessed_data_path, use_augmentation=False, scale=20, full_scale=4096):
        """
        Dataset class for loading preprocessed point cloud data.

        :param preprocessed_data_path: Path to the directory containing the preprocessed pickle file.
        :param use_augmentation: Whether to apply data augmentation to the point cloud.
        :param scale: Scale factor for the point cloud.
        :param full_scale: The size of the voxelization or bounding box scale.
        """
        self.preprocessed_data_path = preprocessed_data_path
        self.use_augmentation = use_augmentation
        self.scale = scale
        self.full_scale = full_scale

        # Load the preprocessed point cloud data from the pickle file
        with open(preprocessed_data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data for the given index
        data_dict = self.data[idx]

        # Extract the point cloud and convert it to a PyTorch tensor
        points = data_dict['points']  # (num_points, 3)
        point_cloud_tensor = torch.tensor(points, dtype=torch.float32)

        # Apply 3D augmentation if specified
        if self.use_augmentation:
            point_cloud_tensor = self.augment_and_scale_3d(point_cloud_tensor, self.scale, self.full_scale)

        # Voxelization or scaling (optional, depending on the model needs)
        # coords = self.voxelize(point_cloud_tensor) if self.full_scale else point_cloud_tensor
        coords = point_cloud_tensor

        return {
            'coords': coords,  # Processed coordinates (either voxelized or raw)
            'feats': torch.ones(len(coords), 1, dtype=torch.float32),  # Dummy feature vector (for now)
        }

    def augment_and_scale_3d(self, points, scale, full_scale, noisy_rot=0.0, flip_x=0.0, flip_y=0.0, rot_z=0.0,
                             transl=False):
        """
        3D point cloud augmentation and scaling from points (in meters) to voxels.
        :param points: 3D points in meters.
        :param scale: Voxel scale in 1 / m, e.g. 20 corresponds to 5cm voxels.
        :param full_scale: Size of the receptive field of SparseConvNet.
        :param noisy_rot: Scale of random noise added to all elements of a rotation matrix.
        :param flip_x: Probability of flipping the x-axis.
        :param flip_y: Probability of flipping the y-axis.
        :param rot_z: Angle in radians around the z-axis (up-axis).
        :param transl: Random translation inside the receptive field of the SCN.
        :return: Scaled and augmented point cloud coordinates.
        """
        # Augment points with rotation, flipping, and random translation
        if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0:
            rot_matrix = np.eye(3, dtype=np.float32)
            if noisy_rot > 0:
                rot_matrix += np.random.randn(3, 3) * noisy_rot
            if flip_x > 0:
                rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
            if flip_y > 0:
                rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
            if rot_z > 0:
                theta = np.random.rand() * rot_z
                z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                         [np.sin(theta), np.cos(theta), 0],
                                         [0, 0, 1]], dtype=np.float32)
                rot_matrix = rot_matrix.dot(z_rot_matrix)
            points = points.dot(rot_matrix)

        # Scale with inverse voxel size
        coords = points * scale

        # Translate points to positive octant
        coords = coords - coords.min(dim=0, keepdim=True)[0]

        if transl:
            offset = np.clip(full_scale - coords.max(0) - 0.001, a_min=0, a_max=None) * np.random.rand(3)
            coords += offset

        return coords

    def voxelize(self, point_cloud):
        """
        Voxelization or scaling of the point cloud data.
        :param point_cloud: Tensor of point cloud coordinates [N, 3]
        :return: Voxelized coordinates or scaled point cloud.
        """
        coords = point_cloud * self.scale
        coords = coords.long()  # Convert to voxelized format if needed
        coords = torch.clamp(coords, min=0, max=self.full_scale - 1)  # Ensure within voxel grid
        return coords


def preprocess_point_cloud(nusc, out_dir, location=None):
    """
    Preprocess the NuScenes dataset to extract LIDAR point cloud data and segmentation labels.

    :param nusc: NuScenes instance
    :param out_dir: Directory to save the preprocessed data
    :param location: Optional, filter data based on location
    """
    # Initialize list to store the data
    point_cloud_data = []
    class_names_to_id = {  # Map object class names to numerical IDs
        'car': 0,
        'truck': 1,
        'bus': 2,
        'trailer': 3,
        'construction_vehicle': 4,
        'pedestrian': 5,
        'motorcycle': 6,
        'bicycle': 7,
        'traffic_cone': 8,
        'barrier': 9,
        'background': 10  # For points that don't belong to any object
    }

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # Filter for specific location, if provided
        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        # Get the LIDAR data token and load point cloud
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_path, boxes_lidar, _ = nusc.get_sample_data(lidar_token)

        print(f'Processing {i + 1}/{len(nusc.sample)}: {curr_scene_name}, {lidar_path}')

        # Load LIDAR points (x, y, z)
        pts = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3].T

        # Transpose to yield shape (num_points, 3)
        pts = pts.T

        # Initialize segmentation labels as "background" for all points
        seg_labels = np.full(pts.shape[0], fill_value=class_names_to_id['background'], dtype=np.uint8)

        # Assign labels to points inside the bounding boxes
        for box in boxes_lidar:
            # Get points that lie inside the box
            fg_mask = points_in_box(box, pts.T)

            # Get the detection class
            det_class = category_to_detection_name(box.name)
            if det_class is not None and det_class in class_names_to_id:
                seg_labels[fg_mask] = class_names_to_id[det_class]

        # Store data in dictionary
        data_dict = {
            'points': pts,
            'seg_labels': seg_labels,  # Add segmentation labels to the data
            'lidar_path': lidar_path,
            "sample_token": sample["token"],
            "scene_name": curr_scene_name,
        }

        # Append to the list of point cloud data
        point_cloud_data.append(data_dict)

    # Save the collected point cloud data and segmentation labels to a pickle file
    save_dir = os.path.join(out_dir, 'preprocess_data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'point_cloud_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(point_cloud_data, f)
        print(f'Wrote preprocessed point cloud data with segmentation labels to {save_path}')

