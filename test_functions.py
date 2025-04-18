from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.engines import test
from dataloader import PTv3_Dataloader
from main import get_student_model, load_weights_ptv3_nucscenes_seg, get_teacher_model
from gpu_main import get_teacher_model as distill_teacher_model
import torch
import torch.nn.functional as F
import time
import os
from gtda.homology import VietorisRipsPersistence
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
from pointcept.models.losses import build_criteria


def get_grid_size(size=0.05):
    CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"

    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

    # grid_size = cfg.data.train.transform[4].grid_size

    cfg.data.train.transform[4].grid_size = size

    print(cfg.data.train.transform[4].grid_size)


def get_sample_data():

    CONFIG_FILE_STUDENT = "configs/nuscenes/semseg-pt-v3m1-0-train-student.py"
    PRETRAINED_PATH_STUDENT = './checkpoints/checkpoint_batch_40001.pth'

    TEACHER_CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"
    PRETRAINED_PATH_TEACHER = './checkpoints/thomas_model_best.pth'

    # Load configuration of teacher
    cfg_teacher = default_config_parser(TEACHER_CONFIG_FILE, None)
    cfg_teacher = default_setup(cfg_teacher)
    cfg_teacher.data.test.transform[1].grid_size = 0.05
    cfg_teacher.model.backbone.enable_flash = False
    cfg_teacher.batch_size = 5

    cfg_student = default_config_parser(CONFIG_FILE_STUDENT, None)
    cfg_student = default_setup(cfg_student)
    cfg_student.data.test.transform[1].grid_size = 0.05
    cfg_student.model.backbone.enable_flash = False
    cfg_student.batch_size = 5
    """
    On test dataset
    grid_size = 0.01 --> Segment shape =  (29844,) | ~ 22 fragments
    grid_size = 0.05 --> Segment shape =  (27815,) | 8 fragments
    grid_size = 0.1 --> Segment shape =  (17659,) | 1 fragment
    
    On train dataset
    grid_size = 0.01 --> Segment shape = (249670)  | 1 fragment
    grid_size = 0.05 --> Segment shape =  (249670) | 1 fragments
    grid_size = 0.1 --> Segment shape =  (249670) | 1 fragment
    """
    class_names = cfg_teacher.names

    teacher_model = get_teacher_model(cfg=cfg_teacher)
    student_model = get_student_model(cfg_student)

    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH_TEACHER)
    student_model = load_weights_ptv3_nucscenes_seg(student_model, PRETRAINED_PATH_STUDENT)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    tester = test.CustomSemSegTester(cfg=cfg_student, model=teacher_model)
    # test_loader = tester.build_test_loader()

    # Data load
    loader = PTv3_Dataloader(cfg_student)
    val_loader = loader.load_training_data()

    teacher_model.eval()
    student_model.eval()
    for batch_ndx, input_dict in enumerate(val_loader):

        if batch_ndx == 0:
            continue

        if batch_ndx == 100:
            break

        # GT
        coord = input_dict['coord']
        segment = input_dict['segment']

        teacher_seg_pred_label = get_pred_time_inference(teacher_model, input_dict, model_type='teacher')
        student_seg_pred_label = get_pred_time_inference(student_model, input_dict, model_type='student')

        # Save paths
        data_type = 'train_data'
        base_path = f'./visualization/{data_type}'

        # Ensure the directory exists
        os.makedirs(base_path, exist_ok=True)

        # Define paths
        teacher_path = f'{base_path}/teacher_train_pred_batch-{cfg_teacher.batch_size}-{batch_ndx}.png'
        student_path = f'{base_path}/student_val_pred_batch-{cfg_student.batch_size}-{batch_ndx}.png'
        GT_path = f'{base_path}/GT_batch-{cfg_student.batch_size}-{batch_ndx}.png'

        tester.visualize_pointcloud_prediction(coords=coord, segment_pred=teacher_seg_pred_label,
                                               class_names=class_names,
                                               save_path=teacher_path,
                                               title="Teacher Prediction")

        tester.visualize_pointcloud_prediction(coords=coord, segment_pred=student_seg_pred_label,
                                               class_names=class_names,
                                               save_path=student_path,
                                               title="Student Prediction")

        tester.visualize_pointcloud_prediction(coords=coord, segment_pred=segment,
                                               class_names=class_names,
                                               save_path=GT_path,
                                               title="Ground Truth")

    # for batch_ndx, input_dict in enumerate(test_loader):
    #     if batch_ndx == 1:
    #         break
    #
    #     input_dict = input_dict[0]
    #     fragment_list = input_dict.pop("fragment_list")
    #     segment = input_dict.pop("segment")
    #     origin_segment = input_dict.pop("origin_segment")
    #
    #     # data_name = input_dict.pop("name")
    #
    #     print("Segment shape = ", segment.shape)
    #     print("Origin segment shape = ", origin_segment.shape)
    #
    #     for i in range(len(fragment_list)):
    #
    #         input_ = fragment_list[i]
    #         coord = input_['coord']
    #         index = input_['index']
    #         offset = input_['offset']
    #         print(coord.shape, index.shape, offset.shape)


def get_pred_time_inference(model, input_dict, model_type='teacher'):

    pred_start = time.time()
    seg_logits = model(input_dict)  # --> 16Gb VRAM is not enough if not splitting into chunks
    pred_end_time = time.time() - pred_start

    seg_pred_softmax = F.softmax(seg_logits, -1)

    seg_pred_label = torch.argmax(seg_pred_softmax, dim=-1)  # Shape: (24150,)

    print(f"Inference {model_type} time = {pred_end_time:.2f} s")

    return seg_pred_label


def compute_ph(features):
    """
    Compute Persistent Homology for input feature maps.
    Args:
        features: torch.Tensor or np.ndarray, shape [n_points, n_features] or [1, n_points, n_features]
    Returns:
        Persistence diagrams
    """
    # Convert torch tensor to numpy array, move to CPU if on GPU
    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()

    # Reshape 2D input to 3D (1, n_points, n_features) if necessary
    if len(features.shape) == 2:
        features = features.reshape(1, *features.shape)

    # Initialize VietorisRipsPersistence
    ph = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1, 2])

    # Compute persistence diagrams
    diagrams = ph.fit_transform(features)
    return diagrams


def compute_ph_sparse_knn(features, k_neighbors=50):
    """
    Compute Persistent Homology using a sparse k-NN distance matrix.
    Args:
        features: torch.Tensor or np.ndarray, shape [n_points, n_features]
        k_neighbors: Number of nearest neighbors for sparse approximation
    Returns:
        Persistence diagrams
    """
    # Convert to numpy if torch tensor
    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()

    # Compute k-nearest neighbors
    n_points = features.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Create sparse distance matrix
    row = np.repeat(np.arange(n_points), k_neighbors)
    col = indices.flatten()
    data = distances.flatten()
    sparse_dist = csr_matrix((data, (row, col)), shape=(n_points, n_points))

    # Ensure sparse matrix is symmetric (required for distance matrix)
    sparse_dist = sparse_dist.maximum(sparse_dist.T)

    # Wrap in a list to match giotto-tda input format
    sparse_dist_list = [sparse_dist]

    # Compute persistent homology
    ph = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2])
    diagrams = ph.fit_transform(sparse_dist_list)
    return diagrams


def compute_ph_batched(features, batch_size=10000):
    """
    Compute Persistent Homology on batches of points.
    Args:
        features: torch.Tensor or np.ndarray, shape [n_points, n_features]
        batch_size: Number of points per batch
    Returns:
        List of persistence diagrams
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()

    n_points = features.shape[0]
    ph = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1, 2])
    diagrams = []

    for i in range(0, n_points, batch_size):
        batch = features[i:i + batch_size]
        batch = batch.reshape(1, *batch.shape)
        batch_diagrams = ph.fit_transform(batch)
        diagrams.append(batch_diagrams[0])  # Store diagrams for each batch

    return diagrams


def test_topo_new_function():
    CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-distill.py"
    PRETRAINED_PATH = './checkpoints/thomas_model_best.pth'

    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)
    cfg.data.test.transform[1].grid_size = 0.05
    cfg.batch_size = 5

    # Enable of disable flash attention | Enable if can request A100 GPU
    cfg.model.teacher_backbone.enable_flash = False

    teacher_model = distill_teacher_model(cfg=cfg)
    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Loss function
    detection_loss_fn = build_criteria(cfg.model.criteria)

    for batch_ndx, input_dict in enumerate(train_loader):

        if batch_ndx == 1:
            break

        # Move all input tensors to the correct device
        input_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in input_dict.items()}

        # Forward pass through teacher model
        # with torch.no_grad():
        seg_logits, feat_enc = teacher_model(input_dict)  # teacher_latent_feature (N, 512) --> for homology, not this

        teacher_preds = torch.argmax(seg_logits, dim=1)

        input_dict['feat'] = input_dict['feat'].to(device).requires_grad_(True)
        ground_truth = input_dict["segment"].to(device).long()

        teacher_loss = detection_loss_fn(seg_logits, ground_truth)

        print(teacher_loss)

        teacher_gradients = torch.autograd.grad(
            outputs=seg_logits,
            inputs=input_dict,
            grad_outputs=torch.ones_like(teacher_loss),
            retain_graph=True,
            create_graph=False
        )
        print(teacher_gradients)

        # print(seg_logits.shape)
        #
        # print(feat_enc.feat.shape)

        # Use Knn to select representative points
        # TDA_features = compute_ph_sparse_knn(feat_enc.feat)

        # # Compute PH based on batch | takes too much time
        # TDA_features = compute_ph_batched(feat_enc.feat, batch_size=5000)

        # print(TDA_features)


if __name__ == "__main__":

    test_topo_new_function()
    # get_grid_size(size=0.1)