from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.engines import test
from dataloader import PTv3_Dataloader
from main import get_student_model, load_weights_ptv3_nucscenes_seg, get_teacher_model
from gpu_main import get_teacher_model as distill_teacher_model
from gpu_main import get_student_model as distill_student_model
from pytorch3d.loss import chamfer_distance
import torch
import torch.nn.functional as F
import time
import os
from gtda.homology import VietorisRipsPersistence
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
from pointcept.models.losses import build_criteria
# from gpu_main import compute_chamfer_loss


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


def compute_ph_sparse_knn(features, k_neighbors=50, logger=None):
    """
    Compute Persistent Homology using a sparse k-NN distance matrix.
    Args:
        features: torch.Tensor or np.ndarray, shape [n_points, n_features]
        k_neighbors: Number of nearest neighbors for sparse approximation
        logger: Loguru logger object
    Returns:
        diagrams: NumPy array of persistence diagrams, shape (n_features, 3)
    """
    if isinstance(features, torch.Tensor):
        if logger:
            logger.info("Detaching tensor for giotto-tda compatibility")
        features = features.cpu().detach().numpy()

    n_points = features.shape[0]
    if n_points < k_neighbors:
        raise ValueError(f"n_points ({n_points}) must be >= k_neighbors ({k_neighbors})")

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(features)
    distances, indices = nbrs.kneighbors(features)

    row = np.repeat(np.arange(n_points), k_neighbors)
    col = indices.flatten()
    data = distances.flatten()
    sparse_dist = csr_matrix((data, (row, col)), shape=(n_points, n_points))
    sparse_dist = sparse_dist.maximum(sparse_dist.T)

    sparse_dist_list = [sparse_dist]
    ph = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2],
                                 reduced_homology=True)
    diagrams = ph.fit_transform(sparse_dist_list)[0]  # Shape: (n_features, 3)
    if logger:
        logger.info(f"Computed PH diagrams with {diagrams.shape[0]} features")
    return diagrams


def compute_persistence_entropy(diagram, dimensions=[0, 1, 2], device='cuda'):
    """
    Compute differentiable persistence entropy features for each homology dimension.
    Args:
        diagram: torch.Tensor of shape (N, 3) with [birth, death, dimension]
        dimensions: List of homology dimensions to compute entropy for
        device: Device for computation
    Returns:
        torch.Tensor: Entropy features of shape (n_dimensions, 1)
    """
    entropy_features = []
    for dim in dimensions:
        mask = diagram[:, 2] == dim
        points = diagram[mask]
        if points.shape[0] == 0:
            entropy_features.append(torch.tensor(0.0, device=device))
            continue

        # Compute persistence (death - birth)
        persistence = points[:, 1] - points[:, 0]  # Shape: (N,)
        persistence = torch.clamp(persistence, min=0.0)  # Handle numerical errors

        # Normalize persistence to sum to 1 (like probabilities)
        total_persistence = torch.sum(persistence)
        if total_persistence > 0:
            p = persistence / (total_persistence + 1e-8)
            # Compute entropy: -sum(p * log(p))
            entropy = -torch.sum(p * torch.log(p + 1e-8))
        else:
            entropy = torch.tensor(0.0, device=device)

        entropy_features.append(entropy)

    return torch.stack(entropy_features).unsqueeze(-1)  # Shape: (n_dimensions, 1)


def normalize_diagram(diagram, max_death_cap=1e6):
    """
    Min-max normalize a persistence diagram's (birth, death) coordinates, handling inf values.
    Args:
        diagram: torch.Tensor of shape (N, 3) with [birth, death, dimension]
        max_death_cap: Finite value to replace inf death times
    Returns:
        torch.Tensor: Normalized tensor with (birth, death) in [0, 1], dimension unchanged
    """
    bd = diagram[:, :2]
    bd = torch.where(torch.isinf(bd), torch.tensor(max_death_cap, device=bd.device), bd)
    finite_mask = torch.isfinite(bd)
    if finite_mask.any():
        min_val = torch.min(bd[finite_mask])
        max_val = torch.max(bd[finite_mask])
        normalized_bd = (bd - min_val) / (max_val - min_val + 1e-8)
    else:
        normalized_bd = bd
    normalized_diagram = torch.cat([normalized_bd, diagram[:, 2:3]], dim=1)
    return normalized_diagram


def compute_chamfer_loss(persistence_diagram_1, persistence_diagram_2, dimensions=[0, 1, 2]):
    """
    Compute differentiable Chamfer loss on persistence entropy features across homology dimensions.
    Args:
        persistence_diagram_1: NumPy array of shape (n_features, 3) with [birth, death, dim]
        persistence_diagram_2: NumPy array of shape (n_features, 3) with [birth, death, dim]
        dimensions: List of homology dimensions to compute loss for
    Returns:
        torch.Tensor: Chamfer loss on entropy features
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pd1 = torch.from_numpy(persistence_diagram_1).float().to(device)
    pd2 = torch.from_numpy(persistence_diagram_2).float().to(device)

    pd1 = normalize_diagram(pd1)
    pd2 = normalize_diagram(pd2)

    # Compute entropy features
    entropy1 = compute_persistence_entropy(pd1, dimensions, device)
    entropy2 = compute_persistence_entropy(pd2, dimensions, device)

    # Treat entropy features as points for Chamfer loss
    points1 = entropy1.unsqueeze(0)  # Shape: [1, n_dimensions, 1]
    points2 = entropy2.unsqueeze(0)  # Shape: [1, n_dimensions, 1]

    if torch.any(torch.isnan(points1)) or torch.any(torch.isinf(points1)) or \
       torch.any(torch.isnan(points2)) or torch.any(torch.isinf(points2)):
        # logger.warning("NaN or inf in entropy features; returning 0 loss")
        return torch.tensor(0.0, device=device, requires_grad=False)

    loss_chamfer, _ = chamfer_distance(points1, points2)
    if torch.isnan(loss_chamfer) or torch.isinf(loss_chamfer):
        # logger.warning("Chamfer loss is NaN or inf; returning 0 loss")
        return torch.tensor(0.0, device=device, requires_grad=False)

    # logger.info(f"Chamfer Loss on entropy features: {loss_chamfer.item():.4f}")
    return loss_chamfer


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

    # Init models
    teacher_model = distill_teacher_model(cfg=cfg)
    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH)

    student_model = distill_student_model(cfg=cfg)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Loss function
    detection_loss_fn = build_criteria(cfg.model.criteria)

    # Optimizer for student model
    student_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    student_model.train()
    for batch_ndx, input_dict in enumerate(train_loader):

        if batch_ndx == 1:
            break

        # Move all input tensors to the correct device
        input_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in input_dict.items()}

        # for param in teacher_model.parameters():
        #     param.requires_grad = True

        teacher_seg_logits, teacher_latent_feature = teacher_model(
            input_dict)  # teacher_latent_feature (N, 512) --> for homology, not this

        student_seg_logits, student_latent_feature = student_model(input_dict)  # student_latent_feature (N, 512)

        input_dict['feat'] = input_dict['feat'].to(device).requires_grad_(True)
        ground_truth = input_dict['segment'].to(device).long()

        # CE + Lovasz Loss
        teacher_loss = detection_loss_fn(teacher_seg_logits, ground_truth)
        student_loss = detection_loss_fn(student_seg_logits, ground_truth)

        # Get gradients
        M_teacher = gradient_guided_features(teacher_loss, teacher_latent_feature)
        M_student = gradient_guided_features(student_loss, student_latent_feature)

        # gkd_loss = F.l1_loss(M_student, M_teacher)

        # Compute Smooth L1 loss (replacing L1 loss)
        gkd_loss = F.smooth_l1_loss(M_student, M_teacher, beta=1.0)  # beta controls smoothness

        # Use Knn to select representative points
        teacher_TDA_features = compute_ph_sparse_knn(teacher_latent_feature.feat, k_neighbors=50)
        student_TDA_features = compute_ph_sparse_knn(student_latent_feature.feat, k_neighbors=50)
        # print(teacher_TDA_features)
        chamfer_loss = compute_chamfer_loss(teacher_TDA_features, student_TDA_features, dimensions=[0, 1, 2])

        # # Compute PH based on batch | takes too much time --> don't recommend this
        # TDA_features = compute_ph_batched(feat_enc.feat, batch_size=5000)

        # Total loss
        total_loss = student_loss + gkd_loss + chamfer_loss

        # Backward pass for student model
        student_optimizer.zero_grad()
        total_loss.backward()
        student_optimizer.step()

        print(f"Batch {batch_ndx}, "
              f"GKD Loss: {gkd_loss.item():.4f}, "
              f"Chamfer Loss: {chamfer_loss.item():.4f}, "
              f"Total Loss: {total_loss.item():.4f}")

        # Explicitly clear teacher-related tensors to free memory
        del teacher_loss, teacher_seg_logits, teacher_latent_feature, M_teacher, teacher_TDA_features
        torch.cuda.empty_cache()  # Clear GPU memory cache (use cautiously)

        # for param in teacher_model.parameters():
        #     param.requires_grad = False
        #
        # teacher_model.eval()

        # ... backward code


def gradient_guided_features(output_loss, enc_feature):
    # enc_feature.feat.retain_grad()

    # 1. Get gradients of loss w.r.t. point-wise features
    grads = torch.autograd.grad(
        outputs=output_loss,
        inputs=enc_feature.feat,
        grad_outputs=torch.ones_like(output_loss),
        retain_graph=True,
        create_graph=False
    )[0]  # Shape: (N, C)

    # 2. Compute importance per channel: w = 1 / N * Sum(L/Activation)
    weights = grads.abs().mean(dim=0)  # Shape: (C,)

    # 3. Weight the original features
    weighted_feat = enc_feature.feat * weights  # broadcasting over points

    M = weighted_feat.sum(dim=1)  # Shape: (N,)
    M = (M - M.min()) / (M.max() - M.min() + 1e-6)  # Normalize

    return M


if __name__ == "__main__":
    test_topo_new_function()
    # get_grid_size(size=0.1)
