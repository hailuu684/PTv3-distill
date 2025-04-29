import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES
from pytorch3d.loss import chamfer_distance

# LOSSES = Registry("losses")  

@LOSSES.register_module()
class KLDistillLoss(nn.Module):
    def __init__(self, reduction="batchmean", loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, student_logits, teacher_logits):
        student_log_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        loss = self.kldiv_loss(student_log_probs, teacher_probs)
        return loss * self.loss_weight




@LOSSES.register_module()
class TDAChamferLoss(nn.Module):
    def __init__(self, 
                 k_neighbors=50, 
                 vectorRips_metric="precomputed", 
                 knn_metric="euclidean", 
                 dimensions=[0, 1, 2], 
                 loss_weight=1.0):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.vectorRips_metric = vectorRips_metric
        self.knn_metric = knn_metric
        self.dimensions = dimensions
        self.loss_weight = loss_weight

    def forward(self, student_latent_feature, teacher_latent_feature):
        teacher_diagram = self.compute_ph_sparse_knn(
            teacher_latent_feature.feat, 
            k_neighbors=self.k_neighbors,
            vectorRips_metric=self.vectorRips_metric,
            knn_metric=self.knn_metric
        )

        student_diagram = self.compute_ph_sparse_knn(
            student_latent_feature.feat, 
            k_neighbors=self.k_neighbors,
            vectorRips_metric=self.vectorRips_metric,
            knn_metric=self.knn_metric
        )

        chamfer = self.compute_chamfer_loss(
            teacher_diagram,
            student_diagram,
            dimensions=self.dimensions
        )

        return chamfer * self.loss_weight


    def compute_ph_sparse_knn(features, k_neighbors=50, logger=None,
                            vectorRips_metric="precomputed",
                            knn_metric="euclidean"):
        """
        Compute Persistent Homology using a sparse k-NN distance matrix.
        Args:
            :param knn_metric: "euclidean", "cosine", ...
            :param features: torch.Tensor or np.ndarray, shape [n_points, n_features]
            :param k_neighbors: Number of nearest neighbors for sparse approximation
            :param logger: Loguru logger object
            :param vectorRips_metric: "precomputed", "euclidean", "manhattan" or "cosine"
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

        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric=knn_metric).fit(features)
        distances, indices = nbrs.kneighbors(features)

        row = np.repeat(np.arange(n_points), k_neighbors)
        col = indices.flatten()
        data = distances.flatten()
        sparse_dist = csr_matrix((data, (row, col)), shape=(n_points, n_points))
        sparse_dist = sparse_dist.maximum(sparse_dist.T)

        sparse_dist_list = [sparse_dist]
        ph = VietorisRipsPersistence(metric=vectorRips_metric, homology_dimensions=[0, 1, 2])
        diagrams = ph.fit_transform(sparse_dist_list)[0]  # Shape: (n_features, 3)
        if logger:
            logger.info(f"Computed PH diagrams with {diagrams.shape[0]} features")
        return diagrams


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


@LOSSES.register_module()
class GKDFeatureLoss(nn.Module):
    def __init__(self, normalize=True, norm_type="minmax", loss_weight=1.0):
        super().__init__()
        self.normalize = normalize
        self.norm_type = norm_type
        self.loss_weight = loss_weight

    def gradient_guided_features(self, output_loss, enc_feature):
        """
        Compute gradient-guided features with optional normalization.
        """
        if not enc_feature.feat.requires_grad:
            raise ValueError("enc_feature.feat must have requires_grad=True")
        if enc_feature.feat.grad_fn is None:
            raise ValueError("enc_feature.feat has no grad_fn; it is detached")

        grads = torch.autograd.grad(
            outputs=output_loss,
            inputs=enc_feature.feat,
            grad_outputs=torch.ones_like(output_loss),
            retain_graph=True,
            create_graph=False
        )[0]  # Shape: (N, C)

        weights = grads.abs().mean(dim=0)  # Shape: (C,)

        weighted_feat = enc_feature.feat * weights  # Shape: (N, C)
        M = weighted_feat.sum(dim=1)  # Shape: (N,)

        if self.normalize:
            if self.norm_type == 'minmax':
                M = (M - M.min()) / (M.max() - M.min() + 1e-6)
            elif self.norm_type == 'zscore':
                M = (M - M.mean()) / (M.std() + 1e-6)
            else:
                raise ValueError(f"Unknown norm_type: {self.norm_type}")
        else:
            pass  # No normalization applied

        return M

    def forward(self, student_loss, student_latent_feature, teacher_loss, teacher_latent_feature):
        M_teacher = self.gradient_guided_features(teacher_loss, teacher_latent_feature).detach()
        M_student = self.gradient_guided_features(student_loss, student_latent_feature)

        loss = F.smooth_l1_loss(M_student, M_teacher, beta=1.0)
        return loss * self.loss_weight

