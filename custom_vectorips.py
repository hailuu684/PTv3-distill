import unittest
import torch
import torch.nn as nn
import numpy as np
from gtda.homology import VietorisRipsPersistence
from pytorch3d.loss import chamfer_distance
from torch_cluster import knn


class DifferentiableVietorisRipsPersistence:
    """
    A PyTorch-based, differentiable approximation of Vietoris-Rips persistent homology.
    Computes persistence diagrams for dimensions 0 and 1 using the full point cloud.
    """

    def __init__(self, homology_dimensions=[0, 1], max_edge_length=0.4):
        """
        Args:
            homology_dimensions: List of homology dimensions to compute (e.g., [0, 1]).
            max_edge_length: Maximum edge length for filtration.
        """
        self.homology_dimensions = homology_dimensions
        self.max_edge_length = max_edge_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_distance_matrix(self, points):
        """
        Compute pairwise distances for k-NN edges.
        Args:
            points: torch.Tensor of shape (n_points, n_features)
        Returns:
            edge_dists: torch.Tensor of shape (n_edges,), distances of valid edges
            edge_pairs: torch.Tensor of shape (n_edges, 2), pairs of point indices
        """
        n_points = points.shape[0]
        if n_points <= 1:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device, dtype=torch.long)

        # Compute k-NN (k=50 neighbors per point)
        k = 50
        row, col = knn(points, points, k, batch_x=None, batch_y=None)
        # Remove duplicates (i < j) and self-loops
        mask = row < col
        edge_pairs = torch.stack([row[mask], col[mask]], dim=1)  # Shape: (n_edges, 2)
        edge_dists = torch.norm(points[row[mask]] - points[col[mask]], dim=-1)  # Shape: (n_edges,)
        # Filter by max_edge_length
        mask = edge_dists <= self.max_edge_length
        return edge_dists[mask], edge_pairs[mask]

    def _approximate_dim0(self, edge_dists, edge_pairs):
        """
        Approximate dimension 0 persistence (connected components) using a soft MST.
        Args:
            edge_dists: torch.Tensor of shape (n_edges,), distances of valid edges
            edge_pairs: torch.Tensor of shape (n_edges, 2), pairs of point indices
        Returns:
            diagram: torch.Tensor of shape (n_features, 3) with [birth, death, dim]
        """
        n_points = edge_pairs.max().item() + 1 if edge_pairs.numel() > 0 else 0
        if n_points == 0:
            return torch.tensor([], device=self.device).reshape(0, 3)

        # Initialize components
        births = torch.zeros(n_points, device=self.device)  # Birth at 0
        if edge_dists.numel() == 0:
            return torch.tensor([[0.0, 0.0, 0.0] for _ in range(n_points)], device=self.device)

        # Sort edges (differentiable approximation)
        edge_weights = torch.softmax(-edge_dists, dim=0)  # Shape: (n_edges,)
        sorted_indices = torch.argsort(edge_dists)

        # Initialize component assignments
        component = torch.arange(n_points, device=self.device, dtype=torch.long)
        deaths = []

        # Process edges
        for idx in sorted_indices:
            i, j = edge_pairs[idx]
            dist = edge_dists[idx]
            if component[i] != component[j]:
                new_comp = torch.min(component[i], component[j])
                old_comp_i = component[i]
                old_comp_j = component[j]
                # Create new component tensor
                new_component = component.clone()
                new_component[component == old_comp_i] = new_comp
                new_component[component == old_comp_j] = new_comp
                component = new_component
                deaths.append(dist)

        # Create diagram
        if not deaths:
            return torch.tensor([], device=self.device).reshape(0, 3)
        deaths = torch.stack(deaths)  # Shape: (n_deaths,)
        births = torch.zeros_like(deaths)
        dims = torch.zeros_like(deaths)
        diagram = torch.stack([births, deaths, dims], dim=1)  # Shape: (n_deaths, 3)
        diagram = diagram[diagram[:, 1] > diagram[:, 0]]  # Filter invalid
        return diagram

    def _approximate_dim1(self, edge_dists, edge_pairs):
        """
        Approximate dimension 1 persistence (loops) using a filtration-based approach.
        Args:
            edge_dists: torch.Tensor of shape (n_edges,), distances of valid edges
            edge_pairs: torch.Tensor of shape (n_edges, 2), pairs of point indices
        Returns:
            diagram: torch.Tensor of shape (n_features, 3) with [birth, death, dim]
        """
        n_points = edge_pairs.max().item() + 1 if edge_pairs.numel() > 0 else 0
        if n_points < 3:
            return torch.tensor([], device=self.device).reshape(0, 3)

        # Sort edges by distance for filtration
        sorted_indices = torch.argsort(edge_dists)
        edge_dists_sorted = edge_dists[sorted_indices]
        edge_pairs_sorted = edge_pairs[sorted_indices]

        # Initialize union-find for tracking connected components
        parent = torch.arange(n_points, device=self.device, dtype=torch.long)
        rank = torch.zeros(n_points, device=self.device, dtype=torch.long)

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return False
            if rank[pu] < rank[pv]:
                pu, pv = pv, pu
            parent[pv] = pu
            if rank[pu] == rank[pv]:
                rank[pu] += 1
            return True

        # Track cycles and their birth/death times
        diagram = []
        edge_index = 0
        n_edges = len(edge_dists_sorted)

        # Process edges in filtration order
        while edge_index < n_edges:
            current_dist = edge_dists_sorted[edge_index]
            cycle_edges = []
            # Collect all edges at the current distance (to handle simplicial complex)
            while edge_index < n_edges and abs(edge_dists_sorted[edge_index] - current_dist) < 1e-6:
                i, j = edge_pairs_sorted[edge_index]
                if not union(i, j):  # Edge creates a cycle
                    cycle_edges.append(edge_index)
                edge_index += 1

            # Assign death times for detected cycles
            for cycle_idx in cycle_edges:
                birth = edge_dists_sorted[cycle_idx]
                # Look ahead for a death time (next distinct distance)
                death_idx = edge_index
                while death_idx < n_edges and edge_dists_sorted[death_idx] <= birth:
                    death_idx += 1
                death = edge_dists_sorted[death_idx] if death_idx < n_edges else birth + 0.1
                if death > birth + 1e-2:  # Filter low-persistence cycles
                    diagram.append([birth, death, 1.0])

        # Convert to tensor and filter
        if not diagram:
            return torch.tensor([], device=self.device).reshape(0, 3)

        diagram = torch.tensor(diagram, device=self.device)
        # Filter by persistence
        persistence = diagram[:, 1] - diagram[:, 0]

        # weights = torch.sigmoid((persistence - 1e-2) * 100)  # Smooth transition
        # diagram = diagram * weights.unsqueeze(1)  # Weight features

        significant = persistence > 1e-2  # Stricter threshold
        diagram = diagram[significant]

        # Limit number of features to avoid over-detection
        if diagram.shape[0] > 50:  # Arbitrary cap based on expected circle features
            persistence = diagram[:, 1] - diagram[:, 0]
            _, top_indices = torch.topk(persistence, k=50, largest=True)
            diagram = diagram[top_indices]

        return diagram

    def fit_transform(self, X):
        """
        Compute persistence diagrams for a batch of point clouds.
        Args:
            X: torch.Tensor of shape (n_samples, n_points, n_features)
        Returns:
            diagrams: List of torch.Tensor, each of shape (n_features, 3) with [birth, death, dim]
        """
        diagrams = []
        for i in range(X.shape[0]):
            points = X[i]  # Shape: (n_points, n_features)
            edge_dists, edge_pairs = self._compute_distance_matrix(points)
            diagram = []
            if 0 in self.homology_dimensions:
                dim0 = self._approximate_dim0(edge_dists, edge_pairs)
                diagram.append(dim0)
            if 1 in self.homology_dimensions:
                dim1 = self._approximate_dim1(edge_dists, edge_pairs)
                diagram.append(dim1)
            if 2 in self.homology_dimensions:
                # Placeholder: no features for dimension 2
                diagram.append(torch.tensor([], device=self.device).reshape(0, 3))
            diagram = torch.cat(diagram, dim=0) if diagram else torch.tensor([], device=self.device).reshape(0, 3)
            diagrams.append(diagram)
        return diagrams

    def transform(self, X):
        """
        Transform point clouds to persistence diagrams.
        Args:
            X: torch.Tensor of shape (n_samples, n_points, n_features)
        Returns:
            diagrams: List of torch.Tensor, each of shape (n_features, 3)
        """
        return self.fit_transform(X)


def generate_circle(n_points=100, radius=1.0, noise=0.05):
    """
    Generate a 2D point cloud sampling a circle with noise.
    """
    theta = torch.linspace(0, 2 * np.pi, n_points)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=1)  # Shape: (n_points, 2)
    if noise > 0:
        points += torch.randn_like(points) * noise
    return points


def generate_clusters(n_points=100, n_clusters=3, spread=0.1):
    """
    Generate a 2D point cloud with multiple clusters.
    """
    centers = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])[:n_clusters]
    points = []
    for _ in range(n_points // n_clusters):
        for center in centers:
            point = center + torch.randn(2) * spread
            points.append(point)
    points = torch.stack(points)  # Shape: (n_points, 2)
    return points


class TestDifferentiableVietorisRipsPersistence(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vr = DifferentiableVietorisRipsPersistence(
            homology_dimensions=[0, 1], max_edge_length=0.4
        )
        self.vr_giotto = VietorisRipsPersistence(homology_dimensions=[0, 1])

    def test_circle_diagram(self):
        """Test diagram correctness for a circle (should have dimension 1 features)."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.05).to(self.device)
        points = points.unsqueeze(0)  # Shape: (1, n_points, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]  # Shape: (n_features, 3)

        # Check format
        self.assertEqual(diagram.shape[1], 3, "Diagram should have 3 columns: [birth, death, dim]")
        self.assertTrue(torch.all(diagram[:, 1] >= diagram[:, 0]), "Death times should be >= birth times")

        # Check dimension 1 features
        dim1_mask = diagram[:, 2] == 1
        self.assertGreater(torch.sum(dim1_mask).item(), 0, "Should have dimension 1 features for a circle")
        print(f"Circle diagram: {diagram.shape[0]} features, {torch.sum(dim1_mask).item()} in dim 1")

    def test_clusters_diagram(self):
        """Test diagram correctness for clusters (should have dimension 0 features)."""
        points = generate_clusters(n_points=90, n_clusters=3, spread=0.1).to(self.device)
        points = points.unsqueeze(0)  # Shape: (1, n_points, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]

        # Check format
        self.assertEqual(diagram.shape[1], 3, "Diagram should have 3 columns")
        self.assertTrue(torch.all(diagram[:, 1] >= diagram[:, 0]), "Death times should be >= birth times")

        # Check dimension 0 features
        dim0_mask = diagram[:, 2] == 0
        self.assertGreater(torch.sum(dim0_mask).item(), 0, "Should have dimension 0 features for clusters")
        print(f"Clusters diagram: {diagram.shape[0]} features, {torch.sum(dim0_mask).item()} in dim 0")

    def test_differentiability(self):
        """Test that gradients flow through the computation."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.05).to(self.device)
        points = points.unsqueeze(0).requires_grad_(True)  # Shape: (1, n_points, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]

        # Compute a dummy loss (sum of death times)
        loss = diagram[:, 1].sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(points.grad, "Points should have gradients")
        grad_norm = points.grad.norm().item()
        self.assertGreater(grad_norm, 0, "Gradient norm should be non-zero")
        print(f"Differentiability test: Gradient norm = {grad_norm}")

    def test_comparison_with_giotto(self):
        """Compare diagrams with giotto-tda's VietorisRipsPersistence."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.05)
        points_torch = points.unsqueeze(0).to(self.device)  # Shape: (1, n_points, 2)
        points_numpy = points.numpy()

        # Our implementation
        diagrams_torch = self.vr.fit_transform(points_torch)
        diagram_torch = diagrams_torch[0]

        # giotto-tda
        diagrams_giotto = self.vr_giotto.fit_transform(points_numpy[None])[0]  # Shape: (n_features, 3)
        diagram_giotto = torch.from_numpy(diagrams_giotto).float().to(self.device)  # Cast to float32

        # Compare dimension 1 features
        dim1_torch = diagram_torch[diagram_torch[:, 2] == 1, :2]
        dim1_giotto = diagram_giotto[diagram_giotto[:, 2] == 1, :2]
        if dim1_torch.shape[0] > 0 and dim1_giotto.shape[0] > 0:
            loss, _ = chamfer_distance(dim1_torch.unsqueeze(0), dim1_giotto.unsqueeze(0))
            self.assertLess(loss.item(), 1.0, "Dimension 1 diagrams should be similar")
            print(f"Chamfer distance (dim 1): {loss.item()}")

    def test_empty_input(self):
        """Test handling of empty point cloud."""
        points = torch.empty(1, 0, 2, device=self.device)  # Shape: (1, 0, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]
        self.assertEqual(diagram.shape[0], 0, "Empty input should produce empty diagram")
        print("Empty input test passed")

    def test_small_input(self):
        """Test handling of small point cloud."""
        points = torch.randn(1, 5, 2, device=self.device)  # Shape: (1, 5, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]
        self.assertEqual(diagram.shape[1], 3, "Small input should produce valid diagram")
        print(f"Small input diagram: {diagram.shape[0]} features")

    def test_noisy_input(self):
        """Test robustness to high noise."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.2).to(self.device)
        points = points.unsqueeze(0)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]
        dim1_mask = diagram[:, 2] == 1
        self.assertGreater(torch.sum(dim1_mask).item(), 0, "Noisy circle should still have dimension 1 features")
        print(f"Noisy input: {torch.sum(dim1_mask).item()} dim 1 features")


def test_on_simple_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)  # Transform point cloud
            self.vr = DifferentiableVietorisRipsPersistence(homology_dimensions=[0, 1, 2],
                                                            max_edge_length=2.0)  # Increased max_edge_length

        def forward(self, points):
            # points: (batch_size, n_points, 2)
            points = self.fc(points)  # Transform points
            diagrams = self.vr.fit_transform(points)  # Compute persistence diagrams
            # Aggregate diagrams (sum of birth and death times for all dimensions)
            batch_features = []
            for diagram in diagrams:
                if diagram.shape[0] == 0:
                    # Handle empty diagram with a differentiable default
                    feature = torch.tensor(0.0, device=points.device, requires_grad=True)
                else:
                    # Sum birth and death times to ensure gradient flow
                    feature = diagram[:, 0].sum() + diagram[:, 1].sum()  # Includes dim 0 and dim 1
                batch_features.append(feature)
            return torch.stack(batch_features)

    def generate_large_circle(n_points=100000, radius=1.0, noise=0.05):
        """
        Generate a 2D point cloud sampling a circle with noise.
        """
        theta = torch.linspace(0, 2 * np.pi, n_points)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        points = torch.stack([x, y], dim=1)  # Shape: (n_points, 2)
        if noise > 0:
            points += torch.randn_like(points) * noise
        return points

    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)

    # Normalize input to unit scale
    # Generate large point cloud
    points = generate_large_circle(n_points=50000, radius=1.0, noise=0.05).to(device)

    # Prepare input
    points = points.unsqueeze(0).requires_grad_(True)  # Shape: (1, n_subsample, 2)
    points = points / points.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # Normalize

    points = points.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Forward pass
    output = model(points)
    loss = output.sum()  # Dummy loss
    loss.backward()

    # Check gradients
    print("Model parameter gradients:", model.fc.weight.grad.norm().item())
    print("Input gradients:", points.grad.norm().item())


if __name__ == "__main__":
    # unittest.main(argv=[''], exit=False)

    test_on_simple_model()