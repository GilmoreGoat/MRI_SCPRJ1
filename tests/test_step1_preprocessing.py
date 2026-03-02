import numpy as np
import networkx as nx
import unittest
import torch
from step1_preprocessing import build_structural_covariance_graph, create_pyg_dataset

class TestStep1(unittest.TestCase):
    def test_build_graph_threshold_above_one(self):
        """Verify that a threshold above 1.0 results in no edges."""
        num_patients = 10
        num_rois = 5
        # Use deterministic data
        np.random.seed(42)
        features = np.random.rand(num_patients, num_rois, 2)

        # Threshold 1.1 is above any possible correlation (max 1.0)
        G = build_structural_covariance_graph(features, threshold=1.1)

        self.assertEqual(G.number_of_nodes(), num_rois)
        self.assertEqual(G.number_of_edges(), 0)

    def test_build_graph_threshold_at_one(self):
        """Verify that a threshold of exactly 1.0 results in no edges due to strict >."""
        num_patients = 10
        num_rois = 5
        features = np.zeros((num_patients, num_rois, 2))
        # Create perfect correlation
        patient_values = np.arange(num_patients)
        for i in range(num_rois):
            features[:, i, 0] = patient_values

        # Threshold 1.0. Correlation is exactly 1.0.
        # Logic is `abs(corr) > threshold`, so 1.0 > 1.0 is False.
        G = build_structural_covariance_graph(features, threshold=1.0)
        self.assertEqual(G.number_of_edges(), 0)

    def test_build_graph_threshold_negative(self):
        """Verify that a negative threshold results in a complete graph."""
        num_patients = 10
        num_rois = 3
        features = np.zeros((num_patients, num_rois, 2))
        patient_values = np.arange(num_patients)
        for i in range(num_rois):
            features[:, i, 0] = patient_values

        # Threshold -0.1. Any absolute correlation (>= 0) is > -0.1.
        G = build_structural_covariance_graph(features, threshold=-0.1)

        expected_edges = num_rois * (num_rois - 1) // 2
        self.assertEqual(G.number_of_edges(), expected_edges)

    def test_build_graph_perfect_correlation(self):
        """Verify that perfect correlation results in a complete graph with threshold < 1.0."""
        num_patients = 10
        num_rois = 4
        features = np.zeros((num_patients, num_rois, 2))
        patient_values = np.linspace(0, 1, num_patients)
        for i in range(num_rois):
            features[:, i, 0] = patient_values

        # Threshold 0.9. Correlation is 1.0.
        G = build_structural_covariance_graph(features, threshold=0.9)

        expected_edges = num_rois * (num_rois - 1) // 2
        self.assertEqual(G.number_of_edges(), expected_edges)

        # Verify weights are stored correctly
        for u, v, d in G.edges(data=True):
            self.assertAlmostEqual(d['weight'], 1.0, places=5)

    def test_build_graph_zero_correlation(self):
        """Verify that zero correlation results in no edges for a positive threshold."""
        # Create two ROIs with exactly zero correlation
        # ROI 0: [1, 0, -1, 0]
        # ROI 1: [0, 1, 0, -1]
        features_zero = np.zeros((4, 2, 2))
        features_zero[0, :, 0] = [1, 0]
        features_zero[1, :, 0] = [0, 1]
        features_zero[2, :, 0] = [-1, 0]
        features_zero[3, :, 0] = [0, -1]

        G = build_structural_covariance_graph(features_zero, threshold=0.01)
        self.assertEqual(G.number_of_edges(), 0)

    def test_create_pyg_dataset_length_mismatch(self):
        """Verify that a ValueError is raised when features and labels have different lengths."""
        features = np.random.rand(10, 5, 2)
        labels = np.random.randint(0, 3, size=9) # Length 9 instead of 10
        G = nx.Graph()

        with self.assertRaises(ValueError) as context:
            create_pyg_dataset(features, labels, G)

        self.assertIn("Features and labels must have the same length", str(context.exception))

    def test_create_pyg_dataset_success(self):
        """Verify that create_pyg_dataset returns a list of Data objects with correct properties."""
        num_patients = 5
        num_rois = 10
        num_features = 2
        features = np.random.rand(num_patients, num_rois, num_features).astype(np.float32)
        labels = np.random.randint(0, 3, size=num_patients)

        # Create a simple graph
        G = nx.Graph()
        G.add_nodes_from(range(num_rois))
        G.add_edge(0, 1, weight=0.5)

        dataset = create_pyg_dataset(features, labels, G)

        self.assertEqual(len(dataset), num_patients)
        for i in range(num_patients):
            data = dataset[i]
            # Check node features
            self.assertTrue(torch.equal(data.x, torch.tensor(features[i])))
            self.assertEqual(data.x.shape, (num_rois, num_features))
            # Check labels
            self.assertEqual(data.y.item(), labels[i])
            # Check edge_index
            self.assertEqual(data.edge_index.shape[0], 2)
            self.assertEqual(data.edge_index.shape[1], 2) # Undirected edge (0,1) and (1,0)
            # Check edge_attr (weights)
            self.assertIsNotNone(data.edge_attr)
            self.assertEqual(data.edge_attr.shape[0], 2)

if __name__ == '__main__':
    unittest.main()
