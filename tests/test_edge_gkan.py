
import sys
import os
import torch
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from step1_preprocessing import simulate_adni_data, build_structural_covariance_graph, create_pyg_dataset
from step3_gkan_model import GKAN

def test_gkan_forward():
    print("Testing GKAN Forward Pass...")

    # Simulate minimal data
    num_patients = 2
    num_rois = 10
    features, labels = simulate_adni_data(num_patients=num_patients, num_rois=num_rois, seed=42)
    G = build_structural_covariance_graph(features, threshold=0.1)
    dataset = create_pyg_dataset(features, labels, G)

    data = dataset[0]

    # Initialize model with edge_dim=1 (as per our new architecture)
    try:
        model = GKAN(in_channels=2, hidden_channels=8, out_channels=3, num_layers=2, edge_dim=1)
        print("Model initialized with edge_dim=1.")
    except Exception as e:
        print(f"Model init failed: {e}")
        raise e

    # Forward pass
    try:
        out = model(data)
        print(f"Forward pass successful. Output shape: {out.shape}")
        assert out.shape == (1, 3)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_gkan_forward()
