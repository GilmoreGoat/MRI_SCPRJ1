import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import torch
from step1_preprocessing import simulate_adni_data, build_structural_covariance_graph, create_pyg_dataset
from step3_gkan_model import EdgeGKAN
from step4_train_interpret import train_model, get_top_connections
from torch_geometric.loader import DataLoader

def generate_rois_coordinates(num_rois=90, seed=42):
    """
    Generates MNI coordinates for ROIs.
    ROIs 0-9 are placed in anatomically relevant locations (Hippocampus, etc.) for AD.
    Others are random.
    """
    np.random.seed(seed)
    coords = np.zeros((num_rois, 3))

    # Anatomical locations (approx MNI)
    # 0: Hippocampus_L
    coords[0] = [-25, -12, -18]
    # 1: Hippocampus_R
    coords[1] = [25, -12, -18]
    # 2: Amygdala_L
    coords[2] = [-22, -2, -18]
    # 3: Amygdala_R
    coords[3] = [22, -2, -18]
    # 4: Parahippocampal_L
    coords[4] = [-22, -25, -18]
    # 5: Parahippocampal_R
    coords[5] = [22, -25, -18]
    # 6: Entorhinal_L (approx, inferior temporal)
    coords[6] = [-18, -10, -25]
    # 7: Entorhinal_R
    coords[7] = [18, -10, -25]
    # 8: Fusiform_L
    coords[8] = [-30, -40, -18]
    # 9: Fusiform_R
    coords[9] = [30, -40, -18]

    # Generate random coordinates for the rest within a brain-like box
    # Brain bounds approx: x[-70, 70], y[-100, 60], z[-50, 80]
    for i in range(10, num_rois):
        coords[i] = [
            np.random.uniform(-60, 60),
            np.random.uniform(-90, 50),
            np.random.uniform(-40, 70)
        ]

    return coords

def main():
    print("Step 5: Visualization")

    # 1. Simulate Data
    print("Simulating Data...")
    features, labels = simulate_adni_data(num_patients=50, num_rois=90, seed=42)
    G = build_structural_covariance_graph(features, threshold=0.1)
    dataset = create_pyg_dataset(features, labels, G)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 2. Train Model
    print("Training Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeGKAN(in_channels=2, hidden_channels=8, out_channels=3, num_layers=2, edge_dim=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10, device=device)

    # 3. Get Top 5 ROIs
    print("Identifying Top 5 ROIs...")
    if len(val_dataset) > 0:
        # Use the first validation patient
        patient_data = val_dataset[0].to(device)
        top_regions = get_top_connections(model, patient_data, top_k=5)

        top_roi_indices = [roi for roi, score in top_regions]
        print(f"Top 5 ROIs: {top_roi_indices}")

        # 4. Visualization
        print("Generating Glass Brain Plot...")

        coords = generate_rois_coordinates(num_rois=90, seed=42)

        # Construct adjacency matrix for visualization
        # Initialize with zeros
        adj_matrix = np.zeros((90, 90))

        # Highlight connections between Top 5 ROIs if they exist
        has_edges = False

        # We need to map G edges to adj_matrix
        # G is a NetworkX graph with integer nodes 0..89

        # Check if there are edges between any pair of Top 5 ROIs
        for i in top_roi_indices:
            for j in top_roi_indices:
                if i < j:
                    if G.has_edge(i, j):
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0
                        has_edges = True

        # If no internal edges, show edges to strongest neighbors
        if not has_edges:
            print("No direct connections between Top 5 ROIs found in graph. Adding connections to strongest neighbors.")
            for i in top_roi_indices:
                neighbors = list(G.neighbors(i))
                if neighbors:
                    # Pick neighbor with highest weight? Or just first
                    # G has 'weight' attribute on edges

                    # Sort neighbors by weight
                    sorted_neighbors = sorted(neighbors, key=lambda n: G[i][n]['weight'], reverse=True)

                    if sorted_neighbors:
                        best_neighbor = sorted_neighbors[0]
                        adj_matrix[i, best_neighbor] = 1.0
                        adj_matrix[best_neighbor, i] = 1.0

        # Prepare node colors and sizes
        # Default: small blue dots
        node_color = ['blue'] * 90
        node_size = [10] * 90

        # Top 5: larger red dots
        for idx in top_roi_indices:
            node_color[idx] = 'red'
            node_size[idx] = 100  # Larger size for emphasis

        output_file = "gkan_top5_rois.png"

        # Plot connectome
        # node_color can be a list of colors
        # But plot_connectome expects 'node_color' as a list of strings

        plotting.plot_connectome(
            adjacency_matrix=adj_matrix,
            node_coords=coords,
            node_color=node_color,
            node_size=node_size,
            edge_cmap='Reds', # Red edges
            edge_vmin=0.1,    # Minimum edge value to show
            edge_vmax=1.0,
            edge_threshold=0.5, # Only show strong edges if we set values to 1.0
            display_mode='lzry', # Left, Z (axial), Right, Y (coronal) views
            output_file=output_file,
            title="Top 5 ROIs & Degraded Connections (Simulated)"
        )

        print(f"Plot saved to {output_file}")
    else:
        print("Validation dataset is empty. Cannot identify Top 5 ROIs.")

if __name__ == "__main__":
    main()
