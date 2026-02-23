
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def simulate_adni_data(num_patients=100, num_rois=90, seed=42):
    np.random.seed(seed)

    # Simulate labels: 0: Normal, 1: MCI, 2: AD
    labels = np.random.choice([0, 1, 2], size=num_patients)

    # Simulate ROI features (Thickness, Volume)
    # We'll make AD patients have lower thickness/volume in some regions (atrophy)

    # Base features
    # Shape: [num_patients, num_rois, 2]
    # Feature 0: Thickness (e.g., mean 2.5mm, std 0.3)
    # Feature 1: Volume (e.g., mean 2000mm3, std 500)

    features = np.zeros((num_patients, num_rois, 2))

    # Generate base data
    features[:, :, 0] = np.random.normal(2.5, 0.3, (num_patients, num_rois))
    features[:, :, 1] = np.random.normal(2000, 500, (num_patients, num_rois))

    # Apply atrophy to AD and MCI cases in specific ROIs (e.g., hippocampus-like regions)
    # Let's assume ROIs 0-10 are sensitive
    sensitive_rois = list(range(10))

    for i in range(num_patients):
        if labels[i] == 2: # AD
            features[i, sensitive_rois, 0] -= 0.5 # Thinner
            features[i, sensitive_rois, 1] -= 800 # Smaller volume
        elif labels[i] == 1: # MCI
            features[i, sensitive_rois, 0] -= 0.2
            features[i, sensitive_rois, 1] -= 300

    return features, labels

def build_structural_covariance_graph(features, threshold=0.1):
    """
    Builds a graph where edges represent structural covariance (correlation of thickness).
    Since structural covariance is a population property, we compute the correlation matrix
    across the population and use it to define the graph topology.
    """
    # Use Thickness (feature 0) for correlation
    thickness_data = features[:, :, 0] # [num_patients, num_rois]

    # Compute correlation matrix (transposed because np.corrcoef expects rows as variables)
    # We want correlation between ROIs, so we transpose thickness_data to be [num_rois, num_patients]
    corr_matrix = np.corrcoef(thickness_data.T)

    # Create a graph
    G = nx.Graph()
    num_rois = features.shape[1]
    G.add_nodes_from(range(num_rois))

    # Add edges based on threshold using vectorized operations for better performance.
    # We use the upper triangle of the correlation matrix to avoid duplicate edges and self-loops.
    i_indices, j_indices = np.triu_indices(num_rois, k=1)
    abs_corrs = np.abs(corr_matrix[i_indices, j_indices])

    # Filter by threshold
    mask = abs_corrs > threshold

    # G.add_weighted_edges_from expects an iterable of (u, v, w) tuples
    edges = zip(i_indices[mask], j_indices[mask], abs_corrs[mask])
    G.add_weighted_edges_from(edges)

    return G

def create_pyg_dataset(features, labels, G):
    """
    Converts patient features and labels into a list of PyTorch Geometric Data objects.

    Args:
        features (np.ndarray): Patient features of shape [num_patients, num_rois, num_features].
        labels (np.ndarray): Patient labels of shape [num_patients].
        G (nx.Graph): A NetworkX graph representing the shared brain topology.

    Returns:
        list: A list of torch_geometric.data.Data objects.
    """
    if len(features) != len(labels):
        raise ValueError(f"Features and labels must have the same length, but got {len(features)} and {len(labels)} respectively.")

    data_list = []

    # Convert NetworkX graph to edge_index (topology is shared)
    base_data = from_networkx(G)

    for i in range(len(labels)):
        # Node features for this patient: [num_rois, 2]
        x = torch.tensor(features[i], dtype=torch.float)
        y = torch.tensor(labels[i], dtype=torch.long)

        # Create Data object
        # We use edge_index and weight from the base_data
        edge_attr = None
        if hasattr(base_data, 'weight'):
             edge_attr = base_data.weight

        data = Data(x=x, edge_index=base_data.edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)

    return data_list

if __name__ == "__main__":
    print("Simulating ADNI data...")
    features, labels = simulate_adni_data(num_patients=100, num_rois=90)
    print(f"Generated features shape: {features.shape}")
    print(f"Generated labels shape: {labels.shape}")

    print("Constructing structural covariance graph...")
    # Using a low threshold because random noise correlations will be small
    G = build_structural_covariance_graph(features, threshold=0.15)
    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print("Converting to PyTorch Geometric dataset...")
    dataset = create_pyg_dataset(features, labels, G)
    print(f"Created dataset with {len(dataset)} patients.")

    # Inspect one sample
    sample = dataset[0]
    print("\nSample Data Object:")
    print(sample)
    print(f"x shape: {sample.x.shape}")
    print(f"edge_index shape: {sample.edge_index.shape}")
    if sample.edge_attr is not None:
        print(f"edge_attr shape: {sample.edge_attr.shape}")
    print(f"y: {sample.y}")
