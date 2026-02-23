import torch
import torch.nn as nn
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from step1_preprocessing import simulate_adni_data, build_structural_covariance_graph, create_pyg_dataset
from step3_gkan_model import EdgeGKAN
from step2_kan_layer import coef2curve

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10, device='cpu'):
    """
    Standard PyTorch training loop for EdgeGKAN model.

    Args:
        model (nn.Module): The EdgeGKAN model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer (e.g., AdamW).
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        epochs (int): Number of training epochs.
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        list: Training loss history.
        list: Validation accuracy history.
    """
    model.to(device)
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            # Pass edge attributes explicitly
            outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(outputs, data.y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)
                _, predicted = torch.max(outputs.data, 1)
                total += data.y.size(0)
                correct += predicted.eq(data.y).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    return train_losses, val_accuracies

def get_symbolic_formula(kan_layer, device='cpu', prune_threshold=0.01):
    """
    Extracts symbolic mathematical formulas for the learned functions in a KAN layer.
    Prunes B-splines with low activation magnitude.
    """
    formulas = {}
    in_dim = kan_layer.in_dim
    out_dim = kan_layer.out_dim

    # Generate evaluation points in [-1, 1]
    # We evaluate all splines simultaneously
    num_points = 100
    x_eval = torch.linspace(-1, 1, steps=num_points).unsqueeze(1).repeat(1, in_dim).to(device) # [100, in_dim]

    with torch.no_grad():
        # Get spline output: [num_points, in_dim, out_dim]
        y_eval = coef2curve(x_eval, kan_layer.grid, kan_layer.coef, kan_layer.k, device=device)

        # Add base function contribution: scale_base * SiLU(x)
        base = kan_layer.base_fun(x_eval) # [num_points, in_dim]

        # Broadcast base to [num_points, in_dim, out_dim]
        y_base = kan_layer.scale_base[None, :, :] * base[:, :, None]

        # Combine: y_total = scale_sp * spline + base_part
        # Note: scale_sp is [in_dim, out_dim], need broadcasting
        y_total = y_eval * kan_layer.scale_sp[None, :, :] + y_base

    x_np = x_eval[:, 0].cpu().numpy() # [100]
    y_np = y_total.cpu().numpy() # [100, in_dim, out_dim]

    # Compute activation magnitude for pruning
    magnitudes = np.mean(np.abs(y_np), axis=0) # [in_dim, out_dim]

    for i in range(in_dim):
        for j in range(out_dim):
            # Prune if magnitude is below threshold
            if magnitudes[i, j] < prune_threshold:
                continue

            y_ij = y_np[:, i, j]

            # Simple symbolic regression heuristic
            # Try Linear: y = ax + b
            coeffs_lin = np.polyfit(x_np, y_ij, 1)
            p_lin = np.poly1d(coeffs_lin)
            mse_lin = np.mean((p_lin(x_np) - y_ij)**2)

            # Try Quadratic: y = ax^2 + bx + c
            coeffs_quad = np.polyfit(x_np, y_ij, 2)
            p_quad = np.poly1d(coeffs_quad)
            mse_quad = np.mean((p_quad(x_np) - y_ij)**2)

            # Decide based on MSE
            if mse_lin < 0.01 or mse_lin < mse_quad * 1.1:
                formula = f"{coeffs_lin[0]:.2f}*x + {coeffs_lin[1]:.2f}"
            else:
                formula = f"{coeffs_quad[0]:.2f}*x^2 + {coeffs_quad[1]:.2f}*x + {coeffs_quad[2]:.2f}"

            formulas[f"Feature_{i}->Hidden_{j}"] = formula

    return formulas

def get_top_connections(model, data, top_k=5):
    """
    Identifies the top K most influential edges (brain regions connections)
    for the model's prediction on a specific patient.
    """
    model.eval()

    # Clone data to avoid modifying original
    # We want to find importance of edge_attr (weights)
    # If edge_attr is None, we can't do this directly on weights.
    # But step1 creates graph with weights.

    if data.edge_attr is None:
        print("No edge attributes found. Cannot compute connection importance.")
        return []

    # Ensure edge_attr requires grad
    edge_attr = data.edge_attr.clone().detach().requires_grad_(True)
    x = data.x.clone().detach()
    edge_index = data.edge_index.clone().detach()
    batch = torch.zeros(x.size(0), dtype=torch.long) # Single graph batch

    # Reconstruct data object for model
    # Note: We need to pass edge_attr to the model.
    # GKANConv uses edge_index. It assumes edge_attr is not directly used in the current implementation
    # of GKANConv in step3?
    # Wait, let's check step3_gkan_model.py

    # GKANConv:
    # def forward(self, x, edge_index):
    #    ...
    #    return self.propagate(edge_index, x=x, norm=norm)
    # def message(self, x_j, norm):
    #    return norm.view(-1, 1) * x_j

    # It seems GKANConv in step3 does NOT use edge_attr/edge_weight!
    # It calculates 'norm' based on degree (structural) but ignores edge weights passed in edge_index or edge_attr?
    # "Step 2: Compute normalization. row, col = edge_index ... deg = degree(...)"
    # It effectively treats the graph as unweighted for the convolution aggregation,
    # EXCEPT if GCNConv usually uses edge_weight in normalization.
    # The custom implementation relies on `norm`.
    # `norm` is computed from degree.
    # So edge weights are IGNORED in the current GKANConv implementation!

    # This means gradients w.r.t edge_attr will be zero.
    # We can only interpret node feature importance (x).

    # If we want "brain-region connections", we might look at the node importance
    # and infer connections between important nodes?
    # Or we can compute gradients w.r.t. 'x' (features) -> which regions are important.

    # Let's switch to Node Importance (Region Importance).
    x.requires_grad_(True)

    # We need to manually construct a Data-like object or modify how we call the model
    # because model(data) extracts x, edge_index from data.

    # Simple wrapper to pass tensors
    # model.forward takes x, edge_index, edge_attr, batch

    output = model(x, edge_index, edge_attr, batch)

    # Target class score
    target_class = data.y.item()
    score = output[0, target_class]

    # Backward
    score.backward()

    # Get gradient for x
    # x shape: [num_rois, 2] (Thickness, Volume)
    # We want importance per ROI. Sum absolute gradients over features.
    grads = x.grad.abs().sum(dim=1) # [num_rois]

    # Top K regions
    top_values, top_indices = torch.topk(grads, top_k)

    top_regions = []
    for i in range(top_k):
        top_regions.append((top_indices[i].item(), top_values[i].item()))

    return top_regions

def interpret_kan(model, data, device='cpu'):
    print("\n--- KAN Interpretability ---")

    # 1. Symbolic Formulas (from the first layer)
    # The first layer transforms input features (Thickness, Volume) to hidden dimension.
    # This tells us how the model processes raw features.
    print("Extracting symbolic formulas from the first KAN layer...")
    first_layer = model.convs[0].kan
    formulas = get_symbolic_formula(first_layer, device=device)

    # Print a few examples
    print("Sample Formulas (Input Feature -> Hidden Node):")
    count = 0
    for key, val in formulas.items():
        if count < 5:
            print(f"  {key}: {val}")
            count += 1

    # 2. Top Brain Regions
    print("\nidentifying top brain regions contributing to classification for this patient...")
    top_regions = get_top_connections(model, data, top_k=5)
    print("Top 5 Important ROIs (Region Index, Importance Score):")
    for roi, score in top_regions:
        print(f"  ROI {roi}: {score:.4f}")

    return formulas, top_regions

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description="Train GKAN model on simulated ADNI data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=8, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GKAN layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_patients", type=int, default=50, help="Number of simulated patients")
    parser.add_argument("--device", type=str, default='auto', help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    # Setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Simulate Data
    print(f"Simulating Data (Patients: {args.num_patients})...")
    features, labels = simulate_adni_data(num_patients=args.num_patients, num_rois=90, seed=args.seed)
    G = build_structural_covariance_graph(features, threshold=0.1)
    dataset = create_pyg_dataset(features, labels, G)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize Model
    # Input: 2 features per node. Output: 3 classes (Normal, MCI, AD)
    # edge_dim=1 because we have 1 edge feature (structural covariance weight)
    model = EdgeGKAN(in_channels=2, hidden_channels=args.hidden_dim, out_channels=3, num_layers=args.num_layers, edge_dim=1)

    # 3. Train
    print("Starting Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_accs = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=args.epochs, device=device)

    # 4. Interpret
    if len(val_dataset) > 0:
        print("Running Interpretability on a validation sample...")
        sample_patient = val_dataset[0].to(device)
        interpret_kan(model, sample_patient, device=device)
    else:
        print("Validation set empty, skipping interpretability.")
