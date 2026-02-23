# Neuro-GKAN: Alzheimer's Classification with Graph Kolmogorov-Arnold Networks

This project implements a Graph Neural Network (GNN) enhanced with Kolmogorov-Arnold Network (KAN) layers for classifying Alzheimer's Disease (AD) stages (Normal, MCI, AD) from structural MRI data. It leverages the interpretability of KANs to identify symbolic formulas for feature transformations and key brain regions contributing to the classification.

## Project Overview

Traditional GNNs use linear transformations ($W x$). Neuro-GKAN replaces these with KAN layers, which use learnable B-spline activation functions on edges. This allows for:
1.  **Non-linear feature transformation** within the graph convolution.
2.  **Symbolic interpretability**: We can extract mathematical formulas representing the learned functions.
3.  **Explainability**: Identify which brain regions (ROIs) are most critical for the diagnosis.

## Dependencies

*   Python 3.8+
*   PyTorch
*   PyTorch Geometric (PyG)
*   NumPy
*   NetworkX
*   Nilearn (optional, for real MRI preprocessing)

## Installation

Ensure you have a Python environment set up. You can install the required packages using pip:

```bash
pip install torch numpy networkx
# Install PyTorch Geometric (see official docs for specific CUDA versions)
pip install torch_geometric
```

## File Structure

*   `step1_preprocessing.py`: Handles data simulation and graph construction. It generates synthetic cortical thickness and volume data for 90 brain regions (ROIs) and builds a structural covariance graph.
*   `step2_kan_layer.py`: Implements the `KANLinear` layer using B-splines. This is the core building block that replaces standard linear layers.
*   `step3_gkan_model.py`: Defines the `GKAN` model architecture, which stacks `GKANConv` layers and a final KAN classification head.
*   `step4_train_interpret.py`: The main script for training the model and running interpretability analysis.
*   `check_env.py`: Utility to check installed dependencies.

## Usage

### 1. Training on Simulated Data

You can run the full pipeline (simulation -> graph construction -> training -> interpretation) using `step4_train_interpret.py`. The script now supports command-line arguments for easy experimentation.

**Basic Usage:**

```bash
python step4_train_interpret.py
```

**Customizing Hyperparameters:**

You can adjust epochs, batch size, model dimensions, and more:

```bash
python step4_train_interpret.py --epochs 20 --batch_size 8 --hidden_dim 16 --lr 0.005
```

**Available Arguments:**

*   `--epochs`: Number of training epochs (default: 10)
*   `--batch_size`: Batch size (default: 4)
*   `--lr`: Learning rate (default: 0.01)
*   `--hidden_dim`: Hidden dimension size (default: 8)
*   `--num_layers`: Number of GKAN layers (default: 2)
*   `--seed`: Random seed for reproducibility (default: 42)
*   `--num_patients`: Number of simulated patients (default: 50)
*   `--device`: Device to run on ('cpu', 'cuda', or 'auto') (default: 'auto')

### 2. Output and Interpretability

After training, the script outputs:
1.  **Training Logs**: Loss and Validation Accuracy per epoch.
2.  **Symbolic Formulas**: Mathematical formulas extracted from the first KAN layer, showing how input features (Thickness, Volume) are transformed.
    *   *Example*: `Feature_0->Hidden_0: 0.52*x + 0.12`
3.  **Top Brain Regions**: Identifying the top 5 Regions of Interest (ROIs) that contributed most to the classification of a validation sample.

### 3. Using Real Data

To use real MRI data:

1.  **Prepare Data**: You need a dataset of shape `[num_patients, num_rois, num_features]` (e.g., shape `[N, 90, 2]` for 90 ROIs with Thickness and Volume).
2.  **Modify `step1_preprocessing.py`**:
    *   Replace `simulate_adni_data` with a function that loads your `.csv` or `.npy` files.
    *   Ensure the `build_structural_covariance_graph` uses your real feature correlations.
3.  **Update `step4_train_interpret.py`**:
    *   Call your new data loading function instead of `simulate_adni_data`.

## Interpretability Details

*   **Symbolic Regression**: The `get_symbolic_formula` function fits linear and quadratic polynomials to the activation patterns of the B-splines. It prunes connections with low magnitude to simplify the expressions.
*   **Node Importance**: We compute the gradient of the predicted class score with respect to the input node features. The magnitude of these gradients indicates which ROIs were most influential for the decision.
