
import torch
import torch.nn as nn
import numpy as np

def B_batch(x, grid, k=0, device='cpu'):
    """
    Computes B-spline basis functions B_i(x) of degree k for a batch of x values.
    """
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1, device=device)

        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + \
                (grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]

    value = torch.nan_to_num(value)
    return value

def coef2curve(x_eval, grid, coef, k, device='cpu'):
    """
    Computes the spline(x) = sum c_i B_i(x)
    """
    # x_eval: [batch, in_dim]
    # grid: [in_dim, num_intervals + 2k + 1]
    # coef: [in_dim, out_dim, num_intervals + k]

    b_splines = B_batch(x_eval, grid, k=k, device=device) # [batch, in_dim, num_coef]

    # coef: [in_dim, out_dim, num_coef]
    # b_splines: [batch, in_dim, num_coef]
    # We want result: [batch, out_dim, in_dim] -> sum over in_dim later?
    # Wait, KAN applies a spline per edge (in_dim -> out_dim).
    # The output should be [batch, out_dim, in_dim] before summation if we want to see individual contributions,
    # or just [batch, out_dim] after summation.

    # The reference implementation does:
    # y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    # i: batch, j: in_dim, k: num_coef (spline basis index)
    # coef indices: j: in_dim, l: out_dim, k: num_coef
    # result: i: batch, j: in_dim, l: out_dim

    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    return y_eval

def curve2coef(x_eval, y_eval, grid, k, device='cpu'):
    """
    Estimate spline coefficients c_i from known values y_eval.
    """
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1

    mat = B_batch(x_eval, grid, k, device=device)
    # mat: [batch, in_dim, n_coef]

    # Rearrange for lstsq
    # mat: [in_dim, out_dim, batch, n_coef] (expanded)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)

    # y_eval: [batch, in_dim, out_dim] -> [in_dim, out_dim, batch, 1]
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)

    try:
        # Solve Ax = b
        # A = mat, b = y_eval
        # solution shape: [in_dim, out_dim, n_coef, 1]
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except Exception as e:
        print(f'lstsq failed: {e}')
        coef = torch.zeros(in_dim, out_dim, n_coef).to(device)

    return coef

def extend_grid(grid, k_extend=0):
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid

class KANLinear(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=5, k=3, noise_scale=0.1,
                 scale_base=1.0, scale_sp=1.0, base_fun=nn.SiLU(),
                 grid_eps=0.02, grid_range=[-1, 1], device='cpu'):
        super(KANLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.k = k
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.device = device

        # Grid initialization
        # Shape: [in_dim, grid_size + 1]
        grid = torch.linspace(grid_range[0], grid_range[1], steps=grid_size + 1)[None, :].expand(in_dim, grid_size + 1)
        grid = extend_grid(grid, k_extend=k)
        self.register_buffer('grid', grid) # Non-trainable grid

        # Coefficients initialization
        noises = (torch.rand(grid_size + 1, in_dim, out_dim) - 0.5) * noise_scale / grid_size
        # We need an initial curve to fit coefficients to.
        # The reference uses curve2coef on the grid points with noise.
        # Let's just initialize coefficients randomly for simplicity,
        # or follow the reference which uses curve2coef to map noise to valid spline coefficients.

        # Using curve2coef to get coefficients that correspond to the noise (interpreted as function values)
        # grid shape: [in_dim, grid_total_points]
        # evaluating at grid points (excluding extended parts for fitting?)
        # The reference does: self.grid[:, k:-k].permute(1,0) as x_eval

        x_eval = self.grid[:, k:-k].permute(1, 0) # [grid_size+1, in_dim]
        # noises shape matches x_eval but with out_dim.
        # Actually noises in reference was [num+1, in_dim, out_dim].
        # curve2coef expects y_eval as [batch, in_dim, out_dim].

        self.coef = nn.Parameter(curve2coef(x_eval, noises, self.grid, k, device=device))

        # Scaling parameters
        self.scale_base = nn.Parameter(torch.ones(in_dim, out_dim) * scale_base / np.sqrt(in_dim))
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp / np.sqrt(in_dim))

        self.to(device)

    def forward(self, x):
        # x: [batch, in_dim]
        batch = x.shape[0]

        # 1. Base activation path (like a residual connection with nonlinearity)
        # The original paper/code adds a base function (SiLU) of input.
        base = self.base_fun(x) # [batch, in_dim]

        # 2. Spline path
        y = coef2curve(x, self.grid, self.coef, self.k, device=self.device) # [batch, in_dim, out_dim]

        # 3. Combine
        # y_combined = scale_base * base + scale_sp * spline
        # We need to broadcast base to out_dim
        # base: [batch, in_dim] -> [batch, in_dim, 1]

        y_base = self.scale_base[None, :, :] * base[:, :, None]
        y_sp = self.scale_sp[None, :, :] * y

        y_out = y_base + y_sp # [batch, in_dim, out_dim]

        # Sum over input dimension (like matrix multiplication summation)
        y_out = torch.sum(y_out, dim=1) # [batch, out_dim]

        return y_out

if __name__ == "__main__":
    print("Testing KANLinear Layer...")

    # Parameters
    batch_size = 10
    in_dim = 5
    out_dim = 3

    # Instantiate layer
    kan_layer = KANLinear(in_dim, out_dim, grid_size=5, k=3)
    print("KANLinear layer created.")

    # Create random input
    x = torch.randn(batch_size, in_dim)
    print(f"Input shape: {x.shape}")

    # Forward pass
    y = kan_layer(x)
    print(f"Output shape: {y.shape}")

    # Check shape
    expected_shape = (batch_size, out_dim)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
    print("Shape verification passed.")

    # Backward pass check
    target = torch.randn(batch_size, out_dim)
    loss = nn.MSELoss()(y, target)
    loss.backward()
    print("Backward pass successful. Gradients computed.")

    # Check if gradients exist
    print(f"Coefficient gradient norm: {kan_layer.coef.grad.norm().item()}")
    print(f"Scale base gradient norm: {kan_layer.scale_base.grad.norm().item()}")
