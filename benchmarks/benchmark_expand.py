import torch
import time
import sys
import os

def benchmark_op(num_points, in_dim, device, mode='repeat', iterations=1000, direct_device=False):
    # Warmup
    for _ in range(10):
        if mode == 'repeat':
            _ = torch.linspace(-1, 1, steps=num_points).unsqueeze(1).repeat(1, in_dim).to(device)
        else:
            if direct_device:
                _ = torch.linspace(-1, 1, steps=num_points, device=device).unsqueeze(1).expand(-1, in_dim)
            else:
                _ = torch.linspace(-1, 1, steps=num_points).unsqueeze(1).expand(-1, in_dim).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        if mode == 'repeat':
            x_eval = torch.linspace(-1, 1, steps=num_points).unsqueeze(1).repeat(1, in_dim).to(device)
        else:
            if direct_device:
                x_eval = torch.linspace(-1, 1, steps=num_points, device=device).unsqueeze(1).expand(-1, in_dim)
            else:
                x_eval = torch.linspace(-1, 1, steps=num_points).unsqueeze(1).expand(-1, in_dim).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start

if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("Torch not found. Theoretical justification provided instead.")
        sys.exit(0)

    num_points = 100
    in_dim = 90  # Typical number of ROIs
    device = 'cpu'
    iterations = 10000

    t_repeat = benchmark_op(num_points, in_dim, device, mode='repeat', iterations=iterations)
    t_expand = benchmark_op(num_points, in_dim, device, mode='expand', iterations=iterations, direct_device=False)
    t_expand_device = benchmark_op(num_points, in_dim, device, mode='expand', iterations=iterations, direct_device=True)

    print(f"--- Benchmark Results ({iterations} iterations) ---")
    print(f"1. Repeat (CPU -> Repeat -> Move): {t_repeat:.4f}s")
    print(f"2. Expand (CPU -> Expand -> Move): {t_expand:.4f}s")
    print(f"3. Expand (Direct Device):         {t_expand_device:.4f}s")

    improvement_expand = (t_repeat - t_expand) / t_repeat * 100
    improvement_total = (t_repeat - t_expand_device) / t_repeat * 100

    print(f"\nImprovement from 'expand' alone: {improvement_expand:.2f}%")
    print(f"Total improvement (expand + direct device): {improvement_total:.2f}%")

    # Memory check
    x_base = torch.linspace(-1, 1, steps=num_points).unsqueeze(1)
    x_repeat = x_base.repeat(1, in_dim)
    x_expand = x_base.expand(-1, in_dim)

    print(f"\n--- Memory Analysis ---")
    print(f"x_repeat storage size: {x_repeat.untyped_storage().size()} bytes")
    print(f"x_expand storage size: {x_expand.untyped_storage().size()} bytes")
    print(f"Memory saved: {x_repeat.untyped_storage().size() - x_expand.untyped_storage().size()} bytes")
