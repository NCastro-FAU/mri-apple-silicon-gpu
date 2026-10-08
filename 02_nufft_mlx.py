"""02 - Non-Cartesian NUFFT on the Apple GPU with MLX (and PyTorch MPS).

One Kaiser-Bessel plan (nufft.py), three back ends. We check accuracy against
an exact DFT, check the adjoint (<A x, y> = <x, A^H y>), and time forward +
adjoint for a multi-coil radial acquisition.
"""
import time

import mlx.core as mx
import numpy as np
import torch

from gpu_ops import NufftMLX, NufftTorch, sync
from nufft import Plan, exact_forward, radial_trajectory

N, COILS, SPOKES = 256, 16, 402
REPS = 5

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k, _ = radial_trajectory(SPOKES, 2 * N)
    t0 = time.perf_counter(); plan = Plan(k, N); t_plan = time.perf_counter() - t0
    print(f"radial: {SPOKES} spokes x {2 * N} samples = {len(k)} points, {COILS} coils, {N}x{N} image")
    print(f"plan (NumPy, once): {t_plan:.2f} s\n")

    # --- accuracy on a small subset (the exact DFT is slow)
    x1 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    sub = rng.choice(len(k), 400, replace=False)
    exact = exact_forward(x1, k[sub])
    mlx_op = NufftMLX(plan)
    approx = np.array(mlx_op.forward(mx.array(x1[None])))[0, sub]
    print(f"MLX NUFFT vs exact DFT: relative error {np.linalg.norm(approx - exact) / np.linalg.norm(exact):.1e}")

    x = (rng.standard_normal((COILS, N, N)) + 1j * rng.standard_normal((COILS, N, N))).astype(np.complex64)
    y = (rng.standard_normal((COILS, len(k))) + 1j * rng.standard_normal((COILS, len(k)))).astype(np.complex64)
    xm, ym = mx.array(x), mx.array(y)
    lhs = np.vdot(np.array(mlx_op.forward(xm)), y); rhs = np.vdot(x, np.array(mlx_op.adjoint(ym)))
    print(f"adjoint test |<Ax,y>/<x,A^H y> - 1| = {abs(lhs / rhs - 1):.1e}\n")

    # --- timing: forward + adjoint, all coils
    def timeit(fn, sync_fn=lambda: None):
        fn(); sync_fn()
        t = time.perf_counter()
        for _ in range(REPS):
            fn()
        sync_fn()
        return (time.perf_counter() - t) / REPS

    res = {}
    res["NumPy (CPU, coil loop)"] = timeit(lambda: [plan.adjoint(plan.forward(xc)) for xc in x])
    res["MLX (GPU)"] = timeit(lambda: mx.eval(mlx_op.adjoint(mlx_op.forward(xm))))
    if torch.backends.mps.is_available():
        top = NufftTorch(plan, "mps")
        xt = torch.from_numpy(x).to("mps")
        res["PyTorch MPS (GPU)"] = timeit(lambda: top.adjoint(top.forward(xt)), lambda: sync("mps"))
    base = res["NumPy (CPU, coil loop)"]
    print("forward + adjoint, all coils:")
    for name, t in res.items():
        print(f"  {name:24s} {t * 1e3:8.1f} ms   x{base / t:5.1f}")
