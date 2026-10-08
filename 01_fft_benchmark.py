"""01 - Multi-coil 2D FFT: NumPy vs PyTorch CPU vs PyTorch MPS vs MLX.

The simplest MRI reconstruction (Cartesian, fully sampled) is one centred
inverse FFT per coil and frame. We time a batch of frames x coils.

Two rules for honest GPU timing:
  * GPU calls are asynchronous: synchronise before stopping the clock
    (torch.mps.synchronize(), mx.eval()).
  * The first call compiles kernels: warm up before timing.
"""
import time

import mlx.core as mx
import numpy as np
import torch

FRAMES, COILS, N = 20, 16, 256
REPS = 5


def bench(fn, sync=lambda: None):
    fn(); sync()                                           # warm-up
    t = time.perf_counter()
    for _ in range(REPS):
        fn()
    sync()
    return (time.perf_counter() - t) / REPS


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = (rng.standard_normal((FRAMES * COILS, N, N)) + 1j * rng.standard_normal((FRAMES * COILS, N, N))).astype(np.complex64)
    print(f"{FRAMES} frames x {COILS} coils x {N}x{N}, complex64\n")

    ax = (-2, -1)
    np_fn = lambda: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=ax)), axes=ax)
    res = {"NumPy (CPU)": bench(np_fn)}

    kt = torch.from_numpy(k)
    tf = lambda x: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=ax)), dim=ax)
    res["PyTorch CPU"] = bench(lambda: tf(kt))
    if torch.backends.mps.is_available():
        km = kt.to("mps")
        res["PyTorch MPS (GPU)"] = bench(lambda: tf(km), torch.mps.synchronize)

    kx = mx.array(k)
    mf = lambda: mx.eval(mx.fft.fftshift(mx.fft.ifft2(mx.fft.ifftshift(kx, axes=ax)), axes=ax))
    res["MLX (GPU)"] = bench(mf)

    # check that all agree
    ref = np_fn()
    err = np.abs(np.array(mx.fft.fftshift(mx.fft.ifft2(mx.fft.ifftshift(kx, axes=ax)), axes=ax)) - ref).max() / np.abs(ref).max()
    base = res["NumPy (CPU)"]
    for name, t in res.items():
        print(f"  {name:20s} {t * 1e3:8.1f} ms   x{base / t:5.1f}")
    print(f"\n  max relative difference MLX vs NumPy: {err:.1e}")
    print("  (the transfer to the GPU is not timed: in a real pipeline the data stay there)")
