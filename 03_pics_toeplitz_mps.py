"""03 - Non-Cartesian PICS on the Apple GPU with a Toeplitz normal operator.

Iterative reconstructions only ever need the normal operator A^H A. For a
NUFFT that is a convolution with the point-spread function, so it can be
applied with two FFTs on a 2x grid and no gridding at all (Toeplitz embedding):

    A^H A x = crop( IFFT( T * FFT( pad(x) ) ) ),   T = FFT(psf), computed once

FFTs are what GPUs do best, so every iteration is: coil maps -> FFT -> multiply
-> IFFT -> coil combine. We solve

    min_x 1/2 ||A S x - y||^2 + lambda ||W x||_1      (W: Haar wavelet, FISTA)

with the same code on the CPU and on the GPU (PyTorch MPS) and time both.
"""
import time

import numpy as np
import torch

from nufft import Plan, radial_trajectory
from phantom import coil_maps, shepp_logan

N, COILS, SPOKES, ITERS = 256, 12, 64, 60


class ToeplitzSense:
    """Normal operator S^H A^H A S for all coils at once, as a torch module-free object."""

    def __init__(self, psf, maps, device):
        self.dev = torch.device(device)
        self.T = torch.fft.fft2(torch.fft.ifftshift(torch.from_numpy(psf))).to(self.dev)   # (2N, 2N)
        self.S = torch.from_numpy(maps).to(self.dev)                                        # (C, N, N)
        self.n = maps.shape[-1]

    def normal(self, x):
        n = self.n
        cx = torch.nn.functional.pad(self.S * x, (0, n, 0, n))            # zero pad to 2N
        cx = torch.fft.ifft2(torch.fft.fft2(cx) * self.T)[..., :n, :n]
        return (self.S.conj() * cx).sum(0)


def haar(x, levels=4):
    x = x.clone(); n = x.shape[-1]
    for _ in range(levels):
        a = x[:n, :n]
        a = torch.cat([(a[0::2] + a[1::2]), (a[0::2] - a[1::2])], 0) / np.sqrt(2)
        a = torch.cat([(a[:, 0::2] + a[:, 1::2]), (a[:, 0::2] - a[:, 1::2])], 1) / np.sqrt(2)
        x[:n, :n] = a; n //= 2
    return x


def ihaar(x, levels=4):
    x = x.clone(); n = x.shape[-1] >> (levels - 1)
    for _ in range(levels):
        a, h = x[:n, :n], n // 2
        b = torch.empty_like(a); b[:, 0::2] = a[:, :h] + a[:, h:]; b[:, 1::2] = a[:, :h] - a[:, h:]
        c = torch.empty_like(b); c[0::2] = b[:h] + b[h:]; c[1::2] = b[:h] - b[h:]
        x[:n, :n] = c / 2; n *= 2
    return x


def soft(z, t):
    mag = z.abs()
    return z * torch.clamp(mag - t, min=0) / torch.clamp(mag, min=1e-12)


def fista(op, b, lam, iters, seed=0):
    """b = S^H A^H y. Step from a few power iterations on the normal operator."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(b.shape, dtype=torch.complex64, generator=g).to(b.device)
    for _ in range(15):
        v = op.normal(v); L = v.abs().pow(2).sum().sqrt(); v = v / L
    step = 1 / L
    x = torch.zeros_like(b); z = x.clone(); t = 1.0
    for _ in range(iters):
        r = z - step * (op.normal(z) - b)
        sy, sx = torch.randint(0, 16, (2,), generator=g).tolist()          # cycle spinning
        r = torch.roll(r, (sy, sx), (0, 1))
        x_new = torch.roll(ihaar(soft(haar(r), lam * step)), (-sy, -sx), (0, 1))
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + (t - 1) / t_new * (x_new - x)
        x, t = x_new, t_new
    return x


if __name__ == "__main__":
    img = shepp_logan(N).astype(np.complex64)
    maps = coil_maps(N, COILS).astype(np.complex64)
    k, _ = radial_trajectory(SPOKES, 2 * N)
    plan = Plan(k, N)
    print(f"radial {SPOKES} spokes (Nyquist ~{int(np.pi / 2 * N)}), {COILS} coils, {N}x{N}, {ITERS} FISTA iterations")

    # simulate data with the NUFFT (NumPy), add noise
    rng = np.random.default_rng(0)
    y = np.stack([plan.forward(m * img) for m in maps])
    y += 0.002 * np.abs(y).max() * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    b = np.sum(maps.conj() * np.stack([plan.adjoint(yc) for yc in y]), 0).astype(np.complex64)

    # point-spread function on the 2N grid: adjoint NUFFT of ones
    plan2 = Plan(k, 2 * N)
    psf = (plan2.adjoint(np.ones(len(k), np.complex64)) * (2 * N) / N ** 2).astype(np.complex64)

    # check the Toeplitz operator against the NUFFT normal operator
    x_test = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    ref = np.sum(maps.conj() * np.stack([plan.adjoint(plan.forward(m * x_test)) for m in maps]), 0)
    toe = ToeplitzSense(psf, maps, "cpu").normal(torch.from_numpy(x_test)).numpy()
    print(f"Toeplitz vs NUFFT normal operator: relative difference {np.linalg.norm(toe - ref) / np.linalg.norm(ref):.1e}\n")

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results = {}
    for dev in devices:
        op = ToeplitzSense(psf, maps, dev)
        bt = torch.from_numpy(b).to(dev)
        fista(op, bt, 0.0, 2)                                             # warm-up
        if dev == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        x = fista(op, bt, lam=0.002, iters=ITERS)
        if dev == "mps":
            torch.mps.synchronize()
        results[dev] = (time.perf_counter() - t0, x.cpu().numpy())
        print(f"  {dev:4s}  {results[dev][0]:6.2f} s   ({results[dev][0] / ITERS * 1e3:.1f} ms / iteration)")
    if "mps" in results:
        print(f"  speed-up MPS vs CPU: x{results['cpu'][0] / results['mps'][0]:.1f}")
        d = np.abs(results["mps"][1] - results["cpu"][1]).max() / np.abs(results["cpu"][1]).max()
        print(f"  max difference CPU vs MPS result: {d:.1e}")

    x = results[devices[-1]][1]
    x_adj = b
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, im, t in zip(ax, [img, x_adj, x], ["truth", f"adjoint, no DCF ({SPOKES} spokes)", f"PICS on {devices[-1].upper()}"]):
        im = np.abs(im); im = im / np.percentile(im, 99.5)
        a.imshow(im, cmap="gray", vmax=1); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig("figures/03_pics_toeplitz.png", dpi=110)
    print("saved figures/03_pics_toeplitz.png")
