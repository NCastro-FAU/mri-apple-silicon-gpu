"""The NUFFT plan from nufft.py applied on the Apple GPU, with MLX and with PyTorch (MPS).

Both operate on a batch of coils at once: x is (C, n, n), y is (C, M).
Everything stays in complex64 / float32: the Apple GPU has no float64.
"""
import mlx.core as mx
import numpy as np
import torch


class NufftMLX:
    def __init__(self, plan):
        self.p = plan
        self.idx = mx.array(plan.idx.astype(np.int32))
        self.w = mx.array(plan.w)
        self.deapod = mx.array(plan.deapod)
        self.s = (plan.G - plan.n) // 2

    def forward(self, x):                                   # (C, n, n) -> (C, M)
        p, s = self.p, self.s
        g = mx.pad(x / self.deapod, [(0, 0), (s, s), (s, s)])
        g = mx.fft.fft2(mx.fft.ifftshift(g, axes=(-2, -1))).reshape(x.shape[0], -1)
        return (mx.take(g, self.idx, axis=1) * self.w).sum(-1) * p.scale

    def adjoint(self, y):                                   # (C, M) -> (C, n, n)
        p, s = self.p, self.s
        C = y.shape[0]
        vals = (y[:, :, None] * self.w).reshape(C, -1)
        flat = self.idx.reshape(-1)
        # MLX's GPU scatter has no complex64: scatter real and imaginary parts as float32
        re = mx.zeros((C, p.G * p.G)).at[:, flat].add(mx.real(vals))
        im = mx.zeros((C, p.G * p.G)).at[:, flat].add(mx.imag(vals))
        grid = re + 1j * im
        img = mx.fft.fftshift(mx.fft.ifft2(grid.reshape(C, p.G, p.G)), axes=(-2, -1)) * (p.G * p.G)
        return img[:, s:s + p.n, s:s + p.n] / self.deapod * p.scale


class NufftTorch:
    def __init__(self, plan, device="mps"):
        self.p, self.dev = plan, torch.device(device)
        self.idx = torch.from_numpy(plan.idx).to(self.dev)
        self.flat = self.idx.reshape(-1)
        self.w = torch.from_numpy(plan.w).to(self.dev)
        self.deapod = torch.from_numpy(plan.deapod).to(self.dev)
        self.s = (plan.G - plan.n) // 2

    def forward(self, x):
        p, s = self.p, self.s
        g = torch.nn.functional.pad(x / self.deapod, (s, s, s, s))
        g = torch.fft.fft2(torch.fft.ifftshift(g, dim=(-2, -1))).reshape(x.shape[0], -1)
        return (g[:, self.idx] * self.w).sum(-1) * p.scale

    def adjoint(self, y):
        p, s = self.p, self.s
        C = y.shape[0]
        vals = (y[:, :, None] * self.w).reshape(C, -1)
        # scatter-add of complex numbers: do real and imaginary parts as float32
        # (complex index_add_ is not available on every backend)
        re = torch.zeros(C, p.G * p.G, device=self.dev).index_add_(1, self.flat, vals.real.contiguous())
        im = torch.zeros(C, p.G * p.G, device=self.dev).index_add_(1, self.flat, vals.imag.contiguous())
        grid = torch.complex(re, im).reshape(C, p.G, p.G)
        img = torch.fft.fftshift(torch.fft.ifft2(grid), dim=(-2, -1)) * (p.G * p.G)
        return img[:, s:s + p.n, s:s + p.n] / self.deapod * p.scale


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
