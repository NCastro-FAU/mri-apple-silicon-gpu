"""A small Kaiser-Bessel NUFFT with one plan and three back ends: NumPy, PyTorch, MLX.

The plan (neighbour indices, interpolation weights, deapodisation) is built once
in NumPy. Applying it is just:

    forward : image / deapod -> zero-pad (2x) -> FFT -> gather J x J neighbours, weighted sum
    adjoint : weighted scatter-add onto the grid -> IFFT -> crop -> / deapod

Gather and scatter-add are exactly the operations GPUs are good at, so the same
plan runs on the Apple GPU through MLX (`take`, `.at[].add`) or PyTorch MPS
(`index_select`, `index_add_`).
"""
import numpy as np


class Plan:
    def __init__(self, k, n, J=6, osf=2.0):
        """k: (M, 2) trajectory in cycles/FOV, range [-0.5, 0.5). n: image size."""
        self.n, self.J = n, J
        self.G = G = int(np.ceil(osf * n / 2) * 2)
        beta = np.pi * np.sqrt((J / osf * (osf - 0.5)) ** 2 - 0.8)          # Beatty et al. 2005
        kg = k * G                                                           # grid units
        base = np.floor(kg - J / 2).astype(int) + 1                          # first neighbour
        offs = np.arange(J)
        idx, w = [], 1.0
        for d in range(2):
            pos = base[:, d, None] + offs                                    # (M, J)
            dist = kg[:, d, None] - pos
            arg = np.clip(1 - (2 * dist / J) ** 2, 0, None)
            w1 = np.i0(beta * np.sqrt(arg)) * (np.abs(dist) <= J / 2)
            idx.append(pos % G)                                              # FFT order
            w = w1 if d == 0 else w[:, :, None] * w1[:, None, :]
        self.idx = (idx[0][:, :, None] * G + idx[1][:, None, :]).reshape(len(k), -1).astype(np.int64)
        self.w = w.reshape(len(k), -1).astype(np.float32)
        # deapodisation: Fourier transform of the kernel at the image positions
        x = (np.arange(n) - n // 2) / G
        t = np.sqrt((np.pi * J * x) ** 2 - beta ** 2 + 0j)
        a = np.abs(np.where(np.abs(t) > 1e-8, np.sin(t) / np.where(t == 0, 1, t), 1).real)
        self.deapod = (np.outer(a, a) * J * J).astype(np.float32)
        self.scale = np.float32(1 / n)

    # ------------------------------------------------------------ numpy reference
    def forward(self, x):
        g = self._pad(x / self.deapod)
        G = np.fft.fft2(np.fft.ifftshift(g)).ravel()
        return (G[self.idx] * self.w).sum(1) * self.scale

    def adjoint(self, y):
        grid = np.zeros(self.G * self.G, np.complex64)
        np.add.at(grid, self.idx.ravel(), (y[:, None] * self.w).ravel())
        img = np.fft.fftshift(np.fft.ifft2(grid.reshape(self.G, self.G))) * self.G * self.G
        return self._crop(img) / self.deapod * self.scale

    def _pad(self, x):
        g = np.zeros((self.G, self.G), np.complex64)
        s = (self.G - self.n) // 2
        g[s:s + self.n, s:s + self.n] = x
        return g

    def _crop(self, g):
        s = (self.G - self.n) // 2
        return g[s:s + self.n, s:s + self.n]


def exact_forward(x, k):
    """Explicit DFT (slow, exact) to check the NUFFT."""
    n = x.shape[0]
    r = np.arange(n) - n // 2
    ex = np.exp(-2j * np.pi * np.outer(k[:, 0], r))
    ey = np.exp(-2j * np.pi * np.outer(k[:, 1], r))
    return np.einsum("mi,ij,mj->m", ex, x, ey) / n


def radial_trajectory(n_spokes, n_read, golden=True):
    ang = np.arange(n_spokes) * (np.deg2rad(111.246) if golden else np.pi / n_spokes)
    r = (np.arange(n_read) - n_read / 2) / n_read                     # [-0.5, 0.5)
    k = np.stack([np.outer(np.cos(ang), r), np.outer(np.sin(ang), r)], -1)
    return k.reshape(-1, 2), np.abs(r)[None].repeat(n_spokes, 0).ravel() + 0.5 / n_read
