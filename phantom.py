"""Shepp-Logan phantom and simulated coil sensitivities."""
import numpy as np


def shepp_logan(n=128):
    """Modified Shepp-Logan phantom (Toft's intensities), values in [0, 1]."""
    ellipses = [  # intensity, a, b, x0, y0, phi
        (1.0, .69, .92, 0, 0, 0), (-.8, .6624, .874, 0, -.0184, 0),
        (-.2, .11, .31, .22, 0, -18), (-.2, .16, .41, -.22, 0, 18),
        (.1, .21, .25, 0, .35, 0), (.1, .046, .046, 0, .1, 0), (.1, .046, .046, 0, -.1, 0),
        (.1, .046, .023, -.08, -.605, 0), (.1, .023, .023, 0, -.606, 0), (.1, .023, .046, .06, -.605, 0)]
    y, x = np.mgrid[1:-1:n * 1j, -1:1:n * 1j]
    img = np.zeros((n, n))
    for val, a, b, x0, y0, phi in ellipses:
        p = np.deg2rad(phi)
        xr = (x - x0) * np.cos(p) + (y - y0) * np.sin(p)
        yr = -(x - x0) * np.sin(p) + (y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += val
    return img


def coil_maps(n=128, nc=8, radius=1.3, seed=0):
    """Smooth, normalised birdcage-like coil sensitivities (nc, n, n), complex."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    maps = []
    for c in range(nc):
        ang = 2 * np.pi * c / nc
        cx, cy = radius * np.cos(ang), radius * np.sin(ang)
        mag = 1 / np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 0.3)
        phase = np.exp(1j * (ang + 0.6 * (x * np.cos(ang) + y * np.sin(ang)) + rng.uniform(0, 0.3)))
        maps.append(mag * phase)
    maps = np.array(maps)
    return maps / np.sqrt(np.sum(np.abs(maps) ** 2, axis=0))
