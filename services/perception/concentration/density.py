"""Small experimental occupancy operations; no metric density claims."""
import numpy as np


def kernel_grid(points, mask, sigma):
    """Unit discrete mass per accepted observation, clipped to eligible cells."""
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    yy, xx = np.indices(mask.shape)
    grid = np.zeros(mask.shape, dtype=float)
    for x, y in points:
        kernel = np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))*mask
        mass = kernel.sum()
        if mass:
            grid += kernel/mass
    return grid


def average_window(maps, times, start, end):
    """Mean of uniformly spaced preview frames, including empty frames."""
    if end <= start:
        raise ValueError('window must have positive duration')
    times = np.asarray(times)
    selected = (times >= start) & (times < end)
    if not np.any(selected):
        raise ValueError('no processed frames in window')
    return np.mean(np.asarray(maps)[selected], axis=0)


def choose_hit(hits, normals, depths):
    """Never look through a rejected nearest surface to select convenient ground."""
    hits, normals, depths = map(np.asarray, (hits, normals, depths))
    keep = np.flatnonzero(depths > 1e-6)
    if not len(keep):
        return None, 'no_surface'
    keep = keep[np.argsort(depths[keep])]
    first = keep[0]
    if abs(normals[first, 2]) < .9:
        return None, 'nonhorizontal_surface'
    # Grazing rays cross stacked ground layers far apart along the ray but at
    # nearly the same height. Only a height difference means another elevation;
    # this matches the 0.05 height-span rule of the interior support raster.
    if any(abs(hits[i][2]-hits[first][2]) > .05 for i in keep[1:]):
        return None, 'multiple_surfaces'
    return hits[first].tolist(), None
