"""Lossless view cropping: keep only triangles a selected ray could reach."""
import numpy as np


def faces_in_view(vertices, faces, R, t, normalized_bounds, margin=0.02):
    """Return a boolean face mask for rays inside normalized_bounds.

    Rays are straight in undistorted normalized camera coordinates. A triangle
    fully in front of the camera projects to the convex hull of its vertices,
    so a ray can only hit it if the ray lies inside that hull's bounding box.
    Triangles that cross the camera plane are kept, because their projection
    is unbounded. Triangles fully behind the camera cannot be hit.
    """
    camera = vertices @ np.asarray(R).T + np.asarray(t).ravel()
    z = camera[faces, 2]
    front = np.all(z > 1e-9, axis=1)
    crossing = np.any(z > 1e-9, axis=1) & ~front
    low, high = np.asarray(normalized_bounds, float)
    low, high = low - margin, high + margin
    keep = crossing.copy()
    idx = np.flatnonzero(front)
    projected = camera[faces[idx], :2] / camera[faces[idx], 2:3]
    tri_low = projected.min(axis=1)
    tri_high = projected.max(axis=1)
    overlap = np.all(tri_high >= low, axis=1) & np.all(tri_low <= high, axis=1)
    keep[idx[overlap]] = True
    return keep
