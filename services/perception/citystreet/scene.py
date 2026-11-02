"""CityStreet ground-plane coordinates, following the official dataset readmes.

Label ground coordinates live on a 640 x 768 (w x h) grid. The density grid is
4x coarser (160 x 192). The official scene map is resized to 392 x 412 and the
2x-upscaled density grid is pasted at offset (2*36, 2*5).
"""
import numpy as np

LABEL_TO_GRID = 4           # 640x768 label grid -> 160x192 density grid
GRID_SHAPE = (192, 160)     # (h, w)
SCENE_SHAPE = (412, 392)    # (h, w) after the official resize
SCENE_OFFSET = (2 * 5, 2 * 36)  # (dy, dx) in scene pixels
MM_PER_LABEL_CELL = 76.25   # official resolution_scaler
METERS_PER_GRID_CELL = MM_PER_LABEL_CELL * LABEL_TO_GRID / 1000


def label_to_grid(points):
    """Label (cx, cy) on the 640x768 grid -> (x, y) on the 160x192 grid."""
    return np.asarray(points, float) / LABEL_TO_GRID


def grid_to_scene(grid_map):
    """Place a 192x160 grid map onto the 412x392 scene canvas (official padding)."""
    grid_map = np.asarray(grid_map)
    if grid_map.shape[:2] != GRID_SHAPE:
        raise ValueError(f'expected grid shape {GRID_SHAPE}, got {grid_map.shape[:2]}')
    up = np.repeat(np.repeat(grid_map, 2, axis=0), 2, axis=1)
    canvas = np.zeros(SCENE_SHAPE + grid_map.shape[2:], dtype=grid_map.dtype)
    dy, dx = SCENE_OFFSET
    canvas[dy:dy + up.shape[0], dx:dx + up.shape[1]] = up
    return canvas
