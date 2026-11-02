"""CityStreet camera centers from the official Tsai calibration.

Rotation and translation values come from the official projection code
(projection_code/camera_proj_Zhang.py). World units are millimeters; negative z
is up. Dataset camera names: view1 = camera1, view2 = camera3, view3 = camera4.
"""
import numpy as np

VIEWS = {
    'view1': {'camera': 'camera1',
              'R': [[-0.03615359, 0.99596323, -0.0821594], [-0.34251474, 0.06488427, 0.93726927], [0.93881658, 0.06202646, 0.33878628]],
              'T': [1275.94012233, 226.52473842, 24230.24027107]},
    'view2': {'camera': 'camera3',
              'R': [[-0.99730688, 0.06324747, 0.03713147], [0.02439897, -0.19132768, 0.9812229], [0.06916415, 0.97948633, 0.18926923]],
              'T': [9680.47673912, 2273.39720387, 31436.76306171]},
    'view3': {'camera': 'camera4',
              'R': [[0.99545502, 0.07578782, -0.05766726], [0.0167508, 0.45675628, 0.88943415], [0.09374816, -0.88635766, 0.45341083]],
              'T': [3309.55016035, -4603.90350916, 29264.19379322]},
}
MM_PER_LABEL_CELL = 76.25
LABEL_OFFSET = (352 * 0.8, 522 * 0.8)  # official bbox offset, label grid units


def center_mm(view):
    v = VIEWS[view]
    return -np.array(v['R']).T @ np.array(v['T'])


def world_to_label(xy_mm):
    """World (x, y) mm -> 640x768 label grid (x, y), per the official readme."""
    return np.asarray(xy_mm, float) / MM_PER_LABEL_CELL + LABEL_OFFSET


def label_to_scene_original(label_xy):
    """Label grid -> pixel in the original 438x461 scene_map.jpg."""
    resized = np.asarray(label_xy, float) / 4 * 2 + (72, 10)
    return resized * (438 / 392, 461 / 412)
