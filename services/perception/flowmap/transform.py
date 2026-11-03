"""Camera pixels -> map pixels through a ground-plane homography.

Valid for flat ground only. Four or more marked point pairs fix the homography;
extra pairs are averaged by least squares.
"""
import cv2
import numpy as np


def fit(frame_points, map_points):
    H, _ = cv2.findHomography(np.asarray(frame_points, np.float64), np.asarray(map_points, np.float64), 0)
    if H is None:
        raise ValueError('ground points are degenerate (three or more on one line?)')
    return H


def apply(H, points):
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if not len(pts):
        return pts
    q = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def view_area(camera):
    """Map polygon a camera covers. Default: hull of its marked map points,
    which is the area the person marking the points vouched for."""
    if 'view_area' in camera:
        poly = apply(fit(camera['frame_points'], camera['map_points']), camera['view_area'])
    else:
        poly = np.asarray(camera['map_points'], float)
    return cv2.convexHull(np.float32(poly)).reshape(-1, 2)


def foot_points(boxes):
    """Bottom-center of each [x1, y1, x2, y2] person box."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])


HEAD_OFFSET = 0.08  # head point sits this fraction of box height below the box top
PERSON_HEIGHT_M = 1.75


def head_points(boxes):
    """Top-center of each [x1, y1, x2, y2] person box, moved down to the head.
    Heads stay visible in crowds where feet are hidden."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 1] + HEAD_OFFSET * (b[:, 3] - b[:, 1])])


def head_to_ground(ground_hits, camera_position, camera_height_m, person_height_m=PERSON_HEIGHT_M):
    """Standing spot of each person from where their head ray meets the ground.

    The ray from the camera (height H, above map point C) through a head at
    height h reaches the ground at G. The head, and so the person, is at
    C + (G - C) * (1 - h / H). Exact for a pinhole camera over flat ground."""
    g = np.asarray(ground_hits, float).reshape(-1, 2)
    c = np.asarray(camera_position, float)
    return c + (g - c) * (1 - person_height_m / camera_height_m)
