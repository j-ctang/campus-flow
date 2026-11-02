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
