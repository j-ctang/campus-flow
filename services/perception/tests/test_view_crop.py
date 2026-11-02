import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'concentration'))
from view_crop import faces_in_view


class ViewCropTest(unittest.TestCase):
    # Camera at origin looking along +z (identity pose).
    R = np.eye(3)
    t = np.zeros(3)

    def triangle(self, cx, cy, z):
        return [[cx - .1, cy - .1, z], [cx + .1, cy - .1, z], [cx, cy + .1, z]]

    def test_keeps_triangle_under_selected_rays_and_drops_offview(self):
        vertices = np.array(self.triangle(0, 0, 5) + self.triangle(3, 3, 1), float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        keep = faces_in_view(vertices, faces, self.R, self.t, [[-.05, -.05], [.05, .05]])
        self.assertEqual(keep.tolist(), [True, False])

    def test_large_triangle_spanning_region_is_kept(self):
        vertices = np.array([[-10, -10, 5], [10, -10, 5], [0, 10, 5]], float)
        keep = faces_in_view(vertices, np.array([[0, 1, 2]]), self.R, self.t, [[0, 0], [.01, .01]])
        self.assertTrue(keep[0])

    def test_behind_camera_dropped_and_crossing_kept(self):
        vertices = np.array(self.triangle(0, 0, -5) + [[0, 0, -1], [0, .1, 2], [.1, 0, 2]], float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        keep = faces_in_view(vertices, faces, self.R, self.t, [[-.05, -.05], [.05, .05]])
        self.assertEqual(keep.tolist(), [False, True])


if __name__ == '__main__':
    unittest.main()
