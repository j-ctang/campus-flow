import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'citystreet'))
from scene import label_to_grid, grid_to_scene, GRID_SHAPE, SCENE_SHAPE, METERS_PER_GRID_CELL


class SceneTests(unittest.TestCase):
    def test_label_corner_maps_to_grid_corner(self):
        np.testing.assert_allclose(label_to_grid([[640, 768]]), [[160, 192]])

    def test_grid_cell_lands_at_official_offset(self):
        g = np.zeros(GRID_SHAPE)
        g[0, 0] = 1
        s = grid_to_scene(g)
        self.assertEqual(s.shape, SCENE_SHAPE)
        self.assertEqual(s.sum(), 4)
        self.assertTrue(np.all(s[10:12, 72:74] == 1))

    def test_rejects_wrong_grid_shape(self):
        with self.assertRaises(ValueError):
            grid_to_scene(np.zeros((10, 10)))

    def test_grid_cell_is_about_thirty_centimeters(self):
        self.assertAlmostEqual(METERS_PER_GRID_CELL, 0.305)


if __name__ == '__main__':
    unittest.main()
