import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'citystreet'))
from cameras import center_mm, world_to_label, label_to_scene_original

# Camera arrows drawn on the official scene_map.jpg (original pixels).
ARROWS = {'view1': (70, 240), 'view2': (280, 20), 'view3': (195, 440)}


class CameraTests(unittest.TestCase):
    def test_centers_land_on_official_scene_arrows(self):
        for view, arrow in ARROWS.items():
            p = label_to_scene_original(world_to_label(center_mm(view)[:2]))
            self.assertLess(np.hypot(*(p - arrow)), 20, view)

    def test_cameras_sit_eight_to_nine_meters_up(self):
        for view in ARROWS:
            self.assertTrue(8000 < -center_mm(view)[2] < 9500, view)


if __name__ == '__main__':
    unittest.main()
