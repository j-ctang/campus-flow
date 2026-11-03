import importlib.util
import unittest
from pathlib import Path

# Load by path under its own name: citystreet/ also has a cameras module.
_spec = importlib.util.spec_from_file_location('scout_cameras', Path(__file__).resolve().parents[1] / 'scout' / 'cameras.py')
cameras = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cameras)


class LabelPlacementTests(unittest.TestCase):
    def test_close_cameras_get_separate_labels(self):
        shape = (400, 400, 3)
        centers = [(200, 200), (212, 205), (205, 218)]  # three co-located cameras
        taken = [(x - 11, y - 11, x + 11, y + 11) for x, y in centers]
        icons = list(taken)
        boxes = []
        for c in centers:
            cameras.place_label(c, 11, (60, 12), shape, taken)
            boxes.append(taken[-1])
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:] + icons:
                self.assertFalse(cameras._overlaps(a, b), (a, b))

    def test_label_stays_inside_image(self):
        taken = []
        x, y = cameras.place_label((395, 5), 11, (60, 12), (400, 400, 3), taken)
        box = taken[-1]
        self.assertGreaterEqual(box[0] + 3, 0)
        self.assertLess(box[2] - 3, 400)
        self.assertGreaterEqual(box[1] + 3, 0)


if __name__ == '__main__':
    unittest.main()
