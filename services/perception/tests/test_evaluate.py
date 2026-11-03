import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'flowmap'))
# Load by path under its own name: citystreet/ also has a cameras module.
for name, file in (('cameras', 'cameras.py'), ('scout_evaluate', 'evaluate.py')):
    spec = importlib.util.spec_from_file_location(name if name != 'cameras' else 'scout_cameras', ROOT / 'scout' / file)
    mod = importlib.util.module_from_spec(spec)
    if name == 'cameras':
        saved = sys.modules.get('cameras')
        sys.modules['cameras'] = mod
    spec.loader.exec_module(mod)
    if name == 'cameras':
        if saved is None:
            sys.modules.pop('cameras')
        else:
            sys.modules['cameras'] = saved
evaluate = mod


def arrow(x, y, dx, dy, rate):
    return np.array([x, y], float), np.array([dx, dy], float) / np.hypot(dx, dy), rate


class EvaluateTests(unittest.TestCase):
    def test_match_is_one_to_one(self):
        a = np.array([[0, 0], [1, 0]], float)
        b = np.array([[0.5, 0]], float)
        self.assertEqual(evaluate.match(a, b, 2), 1)

    def test_arrows_judged_by_place_direction_and_size(self):
        rate = np.zeros((1, 4, 8))
        rate[0, 2, 0] = 1.0  # answer key: weak eastward flow in cell 2
        field = {'cell_px': 20, 'bins': 8, 'rate': rate}
        truth = [arrow(5, 5, 1, 0, 4)]
        ours = [arrow(30, 6, 1, .1, 2),  # one cell over, same way: agrees
                arrow(45, 5, 1, 0, 1),  # no arrow nearby, but real weaker flow
                arrow(8, 8, -1, 0, 1),  # opposite
                arrow(75, 5, 0, 1, 1)]  # nothing there
        r = evaluate.compare_arrows(ours, truth, field, m=1.0, near_m=30)
        k = r['our_arrows_by_kind']
        self.assertEqual((k['agrees'], k['weaker_real_flow'], k['opposite'], k['no_flow_in_answer_key']), (1, 1, 1, 1))
        self.assertEqual(r['truth_arrows_with_our_arrow_nearby'], 1.0)

    def _lag_case(self, shift):
        """One walker at 1.2 m/s (12 px/frame at 0.1 m/px, identity camera);
        our track sits `shift` frames behind the answer key."""
        corners = [[0, 0], [400, 0], [400, 400], [0, 400]]
        b = {'meters_per_pixel': 0.1, 'cameras': [{'id': 'a', 'frame_points': corners, 'map_points': corners}]}
        tracks = {'a': {'1': [[f, 20 + 1.2 * (f - shift), 200] for f in range(shift, 200)]}}
        truth = {f: {7: np.array([20 + 1.2 * f, 200.0])} for f in range(200)}
        return evaluate.frame_lag(b, tracks, truth)

    def test_frame_lag_finds_offset(self):
        self.assertAlmostEqual(self._lag_case(0), 0.0)
        self.assertAlmostEqual(self._lag_case(3), 3.0)

    def test_frame_lag_without_matches(self):
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        b = {'meters_per_pixel': 0.1, 'cameras': [{'id': 'a', 'frame_points': corners, 'map_points': corners}]}
        self.assertIsNone(evaluate.frame_lag(b, {}, {}))

    def test_truth_paths_only_use_given_frames(self):
        raw = {f: {1: np.array([10.0 + f, 10.0])} for f in range(40)}
        covered = np.ones((50, 60), bool)
        full = evaluate.truth_paths(raw, covered, m=0.1)
        half = evaluate.truth_paths({f: p for f, p in raw.items() if f < 20}, covered, m=0.1)
        self.assertEqual(len(full[0]), 40)
        self.assertEqual(len(half[0]), 20)


if __name__ == '__main__':
    unittest.main()
