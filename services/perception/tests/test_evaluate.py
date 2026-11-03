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

    def test_arrows_match_same_cell_and_direction(self):
        truth = [arrow(5, 5, 1, 0, 4), arrow(25, 5, 0, 1, 2)]
        ours = [arrow(6, 6, 1, .2, 3), arrow(26, 6, 0, -1, 2), arrow(45, 5, 1, 0, 1)]
        r = evaluate.compare_arrows(ours, truth, cell_px=20)
        self.assertEqual(r['recall'], .5)  # second truth arrow points the other way
        self.assertAlmostEqual(r['precision'], 1 / 3)

    def test_best_lag_finds_offset(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 9, 300)
        truth = {f: np.zeros((base[f], 2)) for f in range(300)}
        ours = {f: np.zeros((base[f - 3] if f >= 3 else 0, 2)) for f in range(300)}
        self.assertEqual(evaluate.best_lag(ours, truth, list(range(300))), 3)


if __name__ == '__main__':
    unittest.main()
