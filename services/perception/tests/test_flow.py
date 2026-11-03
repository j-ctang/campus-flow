import json, sys, tempfile, unittest
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flowmap'))
import bundle as bundle_io
import flow

FPS = 10.0


def make_bundle(tmp):
    """400x400 px map at 0.1 m/px; identity homography; camera sees it all."""
    cv2.imwrite(str(Path(tmp) / 'map.png'), np.zeros((400, 400, 3), np.uint8))
    corners = [[0, 0], [400, 0], [400, 400], [0, 400]]
    cam = {'id': 'a', 'position': [0, 0], 'frame_size': [400, 400],
           'ground_points': [{'frame': c, 'map': c} for c in corners]}
    path = Path(tmp) / 'bundle.json'
    path.write_text(json.dumps({'version': '0.1', 'map': {'image': 'map.png', 'scale': {'points': [[0, 0], [10, 0]], 'meters': 1}},
                                'cameras': [cam]}))
    return bundle_io.load(path)


def walk(start, velocity_px, seconds, t0=0):
    """Straight walk, one [frame, x, y] per frame."""
    n = int(seconds * FPS)
    return [[t0 + k, start[0] + velocity_px[0] * k, start[1] + velocity_px[1] * k] for k in range(n)]


def field(tracks, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        b = make_bundle(tmp)
        return flow.flow_field(b, {'a': tracks}, FPS, cell_m=2.0, duration_s=60, **kw)


class FlowTests(unittest.TestCase):
    def test_straight_walk_points_the_right_way(self):
        # 1.2 m/s east along y = 200 px for 10 s.
        f = field({str(i): walk([40, 200 + i], [1.2, 0], 10) for i in range(4)})
        items = flow.arrows(f, min_count=3)
        self.assertTrue(items)
        for _, d, _ in items:
            self.assertGreater(d[0], .95)

    def test_opposite_streams_are_both_kept(self):
        east = {f'e{i}': walk([40, 200], [1.2, 0], 20) for i in range(4)}
        west = {f'w{i}': walk([360, 200], [-1.2, 0], 20) for i in range(4)}
        items = flow.arrows(field({**east, **west}), min_count=3)
        dx = np.array([d[0] for _, d, _ in items])
        self.assertTrue((dx > .95).any() and (dx < -.95).any())
        row = [(c, d, r) for c, d, r in items if abs(c[0] - 210) < 1]
        self.assertEqual(sorted(round(r) for *_, r in row), [4, 4])

    def test_short_and_stationary_tracks_are_dropped(self):
        short = walk([40, 200], [1.2, 0], 1.0)  # 1 s
        still = [[k, 200 + np.sin(k) * 2, 200 + np.cos(k) * 2] for k in range(100)]  # jitter in place
        f = field({'short': short, 'still': still})
        self.assertEqual((f['tracks_kept'], f['tracks_dropped']), (0, 2))
        self.assertFalse(f['counts'].any())

    def test_rate_scales_with_count(self):
        def rate(n):
            items = flow.arrows(field({str(i): walk([40, 200], [1.2, 0], 10) for i in range(n)}), min_count=1)
            return max(r for *_, r in items)
        self.assertAlmostEqual(rate(6) / rate(3), 2.0)
        self.assertAlmostEqual(rate(3), 3.0)  # 3 people in 60 s

    def test_jump_splits_track(self):
        a = walk([40, 100], [1.2, 0], 3)
        b = walk([40, 300], [1.2, 0], 3, t0=30)  # 20 m away one frame later: ID swap
        f = field({'x': a + b})
        self.assertEqual(f['tracks_kept'], 2)
        self.assertFalse(f['counts'][..., 2].any() or f['counts'][..., 6].any())  # no north/south crossing

    def test_restarted_ids_are_joined(self):
        w = walk([40, 200], [1.2, 0], 20)
        f = field({'chunk0:1': w[:100], 'chunk100:1': w[100:], 'other': walk([40, 100], [1.2, 0], 3, t0=100)})
        self.assertEqual(f['tracks_kept'], 2)
        self.assertEqual(int(f['counts'].sum(2).max()), 1)  # no cell counts the walker twice

    def test_draw_flow_marks_image(self):
        f = field({str(i): walk([40, 200], [1.2, 0], 10) for i in range(4)})
        img = np.zeros((400, 400, 3), np.uint8)
        ref = flow.draw_flow(img, f, min_count=3)
        self.assertAlmostEqual(ref, 4.0)
        self.assertGreater(int((img == 255).all(2).sum()), 50)

    def test_flat_tracks_file_needs_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 't.json'
            p.write_text(json.dumps({'x': [[0, 1, 2]]}))
            self.assertEqual(flow.load_tracks(p, 'a'), {'a': {'x': [[0, 1, 2]]}})
            with self.assertRaises(ValueError):
                flow.load_tracks(p)


if __name__ == '__main__':
    unittest.main()
