import json, sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flowmap'))
import bundle as bundle_io
import transform
import heatmap

SQUARE_FRAME = [[0, 0], [100, 0], [100, 100], [0, 100]]


def camera(cid, map_pts, **extra):
    return {'id': cid, 'position': [0, 0], 'frame_size': [100, 100],
            'ground_points': [{'frame': f, 'map': m} for f, m in zip(SQUARE_FRAME, map_pts)], **extra}


def write_bundle(tmp, cameras):
    import cv2
    cv2.imwrite(str(Path(tmp) / 'map.png'), np.zeros((200, 200, 3), np.uint8))
    path = Path(tmp) / 'bundle.json'
    path.write_text(json.dumps({'version': '0.1', 'map': {'image': 'map.png', 'scale': {'points': [[0, 0], [10, 0]], 'meters': 1}},
                                'cameras': cameras}))
    return bundle_io.load(path)


class TransformTests(unittest.TestCase):
    def test_four_pairs_recover_exact_homography(self):
        H = np.array([[1.2, .1, 5], [-.05, .9, 7], [.001, .002, 1]])
        frame = np.array(SQUARE_FRAME, float)
        fitted = transform.fit(frame, transform.apply(H, frame))
        np.testing.assert_allclose(transform.apply(fitted, [[37, 61]]), transform.apply(H, [[37, 61]]), atol=1e-6)

    def test_foot_is_box_bottom_center(self):
        np.testing.assert_allclose(transform.foot_points([[10, 20, 30, 80]]), [[20, 80]])

    def test_head_is_near_box_top_center(self):
        np.testing.assert_allclose(transform.head_points([[10, 20, 30, 120]]), [[20, 28]])

    def test_head_ray_ground_hit_pulls_back_toward_camera(self):
        # Camera 10 m up above map (0, 0); head ray hits ground at (100, 0).
        # A 1.75 m head on that ray stands at 100 * (1 - 1.75 / 10) = 82.5.
        np.testing.assert_allclose(transform.head_to_ground([[100, 0]], [0, 0], 10, 1.75), [[82.5, 0]])

    def test_head_mode_needs_camera_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME)])
            with self.assertRaises(ValueError):
                heatmap.run(b, {'a': {'0': []}}, sigma_m=0.5, point='head')


class BundleTests(unittest.TestCase):
    def test_scale_in_meters_per_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME)])
        self.assertAlmostEqual(b['meters_per_pixel'], 0.1)

    def test_rejects_fewer_than_four_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = camera('a', SQUARE_FRAME)
            bad['ground_points'] = bad['ground_points'][:3]
            with self.assertRaises(bundle_io.BundleError):
                write_bundle(tmp, [bad])

    def test_reads_camera_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME, height_m=8.5)])
        self.assertEqual(b['cameras'][0]['height_m'], 8.5)

    def test_rejects_duplicate_camera_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(bundle_io.BundleError):
                write_bundle(tmp, [camera('a', SQUARE_FRAME), camera('a', SQUARE_FRAME)])


class HeatmapTests(unittest.TestCase):
    def test_overlapping_cameras_count_a_person_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME), camera('b', SQUARE_FRAME)])
            same_person = [[45, 0, 55, 50]]
            result = heatmap.run(b, {'a': {'0': same_person}, 'b': {'0': same_person}}, sigma_m=0.5)
        self.assertEqual(len(result['points']['0']), 1)

    def test_heat_mass_equals_people_per_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', [[0, 0], [200, 0], [200, 200], [0, 200]])])
            boxes = [[40, 0, 60, 50], [10, 0, 20, 30]]
            result = heatmap.run(b, {'a': {'0': boxes, '1': []}}, sigma_m=0.5)
        people = result['heat'].sum() * b['meters_per_pixel'] ** 2
        self.assertAlmostEqual(people, 1.0, places=2)  # 2 people, 2 frames -> mean 1

    def test_head_mode_maps_head_to_standing_spot(self):
        # Identity homography, camera 10 m up at map (0, 0). Head ray hits
        # ground at x=100 (box top 92 + 8% of 100 px = 100); person at 82.5.
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', [[0, 0], [100, 0], [100, 100], [0, 100]], height_m=10)])
            b['cameras'][0]['frame_points'] = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], float)
            b['cameras'][0]['map_points'] = b['cameras'][0]['frame_points'].copy()
            result = heatmap.run(b, {'a': {'0': [[90, 92, 110, 192]]}}, sigma_m=0.5, point='head')
        np.testing.assert_allclose(result['points']['0'], [[82.5, 82.5]])

    def test_only_frames_every_camera_has(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME), camera('b', [[100, 100], [200, 100], [200, 200], [100, 200]])])
            result = heatmap.run(b, {'a': {'0': [], '1': []}, 'b': {'1': []}}, sigma_m=0.5)
        self.assertEqual(result['frames'], ['1'])


if __name__ == '__main__':
    unittest.main()
