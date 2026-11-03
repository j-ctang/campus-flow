import json
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flowmap'))
import export as viewer_export
from test_flowmap import camera, write_bundle, SQUARE_FRAME


class ExportTests(unittest.TestCase):
    def test_layers_and_numbers_use_the_same_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME, height_m=8)])
            out = Path(tmp) / 'result'
            viewer_export.export_result(b, {'a': {'0': [[40, 0, 60, 50]], '1': []}}, out,
                                        fps=2, sigma_m=.5, north_deg=12)
            metadata = json.loads((out / 'map.json').read_text())
            self.assertEqual(metadata, {'m_per_px': .1, 'width': 200, 'height': 200, 'north_deg': 12})
            heat = cv2.imread(str(out / 'heat.png'), cv2.IMREAD_UNCHANGED)
            coverage = cv2.imread(str(out / 'coverage.png'), cv2.IMREAD_UNCHANGED)
            self.assertEqual(heat.shape, (200, 200, 4))
            self.assertEqual(int(heat[150, 150, 3]), 0)
            self.assertEqual(int(coverage[150, 150, 3]), 255)
            self.assertEqual(int(coverage[50, 50, 3]), 0)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['people_per_frame_mean'], .5)
            self.assertEqual(report['minutes'], 1 / 60)
            self.assertEqual(json.loads((out / 'flow.json').read_text()), [])
            cam = json.loads((out / 'cameras.json').read_text())[0]
            self.assertEqual(cam['people_per_frame_mean'], .5)
            self.assertEqual(cam['height_m'], 8)
            self.assertAlmostEqual(np.hypot(*cam['direction']), 1)

    def test_flow_preserves_pixel_direction_and_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME)])
            tracks = {'a': {'person': [[f, 10 + f * 2, 50] for f in range(30)]}}
            detections = {'a': {str(f): [] for f in range(30)}}
            out = Path(tmp) / 'result'
            viewer_export.export_result(b, detections, out, fps=10, tracks=tracks,
                                        cell_m=2, min_count=1)
            arrows = json.loads((out / 'flow.json').read_text())
            self.assertTrue(arrows)
            self.assertGreater(arrows[0]['dx'], .99)
            self.assertAlmostEqual(arrows[0]['dy'], 0)
            self.assertAlmostEqual(arrows[0]['people_per_min'], 20)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME)])
            out = Path(tmp) / 'result'
            out.mkdir()
            (out / 'keep.txt').write_text('previous result')
            with self.assertRaises(FileExistsError):
                viewer_export.export_result(b, {'a': {'0': []}}, out)
            self.assertEqual(list(out.iterdir()), [out / 'keep.txt'])

    def test_unknown_north_and_empty_people_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME)])
            out = Path(tmp) / 'result'
            viewer_export.export_result(b, {'a': {'0': []}}, out)
            self.assertIsNone(json.loads((out / 'map.json').read_text())['north_deg'])
            self.assertFalse(cv2.imread(str(out / 'heat.png'), -1)[..., 3].any())
            report = json.loads((out / 'report.json').read_text())
            self.assertIsNone(report['flow'])
            self.assertEqual(report['people_per_frame_max'], 0)

    def test_camera_ownership_counts_overlapping_views_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME), camera('b', SQUARE_FRAME)])
            detections = {c: {'0': [[40, 0, 60, 50]]} for c in ('a', 'b')}
            out = Path(tmp) / 'result'
            viewer_export.export_result(b, detections, out)
            cameras = json.loads((out / 'cameras.json').read_text())
            self.assertEqual([c['people_per_frame_mean'] for c in cameras], [1, 0])
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['people_per_frame_mean'], 1)
            self.assertAlmostEqual(sum(c['area_m2'] for c in cameras), report['covered_area_m2'])

    def test_failed_image_write_leaves_no_result_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = write_bundle(tmp, [camera('a', SQUARE_FRAME)])
            out = Path(tmp) / 'result'
            with patch.object(viewer_export.cv2, 'imwrite', return_value=False):
                with self.assertRaises(OSError):
                    viewer_export.export_result(b, {'a': {'0': []}}, out)
            self.assertFalse(out.exists())
            self.assertFalse(list(Path(tmp).glob('.result-*')))

    def test_rejects_invalid_smoothing_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result'
            for value in (0, -1, float('nan')):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    viewer_export.export_result({}, {}, out, sigma_m=value)
            self.assertFalse(out.exists())

    def test_cli_rejects_flat_tracks_with_multiple_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_bundle(tmp, [camera('a', SQUARE_FRAME), camera('b', SQUARE_FRAME)])
            root = Path(tmp)
            (root / 'detections.json').write_text(json.dumps({'a': {'0': []}, 'b': {'0': []}}))
            (root / 'tracks.json').write_text(json.dumps({'person': [[0, 10, 20]]}))
            result = subprocess.run([sys.executable, viewer_export.__file__, str(root / 'bundle.json'),
                                     str(root / 'detections.json'), str(root / 'result'),
                                     '--tracks', str(root / 'tracks.json')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('camera', result.stderr)
            self.assertFalse((root / 'result').exists())

    def test_rejects_invalid_timebase(self):
        with self.assertRaises(ValueError):
            viewer_export.export_result({}, {}, 'unused', fps=0)
