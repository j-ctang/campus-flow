import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scout'))
sys.path.insert(0, str(ROOT / 'flowmap'))
import bundle as bundle_io
import demo_map
import flow
import heatmap
import merge_cameras

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def camera_dir(root, cid, map_square, boxes, tracks):
    d = Path(root) / cid
    d.mkdir()
    (d / f'{cid}.mp4').write_bytes(b'')
    pts = [{'frame': f, 'map': m} for f, m in zip(SQUARE, map_square)]
    (d / 'bundle.json').write_text(json.dumps({
        'version': '0.1', 'map': {'image': '../map.png', 'scale': {'points': [[0, 0], [10, 0]], 'meters': 1}},
        'cameras': [{'id': cid, 'video': f'{cid}.mp4', 'frame_size': [100, 100], 'position': [0, 0], 'ground_points': pts}]}))
    (d / 'detections.json').write_text(json.dumps({cid: boxes}))
    (d / 'detections-tracks.json').write_text(json.dumps(tracks))
    return d


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        cv2.imwrite(str(root / 'map.png'), np.zeros((300, 300, 3), np.uint8))
        left = [[0, 0], [100, 0], [100, 100], [0, 100]]
        right = [[150, 0], [250, 0], [250, 100], [150, 100]]
        self.a = camera_dir(root, 'cam_a', left, {'0': [[45, 0, 55, 50]], '1': []}, {'t1': [[0, 50, 50]]})
        self.b = camera_dir(root, 'cam_b', right, {'0': [[45, 0, 55, 50]], '1': [[10, 0, 20, 30]]}, {'t2': [[1, 15, 30]]})
        self.out = root / 'merged'
        merge_cameras.merge(self.out, [self.a, self.b])

    def tearDown(self):
        self.tmp.cleanup()

    def test_paths_resolve_from_the_merged_folder(self):
        b = bundle_io.load(self.out / 'bundle.json')
        self.assertTrue(b['map_image'].resolve().exists())
        for cam in b['cameras']:
            self.assertTrue((self.out / cam['video']).resolve().exists(), cam['video'])

    def test_keeps_every_camera_detections_and_tracks(self):
        det = json.loads((self.out / 'detections.json').read_text())
        tracks = json.loads((self.out / 'tracks.json').read_text())
        self.assertEqual(sorted(det), ['cam_a', 'cam_b'])
        self.assertEqual(tracks, {'cam_a': {'t1': [[0, 50, 50]]}, 'cam_b': {'t2': [[1, 15, 30]]}})

    def test_heat_counts_people_from_each_camera(self):
        b = bundle_io.load(self.out / 'bundle.json')
        result = heatmap.run(b, json.loads((self.out / 'detections.json').read_text()), sigma_m=0.5)
        self.assertEqual(len(result['points']['0']), 2)  # one person per camera, separate ground
        self.assertEqual(len(result['points']['1']), 1)


    def test_flow_counts_walkers_from_each_camera(self):
        b = bundle_io.load(self.out / 'bundle.json')
        walk = lambda: [[f, 10 + 4 * f, 50] for f in range(25)]  # noqa: E731  # 80 px in 2.5 s, 8 m
        tracks = {'cam_a': {str(i): walk() for i in range(3)}, 'cam_b': {str(i): walk() for i in range(4)}}
        per_cam = {}
        for cid in ('cam_a', 'cam_b'):
            field = flow.flow_field(b, {cid: tracks[cid]}, fps=10, cell_m=4.0, duration_s=60)
            per_cam[cid] = flow.summary(field, min_count=1)['tracks_kept']
        both = flow.summary(flow.flow_field(b, tracks, fps=10, cell_m=4.0, duration_s=60), min_count=1)['tracks_kept']
        self.assertEqual(per_cam, {'cam_a': 3, 'cam_b': 4})
        self.assertEqual(both, 7)

    def test_command_line(self):
        out = Path(self.tmp.name) / 'cli'
        res = subprocess.run([sys.executable, str(ROOT / 'scout' / 'merge_cameras.py'), str(out), str(self.a), str(self.b)],
                             capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(res.stdout)['cameras'], ['cam_a', 'cam_b'])
        self.assertTrue((out / 'tracks.json').exists())


class FlowFramesTests(unittest.TestCase):
    def test_flow_tracks_keep_only_heat_frames(self):
        tracks = {'cam_a': {'t1': [[0, 1, 1], [5, 2, 2]], 't2': [[7, 3, 3]]}}
        self.assertEqual(demo_map.frames_only(tracks, ['0', '1']), {'cam_a': {'t1': [[0, 1, 1]]}})


if __name__ == '__main__':
    unittest.main()
