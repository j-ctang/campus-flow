"""Merge per-camera SCOUT outputs into one multi-camera bundle.

Each input folder holds bundle.json (from make_bundle.py), detections.json
and detections-tracks.json (from sam_tracks.py). The cameras of one sequence
are synchronized, so frame N means the same moment in every preview video.
Where view areas overlap, the first listed camera owns the ground, so list
the camera with the best view of a shared area first.

Usage: python merge_cameras.py <output dir> <camera dir> [<camera dir> ...]
"""
import json
import os
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    spec, detections, tracks = None, {}, {}
    for d in map(Path, sys.argv[2:]):
        b = json.loads((d / 'bundle.json').read_text())
        cam = b['cameras'][0]
        for key in ('video',):
            cam[key] = os.path.relpath((d / cam[key]).resolve(), out.resolve())
        if spec is None:
            spec = dict(b, cameras=[])
            spec['map'] = dict(b['map'], image=os.path.relpath((d / b['map']['image']).resolve(), out.resolve()))
        spec['cameras'].append(cam)
        detections.update(json.loads((d / 'detections.json').read_text()))
        tracks[cam['id']] = json.loads((d / 'detections-tracks.json').read_text())
    (out / 'bundle.json').write_text(json.dumps(spec, indent=1))
    (out / 'detections.json').write_text(json.dumps(detections))
    (out / 'tracks.json').write_text(json.dumps(tracks))
    print(json.dumps({'cameras': [c['id'] for c in spec['cameras']],
                      'frames': {k: len(v) for k, v in detections.items()}}))


if __name__ == '__main__':
    main()
