"""Pipeline figure for the README, from CityStreet only.

SCOUT terms forbid redistributing any part of SCOUT, so README images use
CityStreet (CC BY-NC-SA 4.0) and the Hong Kong Lands Department aerial map.

Needs the outputs of proof.py. Usage:
python readme_figure.py <citystreet data dir> <proof dir> <output.jpg> [--frame 1236]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'flowmap'))
import bundle as bundle_io  # noqa: E402
import heatmap  # noqa: E402
import transform  # noqa: E402

W = 560  # panel width
GAP = 70  # room for arrows
RED, GREEN, YELLOW, WHITE, BLACK = (40, 40, 230), (60, 200, 60), (0, 220, 255), (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, sub):
    pad = np.full((64, img.shape[1], 3), 255, np.uint8)
    cv2.putText(pad, text, (8, 26), FONT, .72, BLACK, 2, cv2.LINE_AA)
    cv2.putText(pad, sub, (8, 52), FONT, .5, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([pad, img])


def fit(img, width=W):
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)


def dot(img, p, color, r=6, text=None):
    p = tuple(int(v) for v in np.round(p))
    cv2.circle(img, p, r + 2, WHITE, -1, cv2.LINE_AA)
    cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
    if text:
        for col, th in ((BLACK, 4), (WHITE, 1)):
            cv2.putText(img, text, (p[0] + 9, p[1] - 7), FONT, .6, col, th, cv2.LINE_AA)


def row(panels, between='arrow'):
    h = max(p.shape[0] for p in panels)
    out = []
    for i, p in enumerate(panels):
        out.append(np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)]))
        if i < len(panels) - 1:
            gap = np.full((h, GAP, 3), 255, np.uint8)
            y = h // 2
            if between == 'arrow':
                cv2.arrowedLine(gap, (8, y), (GAP - 8, y), (80, 80, 80), 5, cv2.LINE_AA, tipLength=.35)
            else:
                cv2.putText(gap, between, (14, y + 10), FONT, .9, (80, 80, 80), 2, cv2.LINE_AA)
            out.append(gap)
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('proof')
    ap.add_argument('out')
    ap.add_argument('--frame', default='1236')
    ap.add_argument('--camera', default='camera1')
    a = ap.parse_args()
    proof = Path(a.proof)
    b = bundle_io.load(proof / 'bundle.json')
    cam = next(c for c in b['cameras'] if c['id'] == a.camera)
    boxes = json.loads((proof / 'detections.json').read_text())[a.camera][a.frame]
    frame = cv2.imread(str(Path(a.data) / f'extracted/CityStreet/image_frames/{a.camera}/frame_{int(a.frame):04d}.jpg'))
    s = W / frame.shape[1]

    # 1. Camera frame with the four hand-marked ground points.
    one = fit(frame)
    for i, p in enumerate(cam['frame_points']):
        dot(one, p * s, RED, 7, str(i + 1))
    # 2. People found by SAM, with the head point used for mapping.
    two = fit(frame)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(two, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), GREEN, 1, cv2.LINE_AA)
    for p in transform.head_points(boxes):
        cv2.circle(two, tuple(int(v) for v in p * s), 3, YELLOW, -1, cv2.LINE_AA)
    # 3. Same people on the top-down map, through the four point pairs.
    base = cv2.imread(str(b['map_image']))
    m = W / base.shape[1]
    three = fit(base)
    cv2.polylines(three, [np.int32(transform.view_area(cam) * m)], True, RED, 2, cv2.LINE_AA)
    for i, p in enumerate(cam['map_points']):
        dot(three, p * m, RED, 7, str(i + 1))
    H = transform.fit(cam['frame_points'], cam['map_points'])
    for p in heatmap.to_map(cam, H, boxes, 'head'):
        if 0 <= p[0] < base.shape[1] and 0 <= p[1] < base.shape[0]:
            cv2.circle(three, tuple(int(v) for v in p * m), 4, YELLOW, -1, cv2.LINE_AA)
    dot(three, np.array(cam['position']) * m, (255, 170, 0), 8, a.camera)

    top = row([label(one, '1. Camera video', 'Mark 4 ground spots (red) once per camera'),
               label(two, '2. Find people (SAM)', 'Person boxes (green), head points (yellow)'),
               label(three, '3. Place on the map', 'Same 4 spots on the map fix the conversion')])
    header = 64  # heatmap.render puts its own title and color bar on top; drop it
    est = fit(cv2.imread(str(proof / 'heatmap-sam-4points-head.png'))[header:], W)
    key = fit(cv2.imread(str(proof / 'heatmap-labels.png'))[header:], W)
    res = json.loads((proof / 'proof.json').read_text())['sam_4points_head']
    bottom = row([label(est, '4. Crowd heat map (all frames, 3 cameras)', f'Our pipeline: SAM + 4 points per camera, {res["frames"]} frames'),
                  label(key, '5. Answer key: official ground labels', f'Map correlation {res["map_correlation"]:.2f}, '
                                                                f'{res["recall_2m"] * 100:.0f}% of people within 2 m')], between='vs')
    pad = np.full((top.shape[0] + bottom.shape[0] + 110, max(top.shape[1], bottom.shape[1]), 3), 255, np.uint8)
    pad[:top.shape[0], :top.shape[1]] = top
    cv2.arrowedLine(pad, (pad.shape[1] // 2, top.shape[0] + 8), (pad.shape[1] // 2, top.shape[0] + 52), (80, 80, 80), 5,
                    cv2.LINE_AA, tipLength=.35)
    x0 = (pad.shape[1] - bottom.shape[1]) // 2
    pad[top.shape[0] + 60:top.shape[0] + 60 + bottom.shape[0], x0:x0 + bottom.shape[1]] = bottom
    cv2.putText(pad, 'Data: CityStreet (Zhang and Chan, CityU Hong Kong), CC BY-NC-SA 4.0. '
                     'Map data (c) Lands Department, HKSAR Government.', (10, pad.shape[0] - 18), FONT, .5, (60, 60, 60), 1,
                cv2.LINE_AA)
    cv2.imwrite(a.out, pad, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(a.out, pad.shape)


if __name__ == '__main__':
    main()
