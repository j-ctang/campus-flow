"""Score the SCOUT pipeline output against the official SCOUT annotations.

The annotations give every person a global ID and a ground position in scene
meters, in every frame of every camera. They are the answer key for both the
heat map (where people are) and the flow arrows (where people go).

Compares, on the ground the processed cameras cover:
- people per frame, overall and per camera
- heat map agreement and how close each mapped person is to a real one
- flow: rate per cell and direction, and arrow-by-arrow matches

Usage: python evaluate.py <scout data dir> <topdown dir> <run dir> <output.json>
[--image compare.png]. The run dir holds bundle.json, detections.json and
tracks.json, as written by merge_cameras.py.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'flowmap'))
import bundle as bundle_io  # noqa: E402
import cameras as cams  # noqa: E402
import flow  # noqa: E402
import heatmap  # noqa: E402
import transform  # noqa: E402

FPS = 10.0
SIGMA_M = 1.0
CELL_M = 4.0
SAM_CAP = 16  # people SAM follows per request


def load_truth(data, camera_ids, grid):
    """{frame: {person id: map px}} from every listed camera's annotations."""
    truth = {}
    for cid in camera_ids:
        path = Path(data) / f'annotations/mot_format/sequence_2/cam_{cid.removeprefix("cam")}.txt'
        a = np.loadtxt(path, delimiter=',', usecols=(0, 1, 7, 8))
        px = grid.to_px(a[:, 2:4])
        for f, pid, p in zip(a[:, 0].astype(int), a[:, 1].astype(int), px):
            truth.setdefault(f, {}).setdefault(pid, p)  # same ID, same position in every camera
    return truth


def inside(points, mask):
    if not len(points):
        return np.zeros(len(points), bool)
    x, y = np.round(points[:, 0]).astype(int), np.round(points[:, 1]).astype(int)
    ok = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    ok[ok] = mask[y[ok], x[ok]]
    return ok


def match(a, b, radius):
    """Greedy one-to-one matches closer than radius, nearest first."""
    if not len(a) or not len(b):
        return 0
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    used_a, used_b, n = set(), set(), 0
    for i, j in sorted(zip(*np.nonzero(d < radius)), key=lambda ij: d[ij]):
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j)
            n += 1
    return n


def frame_lag(bundle, tracks, truth, fps=FPS, near_m=1.5):
    """Estimate lag using one truth identity per track. Rank candidates by
    cross-track error and lag consistency, rather than nearest point per frame.
    Ambiguous parallel trajectories are excluded; this remains a diagnostic,
    not independent proof of synchronization or identity accuracy."""
    m, lags = bundle['meters_per_pixel'], []
    for cam in bundle['cameras']:
        H = transform.fit(cam['frame_points'], cam['map_points'])
        for t in tracks.get(cam['id'], {}).values():
            t = np.asarray(sorted(t), float)
            if len(t) < 5:
                continue
            p, f = transform.apply(H, flow.feet(t, fps)), t[:, 0].astype(int)
            candidates = {}
            for i in range(2, len(t) - 2):
                v = (p[i + 2] - p[i - 2]) / 4
                if f[i + 2] - f[i - 2] != 4 or not .8 <= np.hypot(*v) * m * fps <= 2.5 or f[i] not in truth:
                    continue
                speed = np.hypot(*v)
                for pid, q in truth[f[i]].items():
                    delta = q - p[i]
                    if np.hypot(*delta) * m <= near_m:
                        lag = float(delta @ v / (v @ v))
                        cross = abs(float(delta[0] * v[1] - delta[1] * v[0])) / speed
                        candidates.setdefault(pid, []).append((lag, cross))
            if not candidates:
                continue
            # Require sustained support, so a single crossing cannot win.
            support = max(map(len, candidates.values()))
            ranked = []
            for samples in candidates.values():
                if len(samples) < max(3, support * .5):
                    continue
                values = np.asarray(samples)
                center = np.median(values[:, 0])
                score = float(np.median(values[:, 1]) * m +
                              np.median(abs(values[:, 0] - center)) * .1)
                ranked.append((score, values[:, 0]))
            ranked.sort(key=lambda item: item[0])
            if ranked and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] > .05):
                lags.extend(ranked[0][1])
    return float(np.median(lags)) if lags else None


def truth_paths(truth, covered, m, min_s=1.5, min_move_m=1.0):
    """Answer-key tracks on the map, cut where they leave covered ground,
    smoothed and filtered like the pipeline's own tracks."""
    tracks = {}
    for f in sorted(truth):
        for pid, p in truth[f].items():
            tracks.setdefault(pid, []).append((f, p))
    w = max(1, int(round(flow.SMOOTH_S * FPS))) | 1
    out = []
    for seq in tracks.values():
        f = np.array([s[0] for s in seq])
        p = np.array([s[1] for s in seq])
        ok = inside(p, covered)
        breaks = np.flatnonzero((np.diff(f) > 1) | (np.diff(ok.astype(int)) != 0)) + 1
        for fs, ps, oks in zip(np.split(f, breaks), np.split(p, breaks), np.split(ok, breaks)):
            if not oks[0] or (fs[-1] - fs[0]) / FPS < min_s:
                continue
            s = flow.smooth(ps, w)
            if np.hypot(*(s[-1] - s[0])) * m >= min_move_m:
                out.append(s)
    return out


def compare_arrows(ours, truth, truth_field, m, near_m=6.0, max_angle=45):
    """Judge arrows by placement, direction and size, not exact cell matches.

    Arrows sit on a coarse grid, so a stream can land one cell over. An arrow
    agrees if the other side has one within near_m pointing within max_angle.
    Our arrows with no such partner are split into weaker real flow (the
    answer key has flow there in that direction, just below the arrow
    threshold), opposite or sideways flow, and flow where the answer key has none.
    """
    cos, near = np.cos(np.radians(max_angle)), near_m / m

    def partner(a, others):
        best = None
        for c, d, r in others:
            if np.hypot(*(c - a[0])) <= near and float(a[1] @ d) >= cos and (best is None or float(a[1] @ d) > float(a[1] @ best[1])):
                best = (c, d, r)
        return best
    cell, bins, rate = truth_field['cell_px'], truth_field['bins'], truth_field['rate']
    kinds, pairs, angles = {'agrees': 0, 'weaker_real_flow': 0, 'opposite': 0, 'sideways': 0, 'no_flow_in_answer_key': 0}, [], []
    for a in ours:
        p = partner(a, truth)
        if p is not None:
            kinds['agrees'] += 1
            pairs.append((p[2], a[2]))
            angles.append(float(np.degrees(np.arccos(np.clip(a[1] @ p[1], -1, 1)))))
            continue
        i, j = (a[0] // cell).astype(int)
        b = int(np.round(np.arctan2(a[1][1], a[1][0]) / (2 * np.pi / bins))) % bins
        same = rate[j, i, [(b - 1) % bins, b, (b + 1) % bins]].sum()
        if same >= .5 * a[2]:
            kinds['weaker_real_flow'] += 1
            continue
        near_t = [t for t in truth if np.hypot(*(t[0] - a[0])) <= near]
        if not near_t:
            kinds['no_flow_in_answer_key'] += 1
        elif max(float(a[1] @ t[1]) for t in near_t) <= -cos:
            kinds['opposite'] += 1
        else:
            kinds['sideways'] += 1
    pairs = np.array(pairs) if pairs else np.zeros((0, 2))
    rank = (lambda x: np.argsort(np.argsort(x)))  # noqa: E731
    return {'near_m': near_m, 'max_angle_deg': max_angle, 'truth_arrows': len(truth), 'our_arrows': len(ours),
            'truth_arrows_with_our_arrow_nearby': sum(partner(t, ours) is not None for t in truth) / max(len(truth), 1),
            'our_arrows_by_kind': kinds,
            'angle_to_partner_deg_median': float(np.median(angles)) if angles else None,
            'rate_ours_over_truth_median': float(np.median(pairs[:, 1] / pairs[:, 0])) if len(pairs) else None,
            'rate_rank_correlation': float(np.corrcoef(rank(pairs[:, 0]), rank(pairs[:, 1]))[0, 1]) if len(pairs) > 2 else None}


def draw(base, covered, ours, truth, path):
    """Truth arrows green, ours white with black edge. Private: SCOUT imagery."""
    img = base.copy()
    img[~covered] = (img[~covered] * .4 + 150).astype(np.uint8)
    for arrows, color, width in ((truth, (60, 200, 60), 7), (ours, (255, 255, 255), 3)):
        top = max((r for *_, r in arrows), default=1)
        for c, d, r in arrows:
            L = 14 + 22 * r / top
            p0, p1 = c - d * L / 2, c + d * L / 2
            if color == (255, 255, 255):
                cv2.arrowedLine(img, tuple(np.int32(p0)), tuple(np.int32(p1)), (0, 0, 0), width + 2, cv2.LINE_AA, tipLength=.35)
            cv2.arrowedLine(img, tuple(np.int32(p0)), tuple(np.int32(p1)), color, width, cv2.LINE_AA, tipLength=.35)
    ys, xs = np.nonzero(covered)
    cv2.imwrite(str(path), img[max(ys.min() - 60, 0):ys.max() + 60, max(xs.min() - 60, 0):xs.max() + 60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('topdown')
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--image')
    a = ap.parse_args()
    run, top = Path(a.run), Path(a.topdown)
    grid = cams.load_grid(top)
    b = bundle_io.load(run / 'bundle.json')
    m = b['meters_per_pixel']
    det = json.loads((run / 'detections.json').read_text())
    ids = [c['id'] for c in b['cameras']]
    result = heatmap.run(b, det, SIGMA_M, 'feet')
    areas = dict(np.load(top / 'view-areas.npz'))
    covered = result['covered'] & np.any([areas[c] for c in ids], 0)  # as in demo_map.py
    owner = np.where(covered, result['owner'], -1)
    ours = {int(k): p[inside(p, covered)] for k, p in result['points'].items()}

    raw = load_truth(a.data, ids, grid)
    truth = {f: np.array(list(p.values())) for f, p in raw.items()}
    truth = {f: p[inside(p, covered)] for f, p in truth.items()}
    frames = sorted(set(ours) & set(truth))

    n_o = np.array([len(ours[f]) for f in frames])
    n_t = np.array([len(truth[f]) for f in frames])
    shape = covered.shape
    dens = lambda pts: heatmap.density(np.vstack([pts[f] for f in frames]), shape, SIGMA_M / m, covered)  # noqa: E731
    h_o, h_t = dens(ours), dens(truth)
    hits = {r: sum(match(ours[f], truth[f], r / m) for f in frames) for r in (1, 2)}
    top10 = lambda d: d >= np.percentile(d[covered], 90)  # noqa: E731

    per_cam = {}
    for i, cid in enumerate(b['cameras']):
        own = owner == i
        co = np.array([inside(ours[f], own).sum() for f in frames])
        ct = np.array([inside(truth[f], own).sum() for f in frames])
        raw_n = np.array([len(det[cid['id']].get(str(f), [])) for f in frames])
        capped = raw_n >= SAM_CAP
        per_cam[cid['id']] = {'area_m2': float(own.sum() * m * m), 'truth_per_frame': float(ct.mean()), 'ours_per_frame': float(co.mean()),
                              'count_correlation': float(np.corrcoef(co, ct)[0, 1]) if co.std() and ct.std() else None,
                              'frames_at_sam_cap_pct': float(capped.mean() * 100),
                              'ours_over_truth_at_cap': float(co[capped].sum() / max(ct[capped].sum(), 1)) if capped.any() else None,
                              'ours_over_truth_below_cap': float(co[~capped].sum() / max(ct[~capped].sum(), 1))}

    minutes = len(frames) / FPS / 60
    keep = set(frames)
    f_o = flow.flow_field(b, {c: {k: [p for p in t if int(p[0]) in keep] for k, t in v.items()}
                              for c, v in flow.load_tracks(run / 'tracks.json', ids[0]).items()},
                          FPS, CELL_M, covered=covered, duration_s=minutes * 60)
    f_t = flow.field_from_paths(truth_paths({f: p for f, p in raw.items() if f in keep}, covered, m), shape, covered, m, minutes * 60, CELL_M)
    r_o, r_t = f_o['rate'], f_t['rate']
    seen = f_o['covered_frac'] >= .5
    both = seen[..., None].repeat(r_o.shape[2], 2)
    a_o, a_t = flow.arrows(f_o), flow.arrows(f_t)  # each threshold scales with its own crowd

    report = {
        'cameras': ids, 'frames': len(frames), 'minutes': minutes, 'covered_m2': float(covered.sum() * m * m),
        'frame_lag': frame_lag(b, flow.load_tracks(run / 'tracks.json', ids[0]), raw),
        'counts': {'truth_per_frame': float(n_t.mean()), 'ours_per_frame': float(n_o.mean()),
                   'ours_over_truth': float(n_o.sum() / n_t.sum()), 'count_correlation': float(np.corrcoef(n_o, n_t)[0, 1]),
                   'truth_peak': int(n_t.max()), 'ours_peak': int(n_o.max())},
        'heat': {'map_correlation': float(np.corrcoef(h_o[covered], h_t[covered])[0, 1]),
                 'hot_area_iou_top10pct': float(((top10(h_o) & top10(h_t)) & covered).sum() / ((top10(h_o) | top10(h_t)) & covered).sum()),
                 'recall_1m': hits[1] / n_t.sum(), 'recall_2m': hits[2] / n_t.sum(),
                 'precision_1m': hits[1] / max(n_o.sum(), 1), 'precision_2m': hits[2] / max(n_o.sum(), 1)},
        'flow': {'cell_m': CELL_M, 'min_count_ours': flow.auto_min_count(f_o), 'min_count_truth': flow.auto_min_count(f_t), 'truth_tracks': f_t['tracks_kept'], 'our_tracks': f_o['tracks_kept'],
                 'rate_correlation_cells_x_directions': float(np.corrcoef(r_o[both], r_t[both])[0, 1]),
                 'total_rate_ours_over_truth': float(r_o[both].sum() / r_t[both].sum()),
                 'arrows': compare_arrows(a_o, a_t, f_t, m)},
        'per_camera': per_cam,
    }
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if a.image:
        draw(cv2.imread(str(top / 'map.png')), covered, a_o, a_t, a.image)


if __name__ == '__main__':
    main()
