"""Person tracks + bundle -> directional flow arrows on the map.

Tracks JSON: {"<camera id>": {"<track id>": [[frame, x, y], ...]}} with x, y
the foot point in frame pixels. A flat {"<track id>": [...]} file is read as
one camera (--camera). Each track is mapped through the ground homography,
joined to a track that starts right where it ends (IDs restart per SAM
chunk), split at gaps and jumps, smoothed, and dropped if too short or
stationary.

Flow is counted on a coarse grid. Every track adds one crossing to each cell it
walks through, in one of `bins` direction bins, so opposite streams stay
separate. Rate = crossings per minute of footage. Speed is not used.
Usage: python flow.py <bundle.json> <tracks.json> <output dir> [--camera cam10] [--fps 10] [--cell-m 2]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle as bundle_io  # noqa: E402
import heatmap  # noqa: E402
import transform  # noqa: E402

MAX_SPEED_MPS = 6.0  # faster steps (beyond pixel noise) are ID swaps
NOISE_PX = 4.0  # box bottoms jitter by a few frame pixels
MAX_GAP_S = 0.5  # a longer gap splits a track
SMOOTH_S = 0.5  # median, then mean, over this window


def load_tracks(path, camera=None):
    """Tracks file -> {camera id: {track id: [[frame, x, y], ...]}}."""
    raw = json.loads(Path(path).read_text())
    if raw and all(isinstance(v, list) for v in raw.values()):
        if not camera:
            raise ValueError('flat tracks file: give the camera id')
        return {camera: raw}
    return raw


def _window(x, w, fn):
    pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode='edge')
    return np.stack([fn(pad[i:i + w], 0) for i in range(len(x))])


def smooth(points, w):
    """Median then moving mean over w samples (w odd), per axis."""
    if w < 2 or len(points) < w:
        return points
    return _window(_window(points, w, np.median), w, np.mean)


def pixel_size(H, xy):
    """Map pixels per frame pixel at each frame point (the larger axis).
    Far from the camera one frame pixel spans much more ground."""
    p = transform.apply(H, xy)
    return np.maximum(np.hypot(*(transform.apply(H, xy + [1, 0]) - p).T),
                      np.hypot(*(transform.apply(H, xy + [0, 1]) - p).T))


def pieces(frames, pts, size, fps, m):
    """Split a mapped track at frame gaps and impossible jumps. A step may be
    walking speed plus a few frame pixels of noise at that distance."""
    dt = np.diff(frames) / fps
    step = np.hypot(*np.diff(pts, axis=0).T) * m
    noise = NOISE_PX * np.maximum(size[:-1], size[1:]) * m
    cut = (dt > MAX_GAP_S) | (dt <= 0) | (step > MAX_SPEED_MPS * dt + noise)
    bounds = np.concatenate([[0], np.nonzero(cut)[0] + 1, [len(pts)]])
    return [(frames[a:b], pts[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def stitch(parts, fps, m):
    """Join a track that ends to one that starts just after, close by. Track
    IDs can restart (SAM chunks, ID swaps); this makes one walk one track.
    parts: [(frames, map points, pixel sizes)]. Nearest pairs join first."""
    if len(parts) < 2:
        return parts
    f0, f1 = np.array([p[0][0] for p in parts]), np.array([p[0][-1] for p in parts])
    p0, p1 = np.array([p[1][0] for p in parts]), np.array([p[1][-1] for p in parts])
    s0, s1 = np.array([p[2][0] for p in parts]), np.array([p[2][-1] for p in parts])
    dt = (f0[None, :] - f1[:, None]) / fps  # [end track, start track]
    dist = np.hypot(*(p0[None, :, :] - p1[:, None, :]).transpose(2, 0, 1)) * m
    reach = MAX_SPEED_MPS * dt + NOISE_PX * np.maximum(s0[None, :], s1[:, None]) * m
    a, b = np.nonzero((dt > 0) & (dt <= MAX_GAP_S) & (dist <= reach))
    nxt, taken = {}, set()
    for k in np.argsort(dist[a, b], kind='stable'):
        if a[k] not in nxt and b[k] not in taken:
            nxt[a[k]] = b[k]
            taken.add(b[k])
    out = []
    for head in (i for i in range(len(parts)) if i not in taken):
        chain = [head]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        out.append(tuple(np.concatenate([parts[i][j] for i in chain]) for j in range(3)))
    return out


def map_tracks(bundle, tracks, fps, owner, min_s=1.5, min_move_m=1.0):
    """Clean map-pixel paths, one per kept track piece. Points a camera does
    not own on the map are cut out, so overlapping cameras count once."""
    m = bundle['meters_per_pixel']
    w = max(1, int(round(SMOOTH_S * fps))) | 1
    h, wid = owner.shape
    kept, dropped = [], 0
    for i, cam in enumerate(bundle['cameras']):
        H = transform.fit(cam['frame_points'], cam['map_points'])
        parts = []
        for track in tracks.get(cam['id'], {}).values():
            t = np.asarray(sorted(track), float).reshape(-1, 3)
            p, size = transform.apply(H, t[:, 1:]), pixel_size(H, t[:, 1:])
            x, y = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
            ok = (x >= 0) & (x < wid) & (y >= 0) & (y < h)
            ok[ok] = owner[y[ok], x[ok]] == i
            if ok.any():
                parts.append((t[ok, 0], p[ok], size[ok]))
        for frames, pts, size in stitch(parts, fps, m):
            for f, q in pieces(frames, pts, size, fps, m):  # leaving the owned area leaves a gap
                s = smooth(q, w)
                if (f[-1] - f[0]) / fps < min_s or np.hypot(*(s[-1] - s[0])) * m < min_move_m:
                    dropped += 1
                    continue
                kept.append(s)
    return kept, dropped


def resample(path, step):
    """Points every `step` pixels along a path."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return path[[0, -1]]
    u = np.arange(0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(u, s, path[:, 0]), np.interp(u, s, path[:, 1])])


def flow_field(bundle, tracks, fps, cell_m=2.0, bins=8, covered=None, duration_s=None, min_s=1.5, min_move_m=1.0):
    """Crossings per cell and direction bin. `covered` (map-shaped bool) limits
    flow to ground that is really seen; default is the cameras' view areas."""
    m = bundle['meters_per_pixel']
    if covered is None:
        base = cv2.imread(str(bundle['map_image']))
        if base is None:
            raise FileNotFoundError(bundle['map_image'])
        shape = base.shape[:2]
    else:
        shape = covered.shape
    owner = heatmap.ownership(bundle['cameras'], shape)
    cov = owner >= 0 if covered is None else covered & (owner >= 0)
    paths, dropped = map_tracks(bundle, tracks, fps, owner, min_s, min_move_m)
    cell = cell_m / m
    ny, nx = int(np.ceil(shape[0] / cell)), int(np.ceil(shape[1] / cell))
    counts = np.zeros((ny, nx, bins))
    vec = np.zeros((ny, nx, bins, 2))
    for path in paths:
        r = resample(path, cell / 4)
        d = np.diff(r, axis=0)
        mid = (r[:-1] + r[1:]) / 2
        x, y = np.round(mid[:, 0]).astype(int), np.round(mid[:, 1]).astype(int)
        ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        ok[ok] = cov[y[ok], x[ok]]
        cx, cy = (mid[ok, 0] // cell).astype(int), (mid[ok, 1] // cell).astype(int)
        net = {}
        for key, dv in zip(zip(cy, cx), d[ok]):
            net[key] = net.get(key, 0) + dv
        for (j, i), v in net.items():
            n = np.hypot(*v)
            if n < cell / 4:  # wandering inside the cell, not passing through
                continue
            b = int(np.round(np.arctan2(v[1], v[0]) / (2 * np.pi / bins))) % bins
            counts[j, i, b] += 1
            vec[j, i, b] += v / n
    if duration_s is None:
        frames = [f for cam in tracks.values() for t in cam.values() for f, *_ in t]
        duration_s = (max(frames) - min(frames) + 1) / fps if frames else 1.0
    frac = np.zeros((ny, nx))
    ys, xs = np.nonzero(cov)
    np.add.at(frac, ((ys // cell).astype(int), (xs // cell).astype(int)), 1 / cell ** 2)
    return {'counts': counts, 'vec': vec, 'rate': counts / (duration_s / 60), 'cell_px': cell, 'bins': bins,
            'covered_frac': frac, 'cell_m': cell_m, 'minutes': duration_s / 60, 'tracks_kept': len(paths), 'tracks_dropped': dropped,
            'paths': paths}


def peaks(counts):
    """Group direction bins around each local peak: every bin joins its
    larger neighbour until a peak. A stream split across two bins becomes one
    arrow; opposite streams have separate peaks."""
    n = len(counts)
    up = [max((b - 1) % n, b, (b + 1) % n, key=lambda k: (counts[k], k == b)) for b in range(n)]
    top = list(range(n))
    for b in range(n):
        while up[top[b]] != top[b]:
            top[b] = up[top[b]]
    return top


def arrows(field, min_count=3, min_cover=.5, min_share=.15):
    """[(center, unit direction, people per minute)], one per direction peak in
    cells with at least `min_count` tracks and `min_share` of the top rate."""
    cell, per_min = field['cell_px'], 1 / field['minutes']
    out = []
    for j, i in zip(*np.nonzero(field['counts'].sum(2) >= min_count)):
        if field['covered_frac'][j, i] < min_cover:
            continue
        c, v, top = field['counts'][j, i], field['vec'][j, i], peaks(field['counts'][j, i])
        for b in set(top):
            group = [k for k in range(len(c)) if top[k] == b]
            n, d = c[group].sum(), v[group].sum(0)
            if n >= min_count and np.hypot(*d) > 0:
                out.append((np.array([(i + .5) * cell, (j + .5) * cell]), d / np.hypot(*d), float(n * per_min)))
    top_rate = max((r for *_, r in out), default=0)
    return [a for a in out if a[2] >= min_share * top_rate]


def draw_flow(img, field, min_count=3, max_width=None, ref_rate=None, color=(255, 255, 255)):
    """Draw one arrow per supported cell and direction. Width grows with people
    per minute; each arrow sits a little to its own right, so opposite streams
    show side by side. Returns the rate drawn at full width."""
    cell = field['cell_px']
    items = arrows(field, min_count)
    if not items:
        return 0.0
    ref = ref_rate or max(r for *_, r in items)
    max_width = max_width or max(2, int(round(cell / 6)))
    for c, d, r in sorted(items, key=lambda a: a[2]):
        side = np.array([-d[1], d[0]]) * cell * .18  # right of travel, image y points down
        a = tuple(int(v) for v in np.round(c + side - d * cell * .4))
        b = tuple(int(v) for v in np.round(c + side + d * cell * .4))
        th = int(np.clip(round(max_width * r / ref), 1, max_width))
        cv2.arrowedLine(img, a, b, (0, 0, 0), th + 2, cv2.LINE_AA, tipLength=.35)
        cv2.arrowedLine(img, a, b, color, th, cv2.LINE_AA, tipLength=.35)
    return ref


def summary(field, min_count=3):
    items = arrows(field, min_count)
    return {'minutes': field['minutes'], 'tracks_kept': field['tracks_kept'], 'tracks_dropped': field['tracks_dropped'],
            'cell_m': field['cell_m'], 'arrows': len(items),
            'max_people_per_min': max((r for *_, r in items), default=0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle')
    ap.add_argument('tracks')
    ap.add_argument('out')
    ap.add_argument('--camera')
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--cell-m', type=float, default=2.0)
    ap.add_argument('--min-count', type=int, default=3)
    a = ap.parse_args()
    b = bundle_io.load(a.bundle)
    base = cv2.imread(str(b['map_image']))
    if base is None:
        raise FileNotFoundError(b['map_image'])
    owner = heatmap.ownership(b['cameras'], base.shape[:2])
    field = flow_field(b, load_tracks(a.tracks, a.camera), a.fps, a.cell_m, covered=owner >= 0)
    img = base.astype(np.float32)
    img[owner < 0] = img[owner < 0] * .45 + 255 * .25
    img = np.uint8(img)
    ref = draw_flow(img, field, a.min_count)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / 'flow.png'), img)
    rep = {**summary(field, a.min_count), 'full_width_people_per_min': ref}
    (out / 'flow.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
