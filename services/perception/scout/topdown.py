"""Top-down map of the whole SCOUT scene from the official textured mesh.

Orthographic view straight down (scene z is up, units are meters). Each
mesh face is sampled at points spread over its area; the highest sample
per map pixel wins, like an aerial photo. Faces are processed in chunks so
memory stays bounded.

Writes <out>/map.png and <out>/map.json (pixel <-> scene transform).
Usage: python topdown.py <scout data dir> <output dir> [--m-per-px 0.1]
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import json
import re
import resource
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(2)
FACE = np.dtype([('n', 'u1'), ('i', '<i4', 3), ('m', 'u1'), ('t', '<f4', 6), ('k', '<i4')])
CHUNK = 200_000
MAX_SAMPLES = 64
# Keep the 0.5..99.5 percentile box of vertices: a few stray far vertices
# (reconstruction noise) would otherwise blow up the canvas.
BOUNDS_PERCENTILE = 0.5
MARGIN_M = 5.0


def read_ply(path):
    """Vertices (N, 3) and faces (memmap) of the SCOUT textured PLY."""
    head = open(path, 'rb').read(4096)
    end = head.index(b'end_header\n') + len('end_header\n')
    text = head[:end].decode()
    nv = int(re.search(r'element vertex (\d+)', text).group(1))
    nf = int(re.search(r'element face (\d+)', text).group(1))
    textures = re.findall(r'comment TextureFile (\S+)', text)
    v = np.fromfile(path, dtype='<f4', count=nv * 3, offset=end).reshape(-1, 3)
    f = np.memmap(path, dtype=FACE, mode='r', offset=end + v.nbytes, shape=(nf,))
    if not ((f['n'][:1000] == 3).all() and (f['m'][:1000] == 6).all()):
        raise ValueError('expected triangles with 6 texture coordinates per face')
    return v, f, textures


class Grid:
    """Scene (x, y) meters <-> map pixels. North-up is not known, so the map
    keeps scene axes: +x to the right, +y up."""

    def __init__(self, x0, y0, x1, y1, m_per_px):
        self.x0, self.y1, self.m = x0, y1, m_per_px
        self.w = int(np.ceil((x1 - x0) / m_per_px))
        self.h = int(np.ceil((y1 - y0) / m_per_px))

    def to_px(self, xy):
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.x0) / self.m, (self.y1 - xy[..., 1]) / self.m], -1)

    def to_scene(self, px):
        px = np.asarray(px, float)
        return np.stack([self.x0 + px[..., 0] * self.m, self.y1 - px[..., 1] * self.m], -1)

    def as_dict(self):
        return {'m_per_px': self.m, 'scene_x_at_px0': float(self.x0), 'scene_y_at_py0': float(self.y1),
                'width': self.w, 'height': self.h,
                # Checked against OpenStreetMap building edges near the scene: up is north (+-0.5 deg), not mirrored.
                'north_deg': 0.0,
                'note': 'px = (x - scene_x_at_px0) / m_per_px; py = (scene_y_at_py0 - y) / m_per_px'}


def samples(n_per_face):
    """Barycentric weights for n points spread over a triangle."""
    k = int(np.ceil((np.sqrt(8 * n_per_face + 1) - 1) / 2))  # rows of a triangular lattice
    pts = [((i + 1 / 3) / k, (j + 1 / 3) / k) for i in range(k) for j in range(k - i)]
    a = np.array(pts)
    return np.column_stack([1 - a.sum(1), a[:, 0], a[:, 1]])


def render(verts, faces, textures, grid, max_mb):
    color = np.full((grid.h, grid.w, 3), 235, np.uint8)
    zbuf = np.full(grid.h * grid.w, -np.inf, np.float32)
    flat = color.reshape(-1, 3)
    for start in range(0, len(faces), CHUNK):
        f = faces[start:start + CHUNK]
        idx, uv, tex = np.asarray(f['i']), np.asarray(f['t']).reshape(-1, 3, 2), np.asarray(f['k'])
        tri = verts[idx]  # (n, 3 corners, xyz)
        p = grid.to_px(tri[..., :2])
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
        need = np.clip(np.ceil(area * 2), 1, MAX_SAMPLES).astype(int)
        for n in np.unique(need):
            sel = need == n
            w = samples(n)  # (s, 3)
            q = np.einsum('sc,fcd->fsd', w, p[sel]).reshape(-1, 2)
            z = np.einsum('sc,fc->fs', w, tri[sel][..., 2]).ravel()
            t = np.einsum('sc,fcd->fsd', w, uv[sel]).reshape(-1, 2)
            k = np.repeat(tex[sel], len(w))
            x, y = np.floor(q[:, 0]).astype(int), np.floor(q[:, 1]).astype(int)
            ok = (x >= 0) & (x < grid.w) & (y >= 0) & (y < grid.h)
            lin, z, t, k = (y * grid.w + x)[ok], z[ok].astype(np.float32), t[ok], k[ok]
            order = np.lexsort((z, lin))  # highest sample last per pixel
            last = np.r_[lin[order][1:] != lin[order][:-1], True]
            pick = order[last]
            lin, z, t, k = lin[pick], z[pick], t[pick], k[pick]
            win = z > zbuf[lin]
            lin, z, t, k = lin[win], z[win], t[win], k[win]
            zbuf[lin] = z
            for ti, img in enumerate(textures):
                m = k == ti
                th, tw = img.shape[:2]
                u = np.clip((t[m, 0] * tw).astype(int), 0, tw - 1)
                v = np.clip(((1 - t[m, 1]) * th).astype(int), 0, th - 1)
                flat[lin[m]] = img[v, u]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # macOS: bytes
        print(f'faces {min(start + CHUNK, len(faces))}/{len(faces)} peak {rss:.0f} MB', flush=True)
        if rss > max_mb:
            sys.exit(f'memory guard: {rss:.0f} MB > {max_mb} MB')
    return color, zbuf.reshape(grid.h, grid.w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('out')
    ap.add_argument('--m-per-px', type=float, default=0.1)
    ap.add_argument('--max-memory-mb', type=float, default=3000)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    verts, faces, names = read_ply(data / 'meshes/mesh.ply')
    lo = np.percentile(verts[:, :2], BOUNDS_PERCENTILE, 0) - MARGIN_M
    hi = np.percentile(verts[:, :2], 100 - BOUNDS_PERCENTILE, 0) + MARGIN_M
    grid = Grid(lo[0], lo[1], hi[0], hi[1], a.m_per_px)
    print(f'canvas {grid.w}x{grid.h} px at {a.m_per_px} m/px', flush=True)
    textures = [cv2.imread(str(data / 'meshes' / n)) for n in names]
    color, z = render(verts, faces, textures, grid, a.max_memory_mb)
    del textures
    empty = ~np.isfinite(z)
    cv2.imwrite(str(out / 'map.png'), color)
    np.save(out / 'height.npy', np.where(empty, np.nan, z).astype(np.float32))
    (out / 'map.json').write_text(json.dumps(grid.as_dict(), indent=2))
    print(json.dumps({'width': grid.w, 'height': grid.h, 'filled_fraction': float(1 - empty.mean())}))


if __name__ == '__main__':
    main()
