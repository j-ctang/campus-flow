"""SAM 3.1 video output -> person boxes per frame, in original video pixels.

SAM writes one line per frame: "<Nf>" (source frame index), then
"id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;...|>" entries separated
by commas. IDs are stable within one response only.

Several crops of one video can be merged. Each crop has an offset and scale
back to the original frame and owns a band of original x: a person belongs
to the crop whose band contains their foot, so overlapping crops never
count a person twice.

A crop can be one stream file or a folder of chunk-<start>-<end>.sse files
written by sam_video.py; chunk frame numbers are shifted by <start>. Track
IDs restart in every chunk, so tracks are tagged per crop and chunk.

Usage: python sam_tracks.py <camera> <out.json> <stream or chunk dir>:<x0>:<y0>:<scale>:<own_x0>:<own_x1> [...]
Output: {"<camera>": {"<frame>": [[x1, y1, x2, y2], ...]}} plus a tracks file
{"<track>": [[frame, foot x, foot y, top y], ...]}.
"""
import json
import re
import sys
from pathlib import Path

FRAME = re.compile(r'<(\d+)f>')
BOX = re.compile(r'(\d+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)\|>')


def stream_text(path):
    """Concatenated output text of a saved SSE stream (or a plain JSON response)."""
    raw = Path(path).read_text()
    if raw.lstrip().startswith('{'):
        r = json.loads(raw)
        return ''.join(c.get('text', '') for o in r['output'] for c in o.get('content', []))
    parts = []
    for line in raw.splitlines():
        if not line.startswith('data: '):
            continue
        try:
            ev = json.loads(line[6:])
        except ValueError:
            continue
        if ev.get('type') == 'response.output_text.delta':
            parts.append(ev.get('delta', ''))
    return ''.join(parts)


def parse(text):
    """{frame: [(track id, x1, y1, x2, y2)]} in the coordinates SAM saw."""
    out = {}
    marks = list(FRAME.finditer(text))
    for i, m in enumerate(marks):
        chunk = text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        out[int(m.group(1))] = [tuple(int(g) for g in b.groups()[:5]) for b in BOX.finditer(chunk)]
    return out


def load(path):
    """{frame: [...]} for one stream file or a chunk folder, plus a chunk tag per frame."""
    path = Path(path)
    if path.is_file():
        return parse(stream_text(path)), {}
    frames, tags = {}, {}
    for f in sorted(path.glob('chunk-*.sse')):
        start, end = map(int, re.match(r'chunk-(\d+)-(\d+)', f.stem).groups())
        for k, v in parse(stream_text(f)).items():
            if k + start < end:
                frames[k + start] = v
                tags[k + start] = start
    return frames, tags


def merge(crops):
    """crops: [(frames, x0, y0, scale, own_x0, own_x1, tag)] -> boxes and tracks."""
    boxes, tracks = {}, {}
    for frames, x0, y0, s, lo, hi, tag, chunk_of in crops:
        for f, items in frames.items():
            for tid, a, b, c, d in items:
                box = [a / s + x0, b / s + y0, c / s + x0, d / s + y0]
                foot = (box[0] + box[2]) / 2
                if not lo <= foot < hi:
                    continue
                boxes.setdefault(str(f), []).append([round(v, 1) for v in box])
                tracks.setdefault(f'{tag}:{chunk_of.get(f, 0)}:{tid}', []).append([f, round(foot, 1), round(box[3], 1), round(box[1], 1)])
    return boxes, tracks


def main():
    camera, out = sys.argv[1], Path(sys.argv[2])
    crops = []
    for i, spec in enumerate(sys.argv[3:]):
        path, *nums = spec.split(':')
        x0, y0, s, lo, hi = map(float, nums)
        crops.append((*load(path)[:1], x0, y0, s, lo, hi, f'crop{i}', load(path)[1]))
    boxes, tracks = merge(crops)
    n = max((max(f) for f, *_ in crops if f), default=-1) + 1
    for f in range(n):
        boxes.setdefault(str(f), [])  # frames with nobody detected
    out.write_text(json.dumps({camera: boxes}))
    out.with_name(out.stem + '-tracks.json').write_text(json.dumps(tracks))
    counts = [len(v) for v in boxes.values()]
    print(json.dumps({'frames': len(boxes), 'people_mean': sum(counts) / max(len(counts), 1),
                      'people_max': max(counts, default=0), 'tracks': len(tracks)}))


if __name__ == '__main__':
    main()
