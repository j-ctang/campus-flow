"""SAM 3.1 person segmentation of a long video, in chunks, resumable.

Why chunks: SAM picks the objects it tracks near the start of a clip and
then mostly follows only those (seen on SCOUT cam10: 8.3 people per frame in
the first 200 frames, about 1 afterwards). Long requests can also fail
mid-stream (seen: server_error after 2,322 of 11,970 frames). So the video is
cut into short chunks (default 100 frames, 10 s at 10 fps), one request each:

- Chunks run one at a time, with --pause seconds between requests.
- Each finished chunk is cached as chunk-<start>-<end>.sse; a rerun skips
  finished chunks, so a failure costs at most one chunk.
- Network errors (timeouts, resets, broken pipes) and server-busy answers
  (HTTP 429, 502, 503, 504) wait --network-wait
  seconds and retry the same chunk, at most --network-resumes times per run.
  Any other API error (such as 400 or 401, or a failed response) stops the run at
  once; rerun to resume.
- Every request is appended to the usage ledger (--ledger) right after it ends.
- --budget caps the nominal cost of new requests in this run.

The API key is read from MODEL_API_KEY in <repo>/.env and never printed.

Usage: python sam_video.py <video.mp4> <output dir> --ledger <usage-ledger.json>
       [--chunk 100] [--pause 3] [--budget 5.0] [--start N --end M]
"""
import argparse
import http.client
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API = 'https://api.meta.ai/v1'
ROOT = Path(__file__).resolve().parents[3]
STALL_S = 60  # a stream read that waits longer counts as a network error
DEADLINE_S = 300  # whole request; a normal 100-frame chunk takes about 10 s
UPLOAD_TIMEOUT_S = 120
RETRY_STATUS = {429, 502, 503, 504}
USD_PER_FRAME = 0.20 / 1000  # official price, checked November 2026
MIN_FRAMES = 50


def key():
    for line in (ROOT / '.env').read_text().splitlines():
        if line.startswith('MODEL_API_KEY='):
            return line.split('=', 1)[1].strip()
    sys.exit('MODEL_API_KEY missing from .env')


def frame_count(video):
    out = subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0',
                          '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', str(video)],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def cut(video, start, end, path):
    """Frames [start, end) as their own clip, near-lossless, 2 threads."""
    tmp = path.with_suffix('.tmp.mp4')
    subprocess.run(['nice', '-n', '15', 'ffmpeg', '-nostdin', '-loglevel', 'error', '-threads', '2', '-i', str(video),
                    '-vf', f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', '-threads', '2',
                    '-an', '-y', str(tmp)], check=True)
    if frame_count(tmp) != end - start:
        sys.exit(f'{tmp}: expected {end - start} frames')
    tmp.replace(path)


def is_network_error(e):
    """True for connection trouble, False for answers from the API."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code in RETRY_STATUS  # server busy or briefly down; other answers are final
    return isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError))


def upload(path, token):
    boundary = uuid.uuid4().hex
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            'Content-Type: video/mp4\r\n\r\n').encode()
    body = head + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(f'{API}/files', body, {'Authorization': f'Bearer {token}',
                                                        'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req, timeout=UPLOAD_TIMEOUT_S) as r:
        info = json.load(r)
    if not info.get('id'):
        sys.exit(f'upload failed: {info}')
    return info['id']


def _stream(req, partial, held, result):
    """Worker: send the request and copy the stream to partial.

    Network trouble after the final event is ignored: the answer is in.
    Errors writing or closing the cache file always count, so an incomplete
    file is never kept.
    """
    try:
        f = open(partial, 'wb')
    except OSError as e:
        result['cache_error'] = e
        return
    try:
        try:
            with urllib.request.urlopen(req, timeout=STALL_S) as r:
                held['r'] = r
                for line in r:
                    try:
                        f.write(line)
                    except OSError as e:
                        result['cache_error'] = e
                        raise
                    if line.startswith(b'data: '):
                        try:
                            ev = json.loads(line[6:])
                        except ValueError:
                            continue
                        if ev.get('type') in ('response.completed', 'response.failed', 'response.incomplete', 'error'):
                            result['final'] = ev
                            break  # final event: stop even if the server keeps the connection open
        except Exception as e:
            if 'cache_error' not in result and 'final' not in result:
                result['error'] = e
    finally:
        try:
            f.close()
        except OSError as e:
            result.setdefault('cache_error', e)


def segment(file_id, token, out):
    """One streamed request. Returns the final stream event (or None).

    The whole request (connect, headers, stream) runs in a worker thread
    with one wall-clock deadline. The read timeout alone misses some hung
    connections (seen: no data for 18 min). At the deadline the socket is
    shut down and the request counts as a network error. Each attempt writes
    its own partial file, so an abandoned worker cannot touch the next try.
    """
    body = {'model': 'sam-3.1', 'stream': True, 'metadata': {'mask_encoding': 'one_bit'},
            'input': [{'type': 'message', 'role': 'user', 'content': [
                {'type': 'input_text', 'text': 'person'}, {'type': 'input_video', 'file_id': file_id}]}]}
    req = urllib.request.Request(f'{API}/responses', json.dumps(body).encode(),
                                 {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    partial = out.with_name(f'{out.stem}.{uuid.uuid4().hex[:8]}.partial')
    held, result = {}, {}
    worker = threading.Thread(target=_stream, args=(req, partial, held, result), daemon=True)
    worker.start()
    worker.join(DEADLINE_S)
    if worker.is_alive():
        r = held.get('r')
        if r is not None:
            try:
                r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
        return {'type': 'network_error', 'error': {'message': f'no answer within {DEADLINE_S} s'}}
    if 'cache_error' in result:
        return {'type': 'client_error', 'error': {'message': f'cache file: {result["cache_error"]}'}}
    if 'error' in result:
        e = result['error']
        return {'type': 'network_error' if is_network_error(e) else 'client_error', 'error': {'message': str(e)}}
    final = result.get('final')
    if (final or {}).get('response', {}).get('status') == 'completed':
        partial.replace(out)
    return final


def log(ledger, entry):
    d = json.loads(ledger.read_text())
    d['requests'].append(entry)
    if entry.get('nominal_cost') is not None:
        d['nominal_total'] = round(d['nominal_total'] + entry['nominal_cost'], 4)
    tmp = ledger.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(ledger)


def chunks(n, size):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('out')
    ap.add_argument('--ledger', required=True)
    ap.add_argument('--chunk', type=int, default=100)
    ap.add_argument('--pause', type=float, default=3)
    ap.add_argument('--budget', type=float, default=5.0)
    ap.add_argument('--start', type=int, default=0, help='first frame to cover')
    ap.add_argument('--end', type=int, help='frame after the last to cover')
    ap.add_argument('--network-resumes', type=int, default=5, help='network errors to wait out per run')
    ap.add_argument('--network-wait', type=float, default=60)
    a = ap.parse_args()
    video, out, ledger = Path(a.video), Path(a.out), Path(a.ledger)
    out.mkdir(parents=True, exist_ok=True)
    plan = [(s, e) for s, e in chunks(frame_count(video), a.chunk) if s >= a.start and (a.end is None or e <= a.end)]
    todo = [(s, e) for s, e in plan if not (out / f'chunk-{s:05d}-{e:05d}.sse').exists()]
    cost = sum(max(e - s, MIN_FRAMES) * USD_PER_FRAME for s, e in todo)
    print(f'{len(plan)} chunks, {len(todo)} to run, nominal ${cost:.2f}', flush=True)
    if cost > a.budget + 1e-9:
        sys.exit(f'nominal ${cost:.2f} over budget ${a.budget:.2f}')
    token = key() if todo else None
    resumes = 0
    for i, (s, e) in enumerate(todo):
        if i:
            time.sleep(a.pause)
        name = f'chunk-{s:05d}-{e:05d}'
        clip = out / f'{name}.mp4'
        if not clip.exists():
            cut(video, s, e, clip)
        while True:
            t0 = time.time()
            try:
                file_id = upload(clip, token)
            except Exception as err:
                if not is_network_error(err):
                    raise
                file_id, final = None, {'type': 'network_error', 'error': {'message': f'upload: {err}'}}
            else:
                final = segment(file_id, token, out / f'{name}.sse')
            resp = (final or {}).get('response', {})
            billed = (resp.get('usage') or {}).get('video_frames_processed')
            ok = resp.get('status') == 'completed'
            if file_id:  # a request was sent: log it
                log(ledger, {'model': 'sam-3.1', 'kind': 'video', 'frames_billed': billed,
                             'nominal_cost': round(billed * USD_PER_FRAME, 4) if billed is not None else None,
                             'worst_case_cost': round((e - s) * USD_PER_FRAME, 4),
                             'status': 'completed' if ok else f'failed: {(final or {}).get("type")} {resp.get("error") or (final or {}).get("error")}',
                             'artifact': f'{out}/{name}.sse (frames {s}-{e - 1} of {video.name}, {file_id})'})
            print(f'{name}: {"ok" if ok else "FAILED"} billed={billed} {time.time() - t0:.0f}s', flush=True)
            if ok:
                break
            if (final or {}).get('type') == 'network_error' and resumes < a.network_resumes:
                resumes += 1
                print(f'network error {resumes}/{a.network_resumes}, retrying {name} in {a.network_wait:.0f}s: '
                      f'{final["error"]["message"]}', flush=True)
                time.sleep(a.network_wait)
                continue
            sys.exit(1)
        clip.unlink()  # the source video can always cut it again
    print('all chunks done', flush=True)


if __name__ == '__main__':
    main()
