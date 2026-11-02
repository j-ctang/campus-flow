"""One SAM 3.1 video request: upload a clip, segment "person", save the stream.

Spending rules: one request per run, no retries, refuses to run if the
output already exists, stops on the first error. The caller logs usage.
The API key is read from MODEL_API_KEY in <repo>/.env and never printed.

Usage: python sam_video.py <video.mp4> <output.sse>
"""
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

API = 'https://api.meta.ai/v1'
ROOT = Path(__file__).resolve().parents[3]


def key():
    for line in (ROOT / '.env').read_text().splitlines():
        if line.startswith('MODEL_API_KEY='):
            return line.split('=', 1)[1].strip()
    sys.exit('MODEL_API_KEY missing from .env')


def upload(path, token):
    boundary = uuid.uuid4().hex
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            'Content-Type: video/mp4\r\n\r\n').encode()
    body = head + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(f'{API}/files', body, {'Authorization': f'Bearer {token}',
                                                        'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req, timeout=900) as r:
        info = json.load(r)
    if not info.get('id'):
        sys.exit(f'upload failed: {info}')
    return info


def segment(file_id, token, out):
    body = {'model': 'sam-3.1', 'stream': True, 'metadata': {'mask_encoding': 'one_bit'},
            'input': [{'type': 'message', 'role': 'user', 'content': [
                {'type': 'input_text', 'text': 'person'}, {'type': 'input_video', 'file_id': file_id}]}]}
    req = urllib.request.Request(f'{API}/responses', json.dumps(body).encode(),
                                 {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    tmp = out.with_suffix(out.suffix + '.partial')
    final = None
    with urllib.request.urlopen(req, timeout=900) as r, open(tmp, 'wb') as f:
        for line in r:
            f.write(line)
            if line.startswith(b'data: '):
                try:
                    ev = json.loads(line[6:])
                except ValueError:
                    continue
                if ev.get('type') in ('response.completed', 'response.failed', 'response.incomplete', 'error'):
                    final = ev
    tmp.replace(out)
    return final


def main():
    video, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        sys.exit(f'{out} exists; refusing to pay twice')
    token = key()
    t0 = time.time()
    info = upload(video, token)
    print(f'uploaded {info["id"]} {info.get("bytes")} bytes in {time.time() - t0:.0f}s', flush=True)
    final = segment(info['id'], token, out)
    resp = (final or {}).get('response', {})
    summary = {'file_id': info['id'], 'event': (final or {}).get('type'), 'status': resp.get('status'),
               'usage': resp.get('usage'), 'error': resp.get('error') or (final or {}).get('error'),
               'seconds': round(time.time() - t0)}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if summary['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
