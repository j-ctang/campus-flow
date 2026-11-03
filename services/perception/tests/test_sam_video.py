import http.client
import http.server
import threading
import time
import socket
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scout'))
import sam_video


class NetworkErrorTests(unittest.TestCase):
    def test_connection_trouble_is_network(self):
        for e in (TimeoutError('The read operation timed out'), ConnectionResetError(54, 'Connection reset by peer'),
                  BrokenPipeError(32, 'Broken pipe'), urllib.error.URLError(BrokenPipeError(32, 'Broken pipe')),
                  socket.timeout('timed out'), http.client.RemoteDisconnected('Remote end closed')):
            self.assertTrue(sam_video.is_network_error(e), e)

    def test_api_answers_are_not_network(self):
        e = urllib.error.HTTPError('https://api', 429, 'Too Many Requests', {}, None)
        self.assertFalse(sam_video.is_network_error(e))
        self.assertFalse(sam_video.is_network_error(ValueError('bad json')))


class HungStream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {"type":"response.created"}\n')
        self.wfile.flush()
        time.sleep(5)  # open socket, no data

    def log_message(self, *args):
        pass


class DeadlineTests(unittest.TestCase):
    def test_hung_stream_is_cut_and_counts_as_network_error(self):
        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), HungStream)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        old = sam_video.API, sam_video.DEADLINE_S, sam_video.STALL_S
        sam_video.API, sam_video.DEADLINE_S, sam_video.STALL_S = f'http://127.0.0.1:{srv.server_address[1]}', 1, 30
        try:
            with tempfile.TemporaryDirectory() as tmp:
                start = time.time()
                final = sam_video.segment('file-x', 'key', Path(tmp) / 'chunk.sse')
                self.assertLess(time.time() - start, 4)
                self.assertFalse((Path(tmp) / 'chunk.sse').exists())
        finally:
            sam_video.API, sam_video.DEADLINE_S, sam_video.STALL_S = old
            srv.shutdown()
        self.assertEqual(final['type'], 'network_error')


if __name__ == '__main__':
    unittest.main()
