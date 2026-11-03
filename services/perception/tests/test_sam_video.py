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


class NoHeaders(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        time.sleep(5)  # never sends a status line

    def log_message(self, *args):
        pass


class Trickle(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        try:
            for _ in range(20):  # a keep-alive line every 0.2 s: each read is fast, the request never ends
                self.wfile.write(b': ping\n')
                self.wfile.flush()
                time.sleep(.2)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client cut the request, as it should

    def log_message(self, *args):
        pass


class CompletesThenHangs(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {"type":"response.output_text.delta","delta":"<0f>"}\n')
        self.wfile.write(b'data: {"type":"response.completed","response":{"status":"completed",'
                         b'"usage":{"video_frames_processed":100}}}\n')
        self.wfile.flush()
        time.sleep(5)  # keeps the connection open after the final event

    def log_message(self, *args):
        pass


class DeadlineTests(unittest.TestCase):
    def run_against(self, handler, keep=None):
        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        old = sam_video.API, sam_video.DEADLINE_S, sam_video.STALL_S
        sam_video.API, sam_video.DEADLINE_S, sam_video.STALL_S = f'http://127.0.0.1:{srv.server_address[1]}', 1, 30
        try:
            with tempfile.TemporaryDirectory() as tmp:
                start = time.time()
                final = sam_video.segment('file-x', 'key', Path(tmp) / 'chunk.sse')
                took = time.time() - start
                if keep is not None:
                    keep['saved'] = (Path(tmp) / 'chunk.sse').exists()
                return final, took
        finally:
            sam_video.API, sam_video.DEADLINE_S, sam_video.STALL_S = old
            srv.shutdown()

    def test_completed_response_is_kept_when_connection_stays_open(self):
        keep = {}
        final, took = self.run_against(CompletesThenHangs, keep)
        self.assertLess(took, 1)
        self.assertEqual(final['response']['status'], 'completed')
        self.assertTrue(keep['saved'])

    def test_stream_that_never_ends_is_cut_by_the_deadline(self):
        final, took = self.run_against(Trickle)
        self.assertLess(took, 2)
        self.assertEqual(final['type'], 'network_error')

    def test_server_that_never_answers_is_cut_by_the_deadline(self):
        final, took = self.run_against(NoHeaders)
        self.assertLess(took, 4)
        self.assertEqual(final['type'], 'network_error')

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
