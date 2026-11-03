import http.client
import socket
import sys
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


if __name__ == '__main__':
    unittest.main()
