"""Regressions for concurrent web sessions opening a persistent vector index."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vector_store import _cached_client, _client


class VectorClientTests(unittest.TestCase):
    def setUp(self):
        _cached_client.cache_clear()

    def tearDown(self):
        _cached_client.cache_clear()

    def test_concurrent_first_requests_share_one_fully_started_client(self):
        ready, release, duplicate = threading.Event(), threading.Event(), threading.Event()
        barrier = threading.Barrier(8)
        client = object()

        def construct(**kwargs):
            if ready.is_set():
                duplicate.set()
            ready.set()
            if not release.wait(5):
                raise TimeoutError("constructor was not released")
            return client

        def connect():
            barrier.wait(timeout=5)
            return _client(ROOT / ".local" / "concurrent-client-test")

        with patch("chromadb.PersistentClient", side_effect=construct) as factory:
            with ThreadPoolExecutor(max_workers=8) as executor:
                requests = [executor.submit(connect) for _ in range(8)]
                try:
                    self.assertTrue(ready.wait(5))
                    self.assertFalse(duplicate.wait(0.15))
                finally:
                    release.set()
                self.assertTrue(all(request.result(timeout=5) is client for request in requests))
            factory.assert_called_once()

    def test_equivalent_paths_share_one_client(self):
        folder = ROOT / ".local" / "canonical-client-test"
        alias = str(folder) + os.sep + "." + os.sep
        with patch("chromadb.PersistentClient", return_value=object()) as factory:
            self.assertIs(_client(folder), _client(alias))
            factory.assert_called_once_with(path=os.path.normcase(str(folder.resolve())))

    def test_failed_construction_can_be_retried(self):
        client = object()
        with patch("chromadb.PersistentClient", side_effect=[RuntimeError("startup failed"), client]) as factory:
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                _client(ROOT / ".local" / "retry-client-test")
            self.assertIs(_client(ROOT / ".local" / "retry-client-test"), client)
            self.assertIs(_client(ROOT / ".local" / "retry-client-test"), client)
            self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
