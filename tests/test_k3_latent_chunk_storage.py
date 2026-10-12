"""Chunk page ownership and admission-shape regressions."""
import unittest
import numpy as np

from openagent.adaptive_engine.k3_execution.cache import MLALatentPages


class K3LatentChunkStorage(unittest.TestCase):
    def test_chunk_matches_incremental_append(self):
        for old_length in (0, 1, 3, 4, 7):
            for new_length in (1, 3, 4, 9):
                with self.subTest(old_length=old_length, new_length=new_length):
                    values = np.arange((old_length+new_length)*5, dtype=np.float32).reshape(-1, 5)
                    reference = MLALatentPages(3, 2, page_tokens=4)
                    candidate = MLALatentPages(3, 2, page_tokens=4)
                    for row in values[:old_length]:
                        reference.append(row[:3], row[3:])
                        candidate.append(row[:3], row[3:])
                    for row in values[old_length:]:
                        reference.append(row[:3], row[3:])
                    candidate.append_many(values[old_length:, :3], values[old_length:, 3:])
                    self.assertEqual(reference.length, candidate.length)
                    self.assertEqual([len(p) for p in reference.pages], [len(p) for p in candidate.pages])
                    for expected, actual in zip(reference.pages, candidate.pages):
                        np.testing.assert_array_equal(actual, expected)
                        self.assertFalse(actual.flags.writeable)

    def test_fork_and_input_buffers_remain_independent(self):
        cache = MLALatentPages(2, 0, page_tokens=4)
        original = np.ones((3, 2), np.float32)
        cache.append_many(original, np.empty((3, 0), np.float32))
        fork = cache.fork()
        fork.append_many(np.full((5, 2), 2, np.float32), np.empty((5, 0), np.float32))
        original[:] = 99
        self.assertEqual(cache.length, 3)
        self.assertEqual(fork.length, 8)
        np.testing.assert_array_equal(cache.pages[0], np.ones((3, 2), np.float32))
        np.testing.assert_array_equal(fork.pages[0][:3], cache.pages[0])

    def test_invalid_chunk_does_not_publish_pages(self):
        cache = MLALatentPages(2, 1, page_tokens=4)
        cache.append(np.ones(2, np.float32), np.ones(1, np.float32))
        before = cache.pages
        for latent, rotary in (
            (np.ones((2, 2), np.float64), np.ones((2, 1), np.float32)),
            (np.ones((2, 2), np.float32), np.ones((1, 1), np.float32)),
            (np.full((2, 2), np.nan, np.float32), np.ones((2, 1), np.float32)),
        ):
            with self.assertRaises(ValueError):
                cache.append_many(latent, rotary)
            self.assertIs(cache.pages, before)
            self.assertEqual(cache.length, 1)
