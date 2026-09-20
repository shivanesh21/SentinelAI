from __future__ import annotations

import shutil
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.anomaly import DenseAutoencoderDetector, LSTMAutoencoderDetector

START = datetime(2026, 9, 18, tzinfo=timezone.utc)
FEATURES = ["f1", "f2", "f3"]


def make_frame(ticks: int = 200, service: str = "svc-a", seed: int = 0, outlier_index: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "timestamp": [START + timedelta(seconds=i * 30) for i in range(ticks)],
            "service": service,
            "f1": rng.normal(0, 1, ticks),
            "f2": rng.normal(10, 2, ticks),
            "f3": rng.normal(50, 5, ticks),
            "in_failure": 0,
            "failure_in_next_10min": 0,
        }
    )
    if outlier_index is not None:
        frame.loc[outlier_index, ["f1", "f2", "f3"]] = [50.0, 120.0, 500.0]
    return frame


class DenseAutoencoderTests(unittest.TestCase):
    def test_outlier_scores_higher(self) -> None:
        detector = DenseAutoencoderDetector(hidden_dims=(4,), epochs=3, batch_size=32, seed=1)
        detector.fit(make_frame(200), FEATURES)
        test = make_frame(40, outlier_index=20)
        scores = detector.score(test)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[20], scores[:19].max())
        self.assertIsNotNone(detector.threshold_)

    def test_normal_mask_excludes_failures(self) -> None:
        frame = make_frame(10)
        frame.loc[3, "in_failure"] = 1
        frame.loc[7, "failure_in_next_10min"] = 1
        detector = DenseAutoencoderDetector()
        mask = detector._normal_mask(frame)
        self.assertEqual(mask.sum(), 8)
        self.assertFalse(mask[3])
        self.assertFalse(mask[7])

    def test_save_load_roundtrip(self) -> None:
        detector = DenseAutoencoderDetector(hidden_dims=(4,), epochs=3, batch_size=32, seed=1)
        detector.fit(make_frame(200), FEATURES)
        test = make_frame(50)
        scores = detector.score(test)

        path = Path(ROOT) / "models" / "anomaly" / "_test_dense_ae"
        detector.save(path)
        try:
            loaded = DenseAutoencoderDetector.load(path)
            np.testing.assert_allclose(loaded.score(test), scores, rtol=1e-4, atol=1e-6)
        finally:
            shutil.rmtree(path, ignore_errors=True)


class LSTMAutoencoderTests(unittest.TestCase):
    def test_scores_finite_and_outlier_ranked(self) -> None:
        detector = LSTMAutoencoderDetector(seq_len=10, lstm_units=8, epochs=2, batch_size=32, seed=1)
        detector.fit(make_frame(140), FEATURES)
        test = make_frame(60, outlier_index=40)
        scores = detector.score(test)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(scores), len(test))
        self.assertGreater(scores[40], np.median(scores))


if __name__ == "__main__":
    unittest.main()
