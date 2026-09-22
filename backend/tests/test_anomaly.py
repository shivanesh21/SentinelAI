from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.anomaly import (
    IQRTukeyDetector,
    IsolationForestDetector,
    RollingIQRDetector,
    ZScoreDetector,
    assign_levels,
)

START = datetime(2026, 9, 18, tzinfo=timezone.utc)
FEATURES = ["f1", "f2", "f3"]


def make_frame(ticks: int = 300, service: str = "svc-a", seed: int = 0, outlier_index: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "timestamp": [START + timedelta(seconds=i * 30) for i in range(ticks)],
        "service": service,
        "f1": rng.normal(0, 1, ticks),
        "f2": rng.normal(10, 2, ticks),
        "f3": rng.normal(50, 5, ticks),
    }
    frame = pd.DataFrame(data)
    if outlier_index is not None:
        frame.loc[outlier_index, ["f1", "f2", "f3"]] = [40.0, 100.0, 400.0]
    return frame


class ZScoreTests(unittest.TestCase):
    def test_outlier_flagged(self) -> None:
        train = make_frame()
        test = make_frame(ticks=50, outlier_index=25)
        detector = ZScoreDetector(contamination=0.05).fit(train, FEATURES)
        scores = detector.score(test)
        self.assertGreater(scores[25], 5.0)
        self.assertGreater(scores[25], scores[:24].max())
        detector.fit_threshold(detector.score(train))
        predictions = detector.predict(test)
        self.assertEqual(predictions[25], 1)
        self.assertLessEqual(int(predictions.sum()), 3)


class IQRTests(unittest.TestCase):
    def test_outlier_flagged(self) -> None:
        train = make_frame()
        test = make_frame(ticks=50, outlier_index=25)
        detector = IQRTukeyDetector(contamination=0.05).fit(train, FEATURES)
        scores = detector.score(test)
        self.assertGreater(scores[25], scores[:24].max())
        detector.fit_threshold(detector.score(train))
        self.assertEqual(detector.predict(test)[25], 1)


class RollingIQRTests(unittest.TestCase):
    def test_scores_finite_and_spike_ranked(self) -> None:
        frame = make_frame(ticks=120, outlier_index=90)
        detector = RollingIQRDetector(contamination=0.05, iqr_window_min=15).fit(frame, FEATURES)
        scores = detector.score(frame)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[90], scores[80:89].max())
        self.assertGreater(scores[90], scores[91:100].max())


class IsolationForestTests(unittest.TestCase):
    def test_outlier_and_roundtrip(self) -> None:
        train = make_frame()
        test = make_frame(ticks=50, outlier_index=25)
        detector = IsolationForestDetector(contamination=0.05, n_estimators=100, random_state=1).fit(train, FEATURES)
        scores = detector.score(test)
        self.assertGreater(scores[25], scores[:24].max())
        detector.fit_threshold(scores)

        path = Path(ROOT) / "models" / "anomaly" / "_test_iforest.joblib"
        detector.save(path)
        try:
            loaded = IsolationForestDetector.load(path)
            np.testing.assert_allclose(loaded.score(test), scores)
        finally:
            path.unlink(missing_ok=True)


class LevelTests(unittest.TestCase):
    def test_levels_are_monotonic(self) -> None:
        train_scores = np.linspace(0, 1, 1000)
        scores = np.array([0.1, 0.55, 0.97, 0.999])
        levels = assign_levels(scores, train_scores, contamination=0.05)
        self.assertEqual(levels.tolist(), ["NORMAL", "WARNING", "ANOMALOUS", "CRITICAL"])


if __name__ == "__main__":
    unittest.main()
