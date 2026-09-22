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

from src.features.labels import LabelConfig, build_labels

START = datetime(2026, 9, 18, tzinfo=timezone.utc)
INTERVAL = 30


def grid(service: str, ticks: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [START + timedelta(seconds=i * INTERVAL) for i in range(ticks)],
            "service": service,
        }
    )


def truth(service: str, start_tick: int, end_tick: int, incident_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": incident_id,
                "service": service,
                "scenario_type": "memory_leak",
                "start_ts": START + timedelta(seconds=start_tick * INTERVAL),
                "end_ts": START + timedelta(seconds=end_tick * INTERVAL),
            }
        ]
    )


class InFailureTests(unittest.TestCase):
    def test_active_window(self) -> None:
        labels = build_labels(grid("svc-a"), truth("svc-a", 20, 30))
        failing = labels["in_failure"].to_numpy()
        self.assertEqual(failing[19], 0)
        self.assertTrue(failing[20:30].all())
        self.assertEqual(failing[30], 0)

    def test_service_isolation(self) -> None:
        timestamps = pd.concat([grid("svc-a"), grid("svc-b")], ignore_index=True)
        labels = build_labels(timestamps, truth("svc-a", 20, 30))
        b = labels[labels["service"] == "svc-b"]["in_failure"].to_numpy()
        self.assertEqual(int(b.sum()), 0)


class FutureLabelTests(unittest.TestCase):
    def test_forward_window_includes_current(self) -> None:
        labels = build_labels(grid("svc-a"), truth("svc-a", 20, 30))
        label = labels["failure_in_next_10min"].to_numpy()
        ticks = LabelConfig().window_ticks(10)
        self.assertEqual(ticks, 20)
        self.assertEqual(label[0], 0)
        self.assertEqual(label[1], 1)
        self.assertEqual(label[20], 1)
        self.assertEqual(label[29], 1)
        self.assertEqual(label[30], 0)

    def test_exclude_current(self) -> None:
        config = LabelConfig(include_current_failure=False)
        labels = build_labels(grid("svc-a"), truth("svc-a", 20, 30), config)
        label = labels["failure_in_next_10min"].to_numpy()
        self.assertEqual(label[28], 1)
        self.assertEqual(label[29], 0)
        self.assertEqual(label[30], 0)

    def test_multiple_incidents(self) -> None:
        first = truth("svc-a", 20, 30, incident_id=1)
        second = truth("svc-a", 45, 50, incident_id=2)
        labels = build_labels(grid("svc-a"), pd.concat([first, second], ignore_index=True))
        failing = labels["in_failure"].to_numpy()
        self.assertTrue(failing[20:30].all())
        self.assertTrue(failing[45:50].all())
        self.assertEqual(int(failing[30:45].sum()), 0)


class TimeToFailureTests(unittest.TestCase):
    def test_counts_down_then_zero(self) -> None:
        labels = build_labels(grid("svc-a"), truth("svc-a", 20, 30))
        ttf = labels["time_to_failure_min"].to_numpy()
        self.assertAlmostEqual(ttf[0], 10.0)
        self.assertAlmostEqual(ttf[19], 0.5)
        self.assertEqual(ttf[25], 0.0)
        self.assertTrue(np.isnan(ttf[40]))

    def test_other_service_nan_without_incidents(self) -> None:
        timestamps = pd.concat([grid("svc-a"), grid("svc-b")], ignore_index=True)
        labels = build_labels(timestamps, truth("svc-a", 20, 30))
        b = labels[labels["service"] == "svc-b"]["time_to_failure_min"].to_numpy()
        self.assertTrue(np.isnan(b).all())


class ValidationTests(unittest.TestCase):
    def test_missing_columns_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_labels(pd.DataFrame({"timestamp": []}), truth("svc-a", 1, 2))


if __name__ == "__main__":
    unittest.main()
