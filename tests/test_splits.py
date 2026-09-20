from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.features.splits import SplitConfig, time_based_split

START = datetime(2026, 9, 18, tzinfo=timezone.utc)


def frame(ticks: int = 100, services: tuple[str, ...] = ("svc-a",)) -> pd.DataFrame:
    rows = []
    for service in services:
        for i in range(ticks):
            rows.append(
                {
                    "timestamp": START + timedelta(seconds=i * 30),
                    "service": service,
                    "temperature": float(i),
                    "failure_in_next_10min": 1 if i >= 80 else 0,
                }
            )
    return pd.DataFrame(rows)


class SplitBoundaryTests(unittest.TestCase):
    def test_chronological_no_overlap(self) -> None:
        splits, report = time_based_split(frame(), SplitConfig(purge_min=0))
        self.assertTrue(report.chronological)
        self.assertEqual(report.overlap_rows, 0)
        self.assertLess(splits["train"]["timestamp"].max(), splits["val"]["timestamp"].min())
        self.assertLess(splits["val"]["timestamp"].max(), splits["test"]["timestamp"].min())

    def test_counts_without_purge(self) -> None:
        splits, report = time_based_split(frame(), SplitConfig(purge_min=0))
        self.assertEqual(report.counts, {"train": 60, "val": 20, "test": 20})

    def test_purge_creates_gap(self) -> None:
        splits, report = time_based_split(frame(), SplitConfig(purge_min=2.5))
        self.assertEqual(report.purge_ticks, 5)
        self.assertEqual(report.counts, {"train": 55, "val": 15, "test": 20})
        gap_after_train = splits["val"]["timestamp"].min() - splits["train"]["timestamp"].max()
        self.assertEqual(gap_after_train, timedelta(seconds=6 * 30))

    def test_all_services_present_in_every_split(self) -> None:
        splits, _ = time_based_split(frame(services=("svc-a", "svc-b", "svc-c")), SplitConfig(purge_min=0))
        for part in splits.values():
            self.assertEqual(part["service"].nunique(), 3)


class SplitReportTests(unittest.TestCase):
    def test_positive_rates(self) -> None:
        _, report = time_based_split(
            frame(), SplitConfig(purge_min=0), label_columns=["failure_in_next_10min"]
        )
        self.assertEqual(report.positive_rates["train"]["failure_in_next_10min"], 0.0)
        self.assertEqual(report.positive_rates["test"]["failure_in_next_10min"], 1.0)

    def test_too_few_timestamps_raises(self) -> None:
        with self.assertRaises(ValueError):
            time_based_split(frame(ticks=2), SplitConfig(purge_min=0))


if __name__ == "__main__":
    unittest.main()
