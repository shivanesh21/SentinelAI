import os
import tempfile
import unittest
from pathlib import Path

from src.mlops import (
    DEFAULT_REGISTRY_PATH,
    Experiment,
    ExperimentStore,
    dataset_version,
    registry_from_settings,
    write_stamp,
)


META = {
    "name": "feature_table",
    "n_rows": 14200,
    "n_services": 5,
    "feature_columns": ["Error_rate", "Latency_15min_avg"],
    "start_ts": "2026-09-01",
    "end_ts": "2026-09-10",
}
SPLIT = {"counts": {"train": 8540, "val": 2780, "test": 2880}, "chronological": True}


def _experiment(family="baseline", model="random_forest", metric_val=0.5):
    return Experiment(
        family=family,
        model=model,
        hyperparameters={"n_estimators": 200, "max_depth": 12},
        metrics={
            "val": {"f1": metric_val, "roc_auc": metric_val + 0.1},
            "test": {"f1": metric_val, "roc_auc": metric_val + 0.1},
            "threshold": 0.5,
        },
        dataset_version="ds-test1234",
        model_version=0,
        model_path="models/baseline/random_forest",
    )


class TestDatasetVersion(unittest.TestCase):
    def test_deterministic(self):
        v1 = dataset_version(META, SPLIT)
        v2 = dataset_version(META, SPLIT)
        self.assertEqual(v1, v2)
        self.assertTrue(v1.startswith("ds-"))
        self.assertEqual(len(v1), 12 + 3)

    def test_changes_with_meta(self):
        changed = dict(META, n_rows=99999)
        self.assertNotEqual(dataset_version(META, SPLIT), dataset_version(changed, SPLIT))

    def test_changes_with_split_report(self):
        changed = {"counts": {"train": 1000}, "chronological": True}
        self.assertNotEqual(dataset_version(META, SPLIT), dataset_version(META, changed))

    def test_empty_inputs(self):
        v = dataset_version(None, None)
        self.assertTrue(v.startswith("ds-"))


class TestRegistryPath(unittest.TestCase):
    def test_relative_resolves_under_base(self):
        path = registry_from_settings({"mlops": {"registry": "data/exp.jsonl"}}, Path("C:/base"))
        self.assertEqual(path, Path("C:/base") / "data" / "exp.jsonl")

    def test_absolute_kept(self):
        path = registry_from_settings({"mlops": {"registry": "C:/abs/exp.jsonl"}}, Path("C:/base"))
        self.assertEqual(path, Path("C:/abs/exp.jsonl"))

    def test_default(self):
        path = registry_from_settings({}, Path("C:/base"))
        self.assertEqual(path, Path("C:/base") / DEFAULT_REGISTRY_PATH)


class TestExperimentStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ExperimentStore(os.path.join(self.tmp.name, "experiments.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_assigns_incrementing_versions(self):
        first = self.store.record(_experiment())
        second = self.store.record(_experiment(metric_val=0.7))
        self.assertEqual(first.model_version, 1)
        self.assertEqual(second.model_version, 2)

    def test_read_filters_and_limit(self):
        self.store.record(_experiment())
        self.store.record(_experiment(family="advanced", model="xgboost", metric_val=0.6))
        self.assertEqual(len(self.store.read()), 2)
        self.assertEqual(len(self.store.read(family="baseline")), 1)
        self.assertEqual(len(self.store.read(model="xgboost")), 1)
        self.assertEqual(self.store.read(limit=0), [])

    def test_get_latest_best(self):
        self.store.record(_experiment(metric_val=0.4))
        self.store.record(_experiment(metric_val=0.8))
        self.store.record(_experiment(metric_val=0.6))
        self.assertEqual(self.store.get("does-not-exist"), None)
        self.assertEqual(self.store.latest("baseline", "random_forest")["metrics"]["test"]["f1"], 0.6)
        self.assertEqual(self.store.best("baseline", "random_forest", "f1", "test")["metrics"]["test"]["f1"], 0.8)

    def test_failed_records_excluded_from_best(self):
        self.store.record(_experiment(metric_val=0.9))
        failed = _experiment(metric_val=0.99)
        failed.status = "failed"
        self.store.record(failed)
        self.assertEqual(
            self.store.best("baseline", "random_forest", "f1", "test")["metrics"]["test"]["f1"], 0.9
        )

    def test_compare_ranks_by_metric(self):
        self.store.record(_experiment(metric_val=0.4))
        self.store.record(_experiment(family="advanced", model="xgboost", metric_val=0.9))
        rows = self.store.compare("f1", "test")
        self.assertEqual(rows[0]["model"], "xgboost")
        self.assertEqual(rows[0]["value"], 0.9)
        self.assertEqual(rows[1]["model"], "random_forest")

    def test_models_list(self):
        self.store.record(_experiment())
        self.store.record(_experiment(family="advanced", model="xgboost"))
        self.assertEqual(self.store.models(), ["advanced/xgboost", "baseline/random_forest"])


class TestWriteStamp(unittest.TestCase):
    def test_stamp_writes_version_and_dataset(self):
        tmp = tempfile.TemporaryDirectory()
        ex = _experiment()
        ex.model_version = 3
        path = write_stamp(tmp.name, ex, "data/experiments.jsonl")
        stamp = path.read_text(encoding="utf-8")
        self.assertIn('"model_version": 3', stamp)
        self.assertIn("ds-test1234", stamp)
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()