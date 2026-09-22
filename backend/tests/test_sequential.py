import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.sequential import (
    SequentialConfig,
    SequentialLSTMClassifier,
    build_sequences,
    build_sequences_for_splits,
)


def make_service_df(n_rows=30, n_features=2, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-09-18", periods=n_rows, freq="30s")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "service": "svc-a",
            **{f"f{i}": rng.normal(size=n_rows) for i in range(n_features)},
            "failure_in_next_10min": rng.integers(0, 2, n_rows),
        }
    )


class TestBuildSequences(unittest.TestCase):
    def test_shapes_and_windows_per_service(self):
        blocks = [
            make_service_df(30, seed=1).assign(service="a"),
            make_service_df(30, seed=2).assign(service="b"),
            make_service_df(12, seed=3).assign(service="c"),
        ]
        df = pd.concat(blocks, ignore_index=True)
        X, y, meta = build_sequences(df, ["f0", "f1"], seq_len=5)
        self.assertEqual(X.shape, (26 + 26 + 8, 5, 2))
        self.assertEqual(len(y), X.shape[0])
        self.assertEqual(len(meta), X.shape[0])
        self.assertEqual(meta.columns.tolist(), ["timestamp", "service"])

    def test_target_is_last_window_step(self):
        labels = np.arange(20)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-09-18", periods=20, freq="30s"),
                "service": "s",
                "f0": labels,
                "f1": labels * 2,
                "failure_in_next_10min": labels,
            }
        )
        X, y, meta = build_sequences(df, ["f0", "f1"], seq_len=4)
        self.assertEqual(X.shape, (17, 4, 2))
        np.testing.assert_array_equal(y, labels[3:20])
        np.testing.assert_allclose(X[:, -1, 0], labels[3:20])
        np.testing.assert_allclose(X[:, 0, 0], labels[0:17])

    def test_sequences_stay_within_service(self):
        a = make_service_df(15, seed=1).assign(service="s1")
        b = make_service_df(20, seed=2).assign(service="s2")
        df = pd.concat([a, b], ignore_index=True)
        X, y, meta = build_sequences(df, ["f0", "f1"], seq_len=4)
        counts = meta["service"].value_counts().to_dict()
        self.assertEqual(counts, {"s1": 15 - 3, "s2": 20 - 3})

    def test_short_group_skipped(self):
        df = make_service_df(3, seed=1).assign(service="tiny")
        X, y, meta = build_sequences(df, ["f0", "f1"], seq_len=5)
        self.assertEqual(X.shape[0], 0)
        self.assertEqual(len(meta), 0)


class TestSequentialModel(unittest.TestCase):
    def _labeled_df(self, n_rows=120, n_pos=14, seed=1):
        rng = np.random.default_rng(seed)
        ts = pd.date_range("2026-09-18", periods=n_rows, freq="30s")
        labels = np.zeros(n_rows, dtype=int)
        labels[:n_pos] = 1
        return pd.DataFrame(
            {
                "timestamp": ts,
                "service": "svc-a",
                "f0": rng.normal(size=n_rows),
                "f1": rng.normal(size=n_rows),
                "f2": rng.normal(size=n_rows),
                "failure_in_next_10min": labels,
            }
        )

    def test_fit_predict_and_save_load(self):
        train = self._labeled_df(120, n_pos=14, seed=1)
        val = self._labeled_df(40, n_pos=6, seed=2)
        test = self._labeled_df(40, n_pos=8, seed=3)
        cfg = SequentialConfig(seq_len=6, lstm_units=8, epochs=3, batch_size=32, patience=3)
        X_tr, y_tr, _ = build_sequences(train, ["f0", "f1", "f2"], seq_len=6)
        X_va, y_va, _ = build_sequences(val, ["f0", "f1", "f2"], seq_len=6)
        X_te, y_te, _ = build_sequences(test, ["f0", "f1", "f2"], seq_len=6)

        model = SequentialLSTMClassifier(cfg).fit(X_tr, y_tr, X_va, y_va)
        self.assertEqual(model.class_weight_[0], 1.0)
        self.assertGreater(model.class_weight_[1], 1.0)

        probs = model.predict_proba(X_te)
        self.assertEqual(probs.shape, (len(y_te),))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))
        self.assertTrue(np.all(np.isfinite(probs)))

        ev = model.evaluate(X_te, y_te, threshold=0.5)
        self.assertIn("metrics", ev)
        self.assertIn("roc_auc", ev["metrics"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"
            model.save(path)
            loaded = SequentialLSTMClassifier.load(path)
            loaded_probs = loaded.predict_proba(X_te)
            np.testing.assert_allclose(probs, loaded_probs, atol=1e-6)

    def test_split_builder_keys(self):
        splits = {"train": make_service_df(40, seed=1), "val": make_service_df(30, seed=2)}
        seq = build_sequences_for_splits(splits, ["f0", "f1"], seq_len=5)
        self.assertEqual(set(seq.keys()), {"train", "val"})
        self.assertEqual(seq["train"][0].shape, (40 - 4, 5, 2))


if __name__ == "__main__":
    unittest.main()