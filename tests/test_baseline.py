import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline import (
    BaselineModelConfig,
    evaluate_binary,
    find_optimal_threshold,
    load_model,
    save_model,
    train_baseline,
)


class TestBaselineModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 1000
        self.feature_columns = ["f1", "f2", "f3"]
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "failure_in_next_10min": np.random.binomial(1, 0.1, n),
        })
        split = int(0.7 * n)
        self.train = df.iloc[:split].copy()
        self.val = df.iloc[split:].copy()

    def test_logistic_regression_train(self):
        config = BaselineModelConfig(name="logistic_regression")
        pipe, report = train_baseline(self.train, self.val, self.feature_columns, "failure_in_next_10min", config)
        self.assertIn("f1", report.metrics)
        self.assertGreaterEqual(report.metrics["roc_auc"], 0.0)
        self.assertLessEqual(report.metrics["roc_auc"], 1.0)

    def test_random_forest_train(self):
        config = BaselineModelConfig(name="random_forest")
        pipe, report = train_baseline(self.train, self.val, self.feature_columns, "failure_in_next_10min", config)
        self.assertIn("f1", report.metrics)
        self.assertIsNotNone(report.feature_importance)
        self.assertEqual(len(report.feature_importance), 3)

    def test_evaluate_binary(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.7, 0.6])
        metrics = evaluate_binary(y_true, y_pred, y_score)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)

    def test_find_optimal_threshold(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.7, 0.6])
        threshold = find_optimal_threshold(y_true, y_score)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_save_load_model(self):
        config = BaselineModelConfig(name="logistic_regression")
        pipe, report = train_baseline(self.train, self.val, self.feature_columns, "failure_in_next_10min", config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"
            save_model(pipe, path, report.threshold, self.feature_columns, meta={"test": True})
            loaded_pipe, loaded_threshold, loaded_features, loaded_meta = load_model(path)

            self.assertEqual(loaded_threshold, report.threshold)
            self.assertEqual(loaded_features, self.feature_columns)
            self.assertEqual(loaded_meta.get("test"), True)

            X_val = self.val[self.feature_columns].to_numpy(dtype=float)
            orig_scores = pipe.predict_proba(X_val)[:, 1]
            loaded_scores = loaded_pipe.predict_proba(X_val)[:, 1]
            np.testing.assert_allclose(orig_scores, loaded_scores)


if __name__ == "__main__":
    unittest.main()