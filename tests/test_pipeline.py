import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trade_validation_poc.pipeline import run_mock_poc
from trade_validation_poc.training import binary_auc


class PipelineTests(unittest.TestCase):
    def test_binary_auc_handles_misaligned_indices(self) -> None:
        y_true = pd.Series([1, 0, 1, 0], index=[10, 11, 12, 13])
        y_prob = pd.Series([0.9, 0.2, 0.8, 0.1], index=[0, 1, 2, 3])

        auc = binary_auc(y_true, y_prob)

        self.assertGreater(auc, 0.9)

    def test_run_mock_poc_returns_models_metrics_and_simulation(self) -> None:
        result = run_mock_poc(
            corrected_count=12,
            non_corrected_count=24,
            top_n=3,
            seed=5,
        )

        self.assertIn("dataset", result)
        self.assertIn("models", result)
        self.assertIn("results", result)
        self.assertIn("simulation", result)
        self.assertEqual(len(result["models"]), len(result["target_fields"]))
        self.assertFalse(result["results"].empty)
        self.assertFalse(result["simulation"].empty)
        self.assertTrue(result["results"]["auc"].notna().all())


if __name__ == "__main__":
    unittest.main()
