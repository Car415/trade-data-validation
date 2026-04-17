import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trade_validation_poc.mock_data import create_mock_datasets


class MockDataTests(unittest.TestCase):
    def test_create_mock_datasets_returns_expected_shapes(self) -> None:
        corrected, non_corrected = create_mock_datasets(
            corrected_count=6,
            non_corrected_count=10,
            seed=7,
        )

        self.assertEqual(len(corrected), 6)
        self.assertEqual(len(non_corrected), 10)
        self.assertTrue(
            {"trade_id", "corrective_report_type", "input_message", "elig_terms", "sub_terms"}
            .issubset(corrected.columns)
        )
        self.assertTrue(
            {"trade_id", "corrective_report_type", "input_message", "elig_terms", "sub_terms"}
            .issubset(non_corrected.columns)
        )
        self.assertTrue((corrected["corrective_report_type"] == "MISS_REPORTING").all())
        self.assertTrue(non_corrected["corrective_report_type"].isna().all())
        sample_payload = json.loads(corrected.loc[0, "input_message"])
        self.assertIn("TradeCorrectiveInfo", sample_payload)


if __name__ == "__main__":
    unittest.main()
