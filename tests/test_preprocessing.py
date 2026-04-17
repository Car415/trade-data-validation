import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trade_validation_poc.preprocessing import (
    align_feature_columns,
    build_feature_tables,
)


class PreprocessingTests(unittest.TestCase):
    def test_build_feature_tables_flattens_and_converts_known_types(self) -> None:
        corrected = pd.DataFrame(
            [
                {
                    "trade_id": "T1",
                    "elig_terms": '{"asset_class":"IR","notional":"1000000","maturity_date":"2030-01-15"}',
                    "sub_terms": '{"currency":"USD","fixed_rate":"0.035"}',
                }
            ]
        )
        non_corrected = pd.DataFrame(
            [
                {
                    "trade_id": "T2",
                    "elig_terms": '{"asset_class":"CR","notional":"500000"}',
                    "sub_terms": '{"currency":"EUR","reference_entity":"ABC Corp"}',
                }
            ]
        )

        corrected_features = build_feature_tables(corrected)
        non_corrected_features = build_feature_tables(non_corrected)
        aligned_corrected, aligned_non_corrected = align_feature_columns(
            corrected_features,
            non_corrected_features,
        )

        self.assertIn("elig_asset_class", aligned_corrected.columns)
        self.assertIn("sub_currency", aligned_corrected.columns)
        self.assertEqual(aligned_corrected.loc[0, "elig_notional"], 1000000.0)
        self.assertIn("elig_maturity_date_year", aligned_corrected.columns)
        self.assertIn("sub_reference_entity", aligned_non_corrected.columns)
        self.assertTrue(set(aligned_corrected.columns) == set(aligned_non_corrected.columns))


if __name__ == "__main__":
    unittest.main()
