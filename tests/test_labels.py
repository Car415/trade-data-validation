import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trade_validation_poc.labels import (
    build_label_frames,
    extract_overridden_keys,
)


class LabelTests(unittest.TestCase):
    def test_extract_overridden_keys_reads_nested_corrective_meta(self) -> None:
        payload = """
        {
          "TradeCorrectiveInfo": [
            {
              "tradeID": "abc123",
              "CorrectiveMeta": {
                "counterparty_lei": "529900CORRECTED",
                "notional": "1500000"
              }
            }
          ]
        }
        """

        keys = extract_overridden_keys(payload)

        self.assertEqual(keys, {"counterparty_lei", "notional"})

    def test_build_label_frames_marks_corrected_and_non_corrected_rows(self) -> None:
        corrected = pd.DataFrame(
            [
                {
                    "trade_id": "T1",
                    "input_message": """
                    {
                      "TradeCorrectiveInfo": [
                        {"CorrectiveMeta": {"notional": "1500000", "currency": "USD"}}
                      ]
                    }
                    """,
                }
            ]
        )
        non_corrected = pd.DataFrame([{"trade_id": "T2"}])

        labels_corrected, labels_non_corrected, target_fields = build_label_frames(
            corrected,
            non_corrected,
            top_n=2,
        )

        self.assertEqual(target_fields, ["notional", "currency"])
        self.assertEqual(labels_corrected.loc[0, "label_notional"], 1)
        self.assertEqual(labels_non_corrected.loc[0, "label_currency"], 0)


if __name__ == "__main__":
    unittest.main()
