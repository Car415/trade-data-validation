import json
import random
from typing import Dict, List, Tuple

import pandas as pd


ASSET_TEMPLATES = [
    {
        "asset_class": "IR",
        "product_type": "SWAP",
        "currency": "USD",
        "source_system": "MUREX",
        "base_notional": 1_000_000,
        "fixed_rate": 0.031,
        "float_index": "SOFR",
    },
    {
        "asset_class": "CR",
        "product_type": "CDS",
        "currency": "EUR",
        "source_system": "CALYPSO",
        "base_notional": 500_000,
        "spread": 0.014,
        "reference_entity": "ABC Corp",
    },
]


def _build_trade_terms(index: int, template: Dict[str, object], rng: random.Random) -> Dict[str, str]:
    maturity_year = 2028 + (index % 5)
    notional = template["base_notional"] + (index % 7) * 125_000
    terms = {
        "asset_class": str(template["asset_class"]),
        "product_type": str(template["product_type"]),
        "notional": str(notional),
        "currency": str(template["currency"]),
        "maturity_date": f"{maturity_year}-0{(index % 9) + 1}-15",
        "counterparty_lei": f"CPTY_{(index % 9) + 1:03d}",
        "source_system": str(template["source_system"]),
    }
    if template["asset_class"] == "IR":
        terms["fixed_rate"] = f"{template['fixed_rate'] + (index % 4) * 0.002:.3f}"
        terms["float_index"] = str(template["float_index"])
        terms["payment_freq"] = rng.choice(["1M", "3M", "6M"])
    else:
        terms["spread"] = f"{template['spread'] + (index % 4) * 0.001:.3f}"
        terms["reference_entity"] = str(template["reference_entity"])
        terms["seniority"] = rng.choice(["SENIOR", "SUBORDINATED"])
    return terms


def _choose_override_fields(template: Dict[str, object], index: int) -> List[str]:
    if template["asset_class"] == "IR":
        patterns = [
            ["notional", "fixed_rate"],
            ["currency"],
            ["counterparty_lei", "notional"],
            ["maturity_date"],
        ]
    else:
        patterns = [
            ["notional", "reference_entity"],
            ["currency"],
            ["counterparty_lei"],
            ["maturity_date", "spread"],
        ]
    return patterns[index % len(patterns)]


def _mutate_submission_terms(elig_terms: Dict[str, str], override_fields: List[str], index: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    sub_terms = dict(elig_terms)
    corrected_values: Dict[str, str] = {}
    for field in override_fields:
        corrected_values[field] = elig_terms[field]
        if field == "notional":
            sub_terms[field] = str(int(elig_terms[field]) + 250_000)
        elif field == "fixed_rate":
            sub_terms[field] = f"{float(elig_terms[field]) + 0.01:.3f}"
        elif field == "spread":
            sub_terms[field] = f"{float(elig_terms[field]) + 0.005:.3f}"
        elif field == "currency":
            sub_terms[field] = "GBP" if elig_terms[field] != "GBP" else "USD"
        elif field == "counterparty_lei":
            sub_terms[field] = f"CPTY_{(index % 9) + 20:03d}"
        elif field == "maturity_date":
            sub_terms[field] = f"203{(index % 3) + 1}-12-31"
        elif field == "reference_entity":
            sub_terms[field] = f"{elig_terms[field]} Holdings"
    sub_terms["action_type"] = "NEWT"
    return sub_terms, corrected_values


def _build_input_message(trade_id: str, corrected_values: Dict[str, str]) -> str:
    return json.dumps(
        {
            "TradeCorrectiveInfo": [
                {
                    "tradeID": trade_id,
                    "CorrectiveMeta": corrected_values,
                }
            ]
        }
    )


def create_mock_datasets(
    corrected_count: int = 48,
    non_corrected_count: int = 120,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    corrected_rows = []
    non_corrected_rows = []

    for index in range(corrected_count):
        template = ASSET_TEMPLATES[index % len(ASSET_TEMPLATES)]
        trade_id = f"CORR_{index + 1:04d}"
        elig_terms = _build_trade_terms(index, template, rng)
        override_fields = _choose_override_fields(template, index)
        sub_terms, corrected_values = _mutate_submission_terms(elig_terms, override_fields, index)
        corrected_rows.append(
            {
                "trade_id": trade_id,
                "corrective_report_type": "MISS_REPORTING",
                "input_message": _build_input_message(trade_id, corrected_values),
                "elig_terms": json.dumps(elig_terms),
                "sub_terms": json.dumps(sub_terms),
            }
        )

    for index in range(non_corrected_count):
        template = ASSET_TEMPLATES[(index + 1) % len(ASSET_TEMPLATES)]
        trade_id = f"NORM_{index + 1:04d}"
        elig_terms = _build_trade_terms(index + 100, template, rng)
        sub_terms = dict(elig_terms)
        sub_terms["action_type"] = "NEWT"
        non_corrected_rows.append(
            {
                "trade_id": trade_id,
                "corrective_report_type": None,
                "input_message": None,
                "elig_terms": json.dumps(elig_terms),
                "sub_terms": json.dumps(sub_terms),
            }
        )

    return pd.DataFrame(corrected_rows), pd.DataFrame(non_corrected_rows)
