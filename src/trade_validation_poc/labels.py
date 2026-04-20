import json
from collections import Counter
from typing import List, Set, Tuple

import pandas as pd


def _extract_overridden_key_sequence(input_message: str) -> List[str]:
    # `input_message` is nested JSON. We only care about which keys were
    # overridden, because those keys become our labels: "was this field wrong?"
    if pd.isna(input_message) or not input_message:
        return []
    try:
        payload = json.loads(input_message)
    except (TypeError, json.JSONDecodeError):
        return []

    ordered_keys: List[str] = []
    for item in payload.get("TradeCorrectiveInfo", []):
        corrective_meta = item.get("CorrectiveMeta", {})
        if isinstance(corrective_meta, dict):
            ordered_keys.extend(corrective_meta.keys())
    return ordered_keys


def extract_overridden_keys(input_message: str) -> Set[str]:
    # Public helper used in tests and debugging when we only need the unique
    # overridden field names.
    return set(_extract_overridden_key_sequence(input_message))


def build_label_frames(
    corrected: pd.DataFrame,
    non_corrected: pd.DataFrame,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # We train one binary model per frequently corrected field. For example,
    # "label_notional = 1" means this trade's notional was overridden in a
    # MISS_REPORTING correction and should be treated as a positive example.
    corrected = corrected.copy()
    corrected["overridden_key_sequence"] = corrected["input_message"].apply(_extract_overridden_key_sequence)
    corrected["overridden_keys"] = corrected["overridden_key_sequence"].apply(set)

    key_counts: Counter[str] = Counter()
    first_seen_order = {}
    for sequence in corrected["overridden_key_sequence"]:
        for key in sequence:
            if key not in first_seen_order:
                first_seen_order[key] = len(first_seen_order)
        key_counts.update(sequence)

    # When two fields have the same frequency, keeping first-seen order makes
    # target selection deterministic. That keeps tests stable and results easier
    # to reason about.
    target_fields = [
        field
        for field, _ in sorted(
            key_counts.items(),
            key=lambda item: (-item[1], first_seen_order[item[0]]),
        )[:top_n]
    ]

    labels_corrected = pd.DataFrame({"trade_id": corrected["trade_id"]})
    for field in target_fields:
        labels_corrected[f"label_{field}"] = corrected["overridden_keys"].apply(
            lambda keys: 1 if field in keys else 0
        )

    # Trades with no corrective report are negative samples for every target
    # field in this POC.
    labels_non_corrected = pd.DataFrame({"trade_id": non_corrected["trade_id"]})
    for field in target_fields:
        labels_non_corrected[f"label_{field}"] = 0

    return labels_corrected, labels_non_corrected, target_fields
