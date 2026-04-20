import json
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


# These sets let us convert important business fields from raw strings into
# model-friendly values. The source JSON stores everything as strings, which is
# convenient for storage but not ideal for machine learning.
NUMERIC_KEYS = {
    "notional",
    "price",
    "quantity",
    "fixed_rate",
    "spread",
    "strike_price",
    "exchange_rate",
}
DATE_KEYS = {
    "maturity_date",
    "trade_date",
    "effective_date",
    "expiry_date",
    "settlement_date",
}


def parse_json_column(series: pd.Series) -> List[dict]:
    # Real extracts often contain nulls or malformed JSON. Returning an empty
    # dict instead of failing lets the rest of the pipeline continue and keeps
    # missing information as NaN after normalization.
    parsed = []
    for value in series:
        if pd.isna(value) or value == "":
            parsed.append({})
            continue
        try:
            parsed.append(json.loads(value))
        except (TypeError, json.JSONDecodeError):
            parsed.append({})
    return parsed


def convert_known_types(
    frame: pd.DataFrame,
    numeric_keys: Iterable[str] = NUMERIC_KEYS,
    date_keys: Iterable[str] = DATE_KEYS,
) -> pd.DataFrame:
    # LightGBM handles numeric columns, categorical columns, and missing values
    # well, but raw date strings are not directly useful. We split dates into
    # simple calendar features so the model can learn date-related patterns.
    numeric_keys = set(numeric_keys)
    date_keys = set(date_keys)
    result = frame.copy()

    for column in list(result.columns):
        if "_" in column:
            _, base_key = column.split("_", 1)
        else:
            base_key = column

        if base_key in numeric_keys:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif base_key in date_keys:
            dt_value = pd.to_datetime(result[column], errors="coerce")
            result[f"{column}_year"] = dt_value.dt.year
            result[f"{column}_month"] = dt_value.dt.month
            result[f"{column}_dayofweek"] = dt_value.dt.dayofweek
            result = result.drop(columns=[column])

    return result


def build_feature_tables(dataset: pd.DataFrame) -> pd.DataFrame:
    # `elig_terms` and `sub_terms` are flat JSON strings. We flatten each JSON
    # object into columns, prefix them to avoid name collisions, then join them
    # back to the trade id so one row still represents one trade.
    elig_features = pd.json_normalize(parse_json_column(dataset["elig_terms"])).add_prefix("elig_")
    sub_features = pd.json_normalize(parse_json_column(dataset["sub_terms"])).add_prefix("sub_")
    elig_features = convert_known_types(elig_features)
    sub_features = convert_known_types(sub_features)
    return pd.concat(
        [
            dataset[["trade_id"]].reset_index(drop=True),
            elig_features.reset_index(drop=True),
            sub_features.reset_index(drop=True),
        ],
        axis=1,
    )


def align_feature_columns(*feature_frames: Sequence[pd.DataFrame]) -> Tuple[pd.DataFrame, ...]:
    # Different product types create different JSON keys. We align every frame
    # to the union of columns so corrected and non-corrected trades can be
    # stacked into one training dataset.
    all_columns = sorted({column for frame in feature_frames for column in frame.columns})
    return tuple(frame.reindex(columns=all_columns) for frame in feature_frames)
