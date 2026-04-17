from typing import Dict, Any

import pandas as pd

from .labels import build_label_frames
from .mock_data import create_mock_datasets
from .preprocessing import align_feature_columns, build_feature_tables
from .training import simulate_catch_rates, train_models


def assemble_dataset(
    corrected: pd.DataFrame,
    non_corrected: pd.DataFrame,
    top_n: int = 5,
) -> Dict[str, Any]:
    corrected_features = build_feature_tables(corrected)
    non_corrected_features = build_feature_tables(non_corrected)
    corrected_features, non_corrected_features = align_feature_columns(
        corrected_features,
        non_corrected_features,
    )

    labels_corrected, labels_non_corrected, target_fields = build_label_frames(
        corrected,
        non_corrected,
        top_n=top_n,
    )

    all_features = pd.concat([corrected_features, non_corrected_features], ignore_index=True)
    all_labels = pd.concat([labels_corrected, labels_non_corrected], ignore_index=True)
    dataset = all_features.merge(all_labels, on="trade_id", how="inner")

    return {
        "dataset": dataset,
        "target_fields": target_fields,
    }


def run_mock_poc(
    corrected_count: int = 48,
    non_corrected_count: int = 120,
    top_n: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    corrected, non_corrected = create_mock_datasets(
        corrected_count=corrected_count,
        non_corrected_count=non_corrected_count,
        seed=seed,
    )
    assembled = assemble_dataset(corrected, non_corrected, top_n=top_n)
    models, results, feature_columns = train_models(
        assembled["dataset"],
        assembled["target_fields"],
        seed=seed,
    )
    simulation = simulate_catch_rates(models, assembled["dataset"], feature_columns)
    return {
        "corrected": corrected,
        "non_corrected": non_corrected,
        "dataset": assembled["dataset"],
        "target_fields": assembled["target_fields"],
        "feature_columns": feature_columns,
        "models": models,
        "results": results,
        "simulation": simulation,
    }
