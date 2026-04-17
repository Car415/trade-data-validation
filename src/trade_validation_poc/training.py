from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _clip_probability(value: float) -> float:
    return min(max(value, 1e-5), 1 - 1e-5)


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return float(np.log(value / (1 - value)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def binary_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    y_true_values = np.asarray(y_true, dtype=int)
    y_prob_values = np.asarray(y_prob, dtype=float)

    if len(y_true_values) == 0 or len(np.unique(y_true_values)) < 2:
        return 0.5

    ranks = pd.Series(y_prob_values).rank(method="average").to_numpy(dtype=float)
    n_pos = int(y_true_values.sum())
    n_neg = int((1 - y_true_values).sum())
    sum_ranks_pos = float(ranks[y_true_values == 1].sum())
    return (sum_ranks_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)


@dataclass
class BaselineFieldModel:
    target_field: str
    categorical_maps: Dict[str, Dict[object, float]] = field(default_factory=dict)
    numeric_rules: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    base_logit: float = 0.0

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "BaselineFieldModel":
        label_mean = float(labels.mean()) if len(labels) else 0.0
        self.base_logit = _logit(label_mean if 0 < label_mean < 1 else 0.5)

        for column in features.columns:
            series = features[column]
            if pd.api.types.is_numeric_dtype(series):
                filled = pd.to_numeric(series, errors="coerce")
                mean = float(filled.mean()) if filled.notna().any() else 0.0
                std = float(filled.std()) if filled.notna().sum() > 1 else 1.0
                std = 1.0 if pd.isna(std) or std == 0 else std
                pos_slice = filled[labels == 1].dropna()
                neg_slice = filled[labels == 0].dropna()
                pos_mean = float(pos_slice.mean()) if not pos_slice.empty else mean
                neg_mean = float(neg_slice.mean()) if not neg_slice.empty else mean
                coef = float(np.clip((pos_mean - neg_mean) / std, -1.0, 1.0))
                self.numeric_rules[column] = (mean, std, coef)
            else:
                grouped = (
                    pd.DataFrame({"value": series.astype("object"), "label": labels})
                    .groupby("value", dropna=False)["label"]
                    .mean()
                )
                self.categorical_maps[column] = {
                    key: float(np.clip(_logit(rate) - self.base_logit, -1.5, 1.5))
                    for key, rate in grouped.items()
                }

        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(features), self.base_logit, dtype=float)
        contribution_count = np.zeros(len(features), dtype=float)

        for column, mapping in self.categorical_maps.items():
            values = features[column].astype("object")
            scores += values.map(lambda value: mapping.get(value, 0.0)).fillna(0.0).to_numpy(dtype=float)
            contribution_count += 1

        for column, (mean, std, coef) in self.numeric_rules.items():
            values = pd.to_numeric(features[column], errors="coerce").fillna(mean)
            scores += ((values.to_numpy(dtype=float) - mean) / std) * coef
            contribution_count += 1

        contribution_count = np.where(contribution_count == 0, 1.0, contribution_count)
        normalized_scores = scores / np.sqrt(contribution_count)
        probabilities = _sigmoid(normalized_scores)
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= threshold).astype(int)

    def feature_importances(self) -> pd.Series:
        importances: Dict[str, float] = {}
        for column, mapping in self.categorical_maps.items():
            importances[column] = max((abs(value) for value in mapping.values()), default=0.0)
        for column, (_, _, coef) in self.numeric_rules.items():
            importances[column] = abs(coef)
        return pd.Series(importances).sort_values(ascending=False)


def train_models(
    dataset: pd.DataFrame,
    target_fields: Iterable[str],
    seed: int = 42,
) -> Tuple[Dict[str, BaselineFieldModel], pd.DataFrame, List[str]]:
    label_columns = [column for column in dataset.columns if column.startswith("label_")]
    feature_columns = [column for column in dataset.columns if column != "trade_id" and column not in label_columns]

    rng = np.random.default_rng(seed)
    shuffled = dataset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_index = max(1, int(len(shuffled) * 0.8))
    train_frame = shuffled.iloc[:split_index]
    test_frame = shuffled.iloc[split_index:]
    if test_frame.empty:
        test_frame = train_frame.copy()

    models: Dict[str, BaselineFieldModel] = {}
    results = []

    for target_field in target_fields:
        label_column = f"label_{target_field}"
        model = BaselineFieldModel(target_field=target_field).fit(
            train_frame[feature_columns],
            train_frame[label_column],
        )
        test_prob = model.predict_proba(test_frame[feature_columns])[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        y_test = test_frame[label_column].astype(int)

        tp = int(((test_pred == 1) & (y_test == 1)).sum())
        fp = int(((test_pred == 1) & (y_test == 0)).sum())
        fn = int(((test_pred == 0) & (y_test == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        models[target_field] = model
        results.append(
            {
                "target_field": target_field,
                "auc": binary_auc(y_test, pd.Series(test_prob)),
                "precision": precision,
                "recall": recall,
                "positive_rate": float(y_test.mean()),
                "train_size": len(train_frame),
                "test_size": len(test_frame),
            }
        )

    return models, pd.DataFrame(results), feature_columns


def simulate_catch_rates(
    models: Dict[str, BaselineFieldModel],
    dataset: pd.DataFrame,
    feature_columns: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    simulation_rows = []
    for field, model in models.items():
        label_column = f"label_{field}"
        probabilities = model.predict_proba(dataset[feature_columns])[:, 1]
        flagged = probabilities >= threshold
        actual = dataset[label_column].astype(int).to_numpy(dtype=int)
        caught = int(np.logical_and(flagged, actual == 1).sum())
        missed = int(np.logical_and(~flagged, actual == 1).sum())
        false_alarm = int(np.logical_and(flagged, actual == 0).sum())
        simulation_rows.append(
            {
                "field": field,
                "threshold": threshold,
                "caught": caught,
                "missed": missed,
                "false_alarm": false_alarm,
                "precision": caught / max(caught + false_alarm, 1),
                "recall": caught / max(caught + missed, 1),
            }
        )
    return pd.DataFrame(simulation_rows)
