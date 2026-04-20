import os
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _clip_probability(value: float) -> float:
    # Prevent exact 0 or 1 before any logit transform. Those values would
    # create infinities.
    return min(max(value, 1e-5), 1 - 1e-5)


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return float(np.log(value / (1 - value)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def binary_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    # We compute AUC ourselves so the project stays explicit about what the
    # metric means and avoids pulling in extra metric helpers for one formula.
    # AUC near 1.0 means the model ranks bad trades above good ones well.
    y_true_values = np.asarray(y_true, dtype=int)
    y_prob_values = np.asarray(y_prob, dtype=float)

    if len(y_true_values) == 0 or len(np.unique(y_true_values)) < 2:
        return 0.5

    ranks = pd.Series(y_prob_values).rank(method="average").to_numpy(dtype=float)
    n_pos = int(y_true_values.sum())
    n_neg = int((1 - y_true_values).sum())
    sum_ranks_pos = float(ranks[y_true_values == 1].sum())
    return (sum_ranks_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)

def _prepare_lightgbm_frames(
    frame: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
) -> pd.DataFrame:
    # LightGBM can work directly with pandas categorical columns. That is very
    # useful here because many business fields are strings such as asset class,
    # source system, or currency.
    prepared = frame[feature_columns].copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].astype("category")
    return prepared


def _build_lightgbm_model(seed: int) -> lgb.LGBMClassifier:
    # These parameters are a practical starting point for a tabular-data POC.
    # They are not heavily tuned yet; the goal is a stable, readable baseline.
    return lgb.LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )


def train_models(
    dataset: pd.DataFrame,
    target_fields: Iterable[str],
    seed: int = 42,
) -> Tuple[Dict[str, lgb.LGBMClassifier], pd.DataFrame, List[str]]:
    # Every `label_*` column is a training target. Everything else except
    # `trade_id` is treated as a feature.
    label_columns = [column for column in dataset.columns if column.startswith("label_")]
    feature_columns = [column for column in dataset.columns if column != "trade_id" and column not in label_columns]
    categorical_columns = dataset[feature_columns].select_dtypes(include=["object"]).columns.tolist()
    models: Dict[str, lgb.LGBMClassifier] = {}
    results = []

    for target_field in target_fields:
        label_column = f"label_{target_field}"
        y = dataset[label_column].astype(int)

        # Stratified splitting helps keep the positive/negative ratio similar in
        # train and test. We fall back to a normal split if there are too few
        # positives to stratify safely.
        stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            dataset[feature_columns],
            y,
            test_size=0.2,
            random_state=seed,
            stratify=stratify,
        )

        X_train_prepared = _prepare_lightgbm_frames(X_train, feature_columns, categorical_columns)
        X_test_prepared = _prepare_lightgbm_frames(X_test, feature_columns, categorical_columns)

        model = _build_lightgbm_model(seed)
        model.fit(
            X_train_prepared,
            y_train,
            eval_set=[(X_test_prepared, y_test)],
            categorical_feature=categorical_columns,
            callbacks=[
                # Early stopping prevents the model from training forever once
                # the validation score stops improving.
                lgb.early_stopping(25, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # `predict_proba` returns probabilities, which are more informative than
        # hard yes/no predictions. We still convert them to 0/1 for precision
        # and recall using a simple 0.5 threshold in this POC.
        test_prob = model.predict_proba(X_test_prepared)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        y_test_values = y_test.to_numpy(dtype=int)
        tp = int(((test_pred == 1) & (y_test_values == 1)).sum())
        fp = int(((test_pred == 1) & (y_test_values == 0)).sum())
        fn = int(((test_pred == 0) & (y_test_values == 1)).sum())
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
                "train_size": len(X_train),
                "test_size": len(X_test),
                # This is helpful when reading results because it tells us how
                # long early stopping actually let the model train.
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
            }
        )

    return models, pd.DataFrame(results), feature_columns


def simulate_catch_rates(
    models: Dict[str, lgb.LGBMClassifier],
    dataset: pd.DataFrame,
    feature_columns: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    # This simulates the business question: "if we ran the model before sending
    # the submission, how many bad trades would it have flagged?"
    categorical_columns = dataset[feature_columns].select_dtypes(include=["object"]).columns.tolist()
    prepared = _prepare_lightgbm_frames(dataset, feature_columns, categorical_columns)
    simulation_rows = []
    for field, model in models.items():
        label_column = f"label_{field}"
        probabilities = model.predict_proba(prepared)[:, 1]
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
