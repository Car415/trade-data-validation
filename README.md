# Trade Validation POC

This project is a proof of concept for the `MISS_REPORTING` idea described in `idea-1-data-validation-plan.md`.

The goal is to train a model that looks at:
- eligibility terms
- submission terms
- past corrective-report behavior

and predicts which submission fields are likely to be wrong before the regulator catches them.

## What This Project Does

The pipeline follows the same core steps as the plan:

1. Load corrected and non-corrected trade extracts.
2. Parse `elig_terms` and `sub_terms`, which are flat JSON strings.
3. Flatten those JSON fields into model features.
4. Convert important string fields into better ML-friendly types.
5. Parse `input_message`, which is nested JSON, to find which fields were overridden.
6. Build binary labels such as `label_notional` or `label_currency`.
7. Train one LightGBM binary model per target field.
8. Simulate how many bad submissions the model would have caught.

## Project Structure

`src/trade_validation_poc/`

- `data_loading.py`
  Reads the two CSV extracts from the plan.
- `mock_data.py`
  Generates synthetic corrected and non-corrected trades for local testing.
- `preprocessing.py`
  Converts the JSON payloads into a tabular feature matrix.
- `labels.py`
  Extracts overridden fields from `input_message` and turns them into labels.
- `training.py`
  Trains LightGBM models and computes evaluation metrics.
- `pipeline.py`
  Orchestrates the full POC flow.

`tests/`

- Unit and integration-style tests for loading, preprocessing, labeling, training, and pipeline execution.

## Why The Code Looks Like This

If you are new to machine learning, the most important design choice to understand is this:

- machine learning models want a table of rows and columns
- your source data starts as nested or semi-structured JSON

So most of the work in a project like this is not the model itself. Most of the work is:
- converting business data into clean features
- deciding what the target labels mean
- keeping train/test evaluation honest

That is why the code spends a lot of time in preprocessing and label-building.

## Installation

Create a virtual environment if you want:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running The POC With Mock Data

This is the fastest way to see the full pipeline work without any real CSV files.

```python
from trade_validation_poc.pipeline import run_mock_poc

result = run_mock_poc(
    corrected_count=48,
    non_corrected_count=120,
    top_n=3,
    seed=42,
)

print(result["target_fields"])
print(result["results"])
print(result["simulation"])
```

## Running The POC With Real CSV Extracts

Prepare two CSV files that follow the plan:

- corrected trades CSV with columns like:
  - `trade_id`
  - `corrective_report_type`
  - `input_message`
  - `elig_terms`
  - `sub_terms`
- non-corrected trades CSV with columns like:
  - `trade_id`
  - `corrective_report_type`
  - `input_message`
  - `elig_terms`
  - `sub_terms`

Then run:

```python
from trade_validation_poc.pipeline import run_csv_poc

result = run_csv_poc(
    corrected_path="data/miss_reporting_trades.csv",
    non_corrected_path="data/non_corrected_trades.csv",
    top_n=5,
    seed=42,
)

print(result["target_fields"])
print(result["results"])
print(result["simulation"])
```

## Understanding The Main Outputs

`result["target_fields"]`

- The top corrected fields chosen for this POC.
- Example: `["notional", "fixed_rate", "currency"]`

`result["results"]`

- One row per target field model.
- Important columns:
  - `auc`: how well the model ranks bad rows above good rows
  - `precision`: of the rows flagged as bad, how many were actually bad
  - `recall`: of the actually bad rows, how many the model caught
  - `best_iteration`: how many boosting rounds LightGBM used before early stopping

`result["simulation"]`

- A business-style summary using a decision threshold.
- This answers a question like:
  - "If we used the model before submission, how many mistakes would it have flagged?"

## How LightGBM Fits In

LightGBM is the model used in this project because it is a strong choice for tabular data.

It works well here because:
- it handles missing values well
- it can work with categorical columns
- it is strong on structured business data
- it trains fast

The important thing to remember is:
- the model is only as useful as the features and labels we feed into it

That is why the preprocessing and label-building steps are so important.

## Tests

Run the full test suite:

```bash
python -m unittest discover tests -v
```

## Current Scope

This repository now includes the working core POC:
- mock-data mode
- CSV-data mode
- LightGBM training
- feature engineering from JSON payloads
- binary label creation from corrective reports
- basic evaluation and simulation

What is not included yet:
- notebook versions of each step from the plan
- stakeholder chart/report generation
- production deployment pieces such as an API or scheduled retraining
