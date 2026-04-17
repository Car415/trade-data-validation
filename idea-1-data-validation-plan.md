# Idea 1: AI-Powered Trade Data Validation (MISS_REPORTING Model)

## POC Implementation Plan (Python + LightGBM)

This plan focuses on building the **MISS_REPORTING model** -- a proof of concept that predicts which submission fields are likely reported incorrectly, before the regulator catches it.

---

## 1. Your Data Pipeline & Where Errors Happen

### 1.1 The Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│   ELIGIBILITY   │────>│   SUBMISSION    │     │ CORRECTIVE REPORTING │
│                 │     │                 │     │  (separate from      │
│ Determines if a │     │ When eligible,  │     │   the pipeline)      │
│ trade must be   │     │ sends required  │     │                      │
│ reported to     │     │ data to the     │     │ Manual remediation   │
│ the regulator   │     │ regulator       │     │ when business spots  │
│                 │     │                 │     │ data quality issues  │
│ terms: {        │     │ terms: {        │     │                      │
│   flat JSON     │     │   flat JSON     │     │ corrective_report_   │
│   with elig     │     │   with SOME     │     │   type: enum         │
│   keys          │     │   elig keys +   │     │ input_message: {     │
│ }               │     │   submission    │     │   flat JSON with     │
│                 │     │   keys          │     │   override values    │
│                 │     │ }               │     │ }                    │
└─────────────────┘     └─────────────────┘     └──────────────────────┘
```

### 1.2 The Terms Columns

Both eligibility and submission have a `terms` column containing a **flat JSON**. The keys vary by asset class and product type:

```
Eligibility terms (Interest Rate Swap):     Eligibility terms (Credit Default Swap):
{                                           {
  "asset_class": "IR",                        "asset_class": "CR",
  "product_type": "SWAP",                     "product_type": "CDS",
  "notional": "1000000",                      "notional": "500000",
  "currency": "USD",                          "currency": "EUR",
  "fixed_rate": "0.035",                      "reference_entity": "ABC Corp",
  "float_index": "SOFR",                      "seniority": "SENIOR",
  "payment_freq": "3M",                       "restructuring": "MOD_MOD_R",
  "maturity_date": "2030-01-15",              "maturity_date": "2028-06-20",
  "counterparty_lei": "529900ABC...",          "counterparty_lei": "529900XYZ...",
  "source_system": "MUREX"                     "source_system": "CALYPSO"
}                                           }

Submission terms inherit SOME eligibility keys + add submission-specific keys.
Different asset classes/product types have different keys.
All values are stored as strings, even numbers and dates.
```

### 1.3 The 6 Corrective Reporting Types

| Type | Meaning | Action | Modelable? |
|------|---------|--------|------------|
| **MISS_REPORTING** | Trade data reported incorrectly | Override specific fields via `input_message` | **High -- POC focus** |
| **RE_EVALUATION** | Source data changed, rerun needed | Reprocess with new source | Low (external event) |
| **UNDER_REPORTING** | Trade is eligible but deemed ineligible | Submit the missed trade | Medium (no submission terms) |
| **OVER_REPORTING_TERM** | Trade is over-reported | Terminate the trade | Medium (eligibility error) |
| **OVER_REPORTING_ERROR** | Trade is over-reported | Cancel the trade | Medium (eligibility error) |
| **FORCE_INELIGIBLE** | Trade is ineligible but deemed eligible | Force to ineligible | Medium (eligibility error) |

### 1.4 Model Strategy (Phased)

```
POC (this plan):    Model A -- MISS_REPORTING
                    "Predict which submission fields are wrong"
                    Uses: eligibility terms + submission terms → predict bad fields

Future Phase 2:     Model B -- Eligibility Error Detection
                    FORCE_INELIGIBLE + OVER_REPORTING_TERM + OVER_REPORTING_ERROR
                    "Predict if eligibility decision is wrong"
                    Uses: eligibility terms only

Future Phase 3:     Model C -- Under-Reporting Detection
                    UNDER_REPORTING
                    "Predict if a trade was incorrectly excluded"
                    Uses: eligibility terms only (no submission data exists)
```

---

## 2. The POC Hypothesis

> "By training on MISS_REPORTING corrective data, a LightGBM model can look at
> new submissions and predict which fields are likely reported incorrectly --
> before the regulator catches it."

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│  TRAINING DATA                   │     │  TEST DATA                       │
│                                  │     │                                  │
│  Corrected trades (older):       │     │  Recent submissions that were    │
│                                  │     │  later corrected:                │
│  Eligibility terms  ─┐          │     │                                  │
│  Submission terms   ─┼─> features│     │  Can the model predict WHICH     │
│  input_message keys ─┘─> labels  │     │  fields will need correction?    │
│                                  │     │                                  │
│  + Non-corrected trades          │     │                                  │
│    (negative samples)            │     │                                  │
└──────────────────────────────────┘     └──────────────────────────────────┘
```

---

## 3. What You Need Before Starting

### 3.1 Data Extracts

**Extract 1: MISS_REPORTING corrective records with eligibility and submission terms**

```sql
SELECT
    c.trade_id,
    c.corrective_report_type,
    c.input_message,               -- JSON: the fields that were overridden
    e.terms   AS elig_terms,       -- JSON: eligibility snapshot
    s.terms   AS sub_terms         -- JSON: submission snapshot (the "bad" data)
FROM corrective_reporting c
JOIN eligibility e ON e.trade_id = c.trade_id
JOIN submission s  ON s.trade_id = c.trade_id
WHERE c.corrective_report_type = 'MISS_REPORTING';
```

**Extract 2: Non-corrected trades (negative samples)**

```sql
SELECT
    s.trade_id,
    NULL AS corrective_report_type,
    NULL AS input_message,
    e.terms   AS elig_terms,
    s.terms   AS sub_terms
FROM submission s
JOIN eligibility e ON e.trade_id = s.trade_id
LEFT JOIN corrective_reporting c
    ON c.trade_id = s.trade_id
    AND c.corrective_report_type = 'MISS_REPORTING'
WHERE c.trade_id IS NULL;
```

**Data volume targets:**

| Source | Ideal Volume | Minimum for POC |
|--------|-------------|-----------------|
| MISS_REPORTING records | 5,000+ | 1,000+ |
| Non-corrected trades | 50,000+ | 10,000+ |
| Time window | 2+ years | 1 year |

### 3.2 Python Environment

```bash
mkdir trade-validation-poc
cd trade-validation-poc
python -m venv venv
# venv\Scripts\activate   (Windows)
# source venv/bin/activate (Linux/Mac)

pip install pandas lightgbm scikit-learn matplotlib seaborn jupyter openpyxl
```

### 3.3 Anonymization

Work with your DB team to:
- Replace counterparty names/LEIs with anonymized IDs (CPTY_001, CPTY_002, ...)
- Remove trader names or other PII
- Keep trade attributes (asset class, product type, source system) intact

---

## 4. Step-by-Step POC Guide

### Step 1: Load the Raw Data

```python
import pandas as pd
import json

# Load the two extracts
corrected = pd.read_csv("data/miss_reporting_trades.csv")
non_corrected = pd.read_csv("data/non_corrected_trades.csv")

print(f"Corrected trades (MISS_REPORTING): {len(corrected)}")
print(f"Non-corrected trades:              {len(non_corrected)}")

# Quick look at what the JSON looks like
print("\nSample elig_terms:")
print(corrected['elig_terms'].iloc[0])

print("\nSample sub_terms:")
print(corrected['sub_terms'].iloc[0])

print("\nSample input_message (the correction):")
print(corrected['input_message'].iloc[0])
```

---

### Step 2: Convert Flat JSON Into Features (Detailed Tutorial)

This is the most important preprocessing step. Your `terms` columns are JSON strings that need to become a table of columns the model can learn from.

#### 2.1 Understanding What We're Doing

```
BEFORE (raw data -- one row per trade):

trade_id | elig_terms                                    | sub_terms
---------|-----------------------------------------------|------------------------------------------
T001     | {"asset_class":"IR","notional":"1000000",...} | {"notional":"1000000","fixed_rate":"0.035",...}
T002     | {"asset_class":"CR","notional":"500000",...}  | {"notional":"500000","ref_entity":"ABC",...}


AFTER (flattened -- one column per JSON key):

trade_id | elig_asset_class | elig_notional | elig_fixed_rate | elig_ref_entity | sub_notional | sub_fixed_rate | sub_ref_entity
---------|------------------|---------------|-----------------|-----------------|--------------|----------------|---------------
T001     | IR               | 1000000.0     | 0.035           | NaN             | 1000000.0    | 0.035          | NaN
T002     | CR               | 500000.0      | NaN             | ABC             | 500000.0     | NaN            | ABC

Keys that don't exist for a trade type become NaN. This is fine -- LightGBM handles NaN natively.
```

#### 2.2 Parse the JSON Strings

```python
def parse_json_column(series):
    """
    Parse a pandas Series of JSON strings into a list of dicts.
    Handles nulls, empty strings, and malformed JSON gracefully.
    """
    parsed = []
    for val in series:
        if pd.isna(val) or val == '':
            parsed.append({})
        else:
            try:
                parsed.append(json.loads(val))
            except json.JSONDecodeError:
                parsed.append({})
    return parsed

# Parse the FEATURE columns (elig_terms and sub_terms are flat JSON)
elig_dicts = parse_json_column(corrected['elig_terms'])
sub_dicts = parse_json_column(corrected['sub_terms'])

# NOTE: input_message is NOT parsed here. It is a nested JSON used
# only as a label source (Step 3). It is NOT a feature for the model.

# Do the same for non-corrected trades
elig_dicts_nc = parse_json_column(non_corrected['elig_terms'])
sub_dicts_nc = parse_json_column(non_corrected['sub_terms'])
```

#### 2.3 Flatten Into DataFrames With Prefixes

```python
# pd.json_normalize turns a list of dicts into a DataFrame
# Each unique key becomes a column. Missing keys become NaN.

elig_features = pd.json_normalize(elig_dicts)
sub_features = pd.json_normalize(sub_dicts)

# Add prefixes so you know where each feature came from
# (eligibility and submission may share key names like "notional")
elig_features = elig_features.add_prefix('elig_')
sub_features = sub_features.add_prefix('sub_')

print(f"Eligibility features: {elig_features.shape[1]} columns")
print(f"Submission features:  {sub_features.shape[1]} columns")
print(f"\nEligibility columns: {elig_features.columns.tolist()}")
print(f"\nSubmission columns:  {sub_features.columns.tolist()}")
```

**What this looks like in practice:**

```
>>> elig_features.head(2)

   elig_asset_class  elig_notional  elig_currency  elig_fixed_rate  elig_float_index  elig_ref_entity ...
0  IR                1000000        USD             0.035            SOFR              NaN
1  CR                500000         EUR             NaN              NaN               ABC Corp

>>> sub_features.head(2)

   sub_notional  sub_currency  sub_fixed_rate  sub_ref_entity  sub_action_type ...
0  1000000       USD            0.035           NaN             NEWT
1  500000        EUR            NaN             ABC Corp        NEWT
```

#### 2.4 Handle Data Types (All Values Are Strings)

Since all JSON values are stored as strings, you need to convert numeric and date fields. There are two approaches:

**Approach A: Auto-detect (quick and dirty, good for exploration)**

```python
def auto_convert_types(df):
    """
    Try to convert string columns to numeric where possible.
    If 80%+ of values parse as numbers, convert the whole column.
    """
    for col in df.columns:
        if df[col].dtype == object:  # string columns only
            numeric_attempt = pd.to_numeric(df[col], errors='coerce')
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            numeric_ratio = numeric_attempt.notna().sum() / len(non_null)
            if numeric_ratio >= 0.8:
                df[col] = numeric_attempt
    return df

elig_features = auto_convert_types(elig_features)
sub_features = auto_convert_types(sub_features)

print("\nAfter auto-conversion:")
print(elig_features.dtypes)
```

**Approach B: Explicit mapping (better for production, use your domain knowledge)**

```python
# You know your data -- list the fields that should be numeric
NUMERIC_KEYS = [
    'notional', 'price', 'quantity', 'fixed_rate', 'spread',
    'strike_price', 'exchange_rate',
    # add your actual numeric keys here
]

DATE_KEYS = [
    'maturity_date', 'trade_date', 'effective_date', 'expiry_date',
    'settlement_date',
    # add your actual date keys here
]

def convert_known_types(df, numeric_keys, date_keys):
    """Convert columns based on known data types."""
    for col in df.columns:
        # Strip the prefix (elig_ or sub_) to match against known keys
        base_key = col.split('_', 1)[1] if '_' in col else col

        if base_key in numeric_keys:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        elif base_key in date_keys:
            dt = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = dt.dt.year
            df[f'{col}_month'] = dt.dt.month
            df[f'{col}_dayofweek'] = dt.dt.dayofweek
            df.drop(columns=[col], inplace=True)

    return df

elig_features = convert_known_types(elig_features, NUMERIC_KEYS, DATE_KEYS)
sub_features = convert_known_types(sub_features, NUMERIC_KEYS, DATE_KEYS)
```

**Why dates get split into year/month/dayofweek:**
```
"2030-01-15" as a string → model can't use it
"2030-01-15" as a datetime → model still can't use it directly

But splitting into:
  year=2030, month=1, dayofweek=2 (Wednesday)
  → now the model can learn "trades maturing in January have more errors"
  → or "year-end trades (month=12) tend to have data issues"
```

#### 2.5 Combine Features Into One Table

```python
# Reset indexes to align rows correctly
elig_features.index = range(len(elig_features))
sub_features.index = range(len(sub_features))

# Combine: each row is one trade with all its features
features_corrected = pd.concat([
    corrected[['trade_id']].reset_index(drop=True),
    elig_features,
    sub_features,
], axis=1)

print(f"Combined feature matrix: {features_corrected.shape}")
print(f"  {features_corrected.shape[0]} trades")
print(f"  {features_corrected.shape[1]} columns (including trade_id)")
```

#### 2.6 Do the Same for Non-Corrected Trades

```python
elig_features_nc = pd.json_normalize(elig_dicts_nc).add_prefix('elig_')
sub_features_nc = pd.json_normalize(sub_dicts_nc).add_prefix('sub_')

# Apply the same type conversions
elig_features_nc = convert_known_types(elig_features_nc, NUMERIC_KEYS, DATE_KEYS)
sub_features_nc = convert_known_types(sub_features_nc, NUMERIC_KEYS, DATE_KEYS)

features_non_corrected = pd.concat([
    non_corrected[['trade_id']].reset_index(drop=True),
    elig_features_nc,
    sub_features_nc,
], axis=1)
```

#### 2.7 Align Columns Between Corrected and Non-Corrected

Different trades may produce different JSON keys. We need both DataFrames to have the same columns:

```python
# Get the union of all columns
all_columns = list(set(features_corrected.columns) | set(features_non_corrected.columns))

# Reindex both DataFrames -- missing columns get NaN (which LightGBM handles)
features_corrected = features_corrected.reindex(columns=all_columns)
features_non_corrected = features_non_corrected.reindex(columns=all_columns)

print(f"Aligned to {len(all_columns)} columns")
print(f"Corrected trades:     {len(features_corrected)}")
print(f"Non-corrected trades: {len(features_non_corrected)}")
```

#### 2.8 Summary: The Full Pipeline

```
Raw JSON strings
    │
    ▼
parse_json_column()          Parse strings into Python dicts
    │
    ▼
pd.json_normalize()          Flatten dicts into DataFrame columns
    │
    ▼
.add_prefix('elig_'/'sub_') Distinguish same-name keys
    │
    ▼
convert_known_types()        String → numeric / date extraction
    │
    ▼
pd.concat([elig, sub])      Combine into one feature row per trade
    │
    ▼
.reindex(columns=all)       Align columns across all trades
    │
    ▼
Ready for model training ✓
```

---

### Step 3: Build Labels From input_message (Nested JSON)

The `input_message` is your **label source** -- it tells you which fields were overridden.
It is **not** a feature. The model never sees it. You only extract the override key names
from it to create binary labels.

**Important:** Unlike `elig_terms` and `sub_terms` (which are flat JSON), `input_message` is
a **nested JSON**. You cannot use `pd.json_normalize` on it directly. Instead, you write a
custom function to navigate to where the override keys live.

#### 3.1 Understand the input_message Structure

```python
# First, inspect a few real input_message values to understand the nesting
for i, msg in enumerate(corrected['input_message'].head(5)):
    print(f"\n--- Record {i} ---")
    try:
        parsed = json.loads(msg)
        print(json.dumps(parsed, indent=2))
    except (json.JSONDecodeError, TypeError):
        print(f"(unparseable: {msg})")
```

The structure looks like:

```json
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
```

The overridden fields live inside `TradeCorrectiveInfo[*].CorrectiveMeta`.
The **keys** of `CorrectiveMeta` are the fields that were wrong.

#### 3.2 Extract Override Keys From the Nested JSON

```python
def extract_overridden_keys(input_message_str):
    """
    Navigate the nested input_message JSON and extract the field names
    that were overridden (the keys inside CorrectiveMeta).

    Returns a set of key names, e.g. {'counterparty_lei', 'notional'}.

    IMPORTANT: Adjust the path (TradeCorrectiveInfo / CorrectiveMeta)
    to match your actual JSON structure. Inspect sample records first.
    """
    if pd.isna(input_message_str) or input_message_str == '':
        return set()

    try:
        msg = json.loads(input_message_str)
    except (json.JSONDecodeError, TypeError):
        return set()

    overridden_keys = set()

    corrective_info_list = msg.get('TradeCorrectiveInfo', [])

    for item in corrective_info_list:
        if isinstance(item, dict):
            meta = item.get('CorrectiveMeta', {})
            if isinstance(meta, dict):
                overridden_keys.update(meta.keys())

    return overridden_keys

# Test on a sample
sample = '{"TradeCorrectiveInfo": [{"tradeID": "abc123", "CorrectiveMeta": {"counterparty_lei": "529900CORRECTED", "notional": "1500000"}}]}'
print(extract_overridden_keys(sample))
# Output: {'counterparty_lei', 'notional'}
```

#### 3.3 Apply to All Corrective Records

```python
# Extract overridden keys for every MISS_REPORTING record
corrected['overridden_keys'] = corrected['input_message'].apply(
    extract_overridden_keys
)

# Sanity check: how many records had parseable overrides?
has_keys = corrected['overridden_keys'].apply(len) > 0
print(f"Records with parseable overrides: {has_keys.sum()} / {len(corrected)}")
print(f"Records with no overrides found:  {(~has_keys).sum()}")

# If many records have no overrides, inspect them -- the JSON path may vary
if (~has_keys).sum() > 0:
    print("\nSample records with no overrides found (check JSON path):")
    for msg in corrected.loc[~has_keys, 'input_message'].head(3):
        print(json.dumps(json.loads(msg), indent=2))
```

#### 3.4 Find the Most Commonly Overridden Fields

```python
from collections import Counter

all_keys = Counter()
for key_set in corrected['overridden_keys']:
    all_keys.update(key_set)

print("Most commonly overridden fields in MISS_REPORTING:")
for key, count in all_keys.most_common(20):
    print(f"  {key}: {count} times")
```

#### 3.5 Build Binary Labels

```python
# Pick the top N fields to build models for
TOP_N = 5  # start small
target_fields = [k for k, _ in all_keys.most_common(TOP_N)]
print(f"\nTarget fields for POC: {target_fields}")

# Create binary labels: for each target field, was it overridden?
# 1 = this field was wrong (overridden in input_message)
# 0 = this field was fine (not overridden, or trade was never corrected)

labels_corrected = pd.DataFrame({'trade_id': corrected['trade_id']})
for field in target_fields:
    labels_corrected[f'label_{field}'] = corrected['overridden_keys'].apply(
        lambda keys: 1 if field in keys else 0
    )

print("\nLabel distribution for corrected trades:")
for field in target_fields:
    col = f'label_{field}'
    print(f"  {field}: {labels_corrected[col].sum()} overridden "
          f"out of {len(labels_corrected)} corrected trades "
          f"({labels_corrected[col].mean():.1%})")

# Non-corrected trades: all labels are 0 (nothing was overridden)
labels_non_corrected = pd.DataFrame({'trade_id': non_corrected['trade_id']})
for field in target_fields:
    labels_non_corrected[f'label_{field}'] = 0
```

#### 3.6 Combine Features + Labels Into Final Dataset

```python
# Combine features
all_features = pd.concat([features_corrected, features_non_corrected],
                         ignore_index=True)
all_labels = pd.concat([labels_corrected, labels_non_corrected],
                       ignore_index=True)

# Merge on trade_id
full_dataset = all_features.merge(all_labels, on='trade_id')

print(f"\nFull dataset: {full_dataset.shape}")
print(f"Corrected trades:     {len(features_corrected)}")
print(f"Non-corrected trades: {len(features_non_corrected)}")
```

#### 3.7 Summary: What Each Column Became

```
CSV columns:                    Role:                      How it's used:
─────────────────────────────   ─────────────────────────  ──────────────────────────────
elig_terms (flat JSON)       →  FEATURES                   json_normalize → elig_* columns
sub_terms (flat JSON)        →  FEATURES                   json_normalize → sub_* columns
corrective_report_type       →  FILTER                     Keep only MISS_REPORTING rows
input_message (nested JSON)  →  LABEL SOURCE               extract_overridden_keys() →
                                NOT a feature.              binary label_* columns
                                Model never sees this.
```

---

### Step 4: Explore the Data

Before training, understand what you're working with.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Which fields get overridden most?
override_counts = pd.Series(key_counts).sort_values(ascending=True).tail(20)
override_counts.plot(kind='barh', figsize=(10, 8), color='steelblue')
plt.title("Top 20 Most Frequently Overridden Fields (MISS_REPORTING)")
plt.xlabel("Number of Overrides")
plt.tight_layout()
plt.savefig("output/top_overridden_fields.png", dpi=150)
plt.show()
```

```python
# Are corrections concentrated by asset class?
if 'elig_asset_class' in full_dataset.columns:
    asset_correction_rate = (
        full_dataset.groupby('elig_asset_class')
        .agg(
            total=('trade_id', 'count'),
            corrected=(f'label_{target_fields[0]}', 'sum')
        )
    )
    asset_correction_rate['rate'] = (
        asset_correction_rate['corrected'] / asset_correction_rate['total']
    )
    print("Correction rate by asset class:")
    print(asset_correction_rate.sort_values('rate', ascending=False))
```

```python
# Are corrections concentrated by source system?
if 'elig_source_system' in full_dataset.columns:
    source_correction_rate = (
        full_dataset.groupby('elig_source_system')
        .agg(
            total=('trade_id', 'count'),
            corrected=(f'label_{target_fields[0]}', 'sum')
        )
    )
    source_correction_rate['rate'] = (
        source_correction_rate['corrected'] / source_correction_rate['total']
    )
    print("Correction rate by source system:")
    print(source_correction_rate.sort_values('rate', ascending=False))
```

```python
# How sparse is the feature matrix?
total_cells = full_dataset.shape[0] * full_dataset.shape[1]
null_cells = full_dataset.isna().sum().sum()
print(f"\nFeature matrix sparsity: {null_cells/total_cells:.1%} null")
print(f"(This is expected -- different asset classes have different keys)")

# Columns with most nulls
null_pct = (full_dataset.isna().sum() / len(full_dataset)).sort_values(ascending=False)
print("\nTop 10 most sparse columns:")
print(null_pct.head(10))
```

---

### Step 5: Train the LightGBM Model

**Why LightGBM?**
- Handles categorical features natively (no encoding needed for string columns)
- Handles missing values (NaN) natively (critical for your variable JSON keys)
- Built-in class imbalance support (most trades are NOT corrected)
- Feature importance tells stakeholders WHY a trade was flagged
- Fast training and proven best-in-class for tabular data

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, confusion_matrix)

# Identify feature columns (everything except trade_id and labels)
label_cols = [c for c in full_dataset.columns if c.startswith('label_')]
feature_cols = [c for c in full_dataset.columns
                if c != 'trade_id' and c not in label_cols]

# Identify categorical columns for LightGBM
cat_cols = full_dataset[feature_cols].select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to pandas Categorical type (LightGBM requirement)
for col in cat_cols:
    full_dataset[col] = full_dataset[col].astype('category')

print(f"Feature columns: {len(feature_cols)}")
print(f"  Numeric:     {len(feature_cols) - len(cat_cols)}")
print(f"  Categorical: {len(cat_cols)}")
```

```python
def train_field_model(dataset, target_field, feature_cols, cat_cols):
    """
    Train a LightGBM model for one target field.
    Predicts: is this field likely reported incorrectly?
    """
    label_col = f'label_{target_field}'

    # Time-based split: train on older, test on newer
    # If you have a date column, use it. Otherwise use 80/20.
    df = dataset.copy()

    # Option A: split by date (preferred -- simulates real usage)
    # df = df.sort_values('elig_trade_date_year')  # if you have this
    # split_idx = int(len(df) * 0.8)

    # Option B: random split (simpler for POC)
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'='*60}")
    print(f"TARGET FIELD: {target_field}")
    print(f"{'='*60}")
    print(f"Training: {len(X_train)} rows ({y_train.mean():.2%} positive)")
    print(f"Testing:  {len(X_test)} rows ({y_test.mean():.2%} positive)")

    model = lgb.LGBMClassifier(
        objective='binary',
        is_unbalance=True,
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        categorical_feature=cat_cols,
        verbose=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0),  # suppress per-iteration output
        ],
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['OK', 'Needs Correction'],
        zero_division=0,
    ))

    auc = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0
    print(f"ROC AUC: {auc:.3f}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print("\nTop 15 Most Important Features:")
    for feat, imp in importance.head(15).items():
        print(f"  {feat}: {imp}")

    return model, {
        'target_field': target_field,
        'auc': auc,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'positive_rate': y_test.mean(),
        'best_iteration': model.best_iteration_,
    }
```

```python
# Train one model per target field
all_models = {}
all_results = []

for field in target_fields:
    model, metrics = train_field_model(
        full_dataset, field, feature_cols, cat_cols
    )
    all_models[field] = model
    all_results.append(metrics)

# Summary
results_df = pd.DataFrame(all_results)
print("\n" + "=" * 60)
print("POC RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))
```

---

### Step 6: Simulate Prediction on Real Data

Show what would have happened if the model had reviewed submissions before they went out.

```python
def simulate_catch_rate(model, dataset, target_field, feature_cols,
                        threshold=0.5):
    """
    For trades that were actually corrected on this field,
    how many would the model have caught?
    """
    label_col = f'label_{target_field}'
    X = dataset[feature_cols]
    y_actual = dataset[label_col]

    y_prob = model.predict_proba(X)[:, 1]

    # Flag trades above threshold
    flagged = y_prob >= threshold
    actually_wrong = y_actual == 1

    caught = (flagged & actually_wrong).sum()
    missed = (~flagged & actually_wrong).sum()
    false_alarm = (flagged & ~actually_wrong).sum()
    total_wrong = actually_wrong.sum()
    total_flagged = flagged.sum()

    print(f"\n{'='*60}")
    print(f"SIMULATION: {target_field} (threshold={threshold})")
    print(f"{'='*60}")
    print(f"Total trades reviewed:        {len(dataset)}")
    print(f"Model flagged:                {total_flagged}")
    print(f"Actually needed correction:   {total_wrong}")
    print(f"")
    print(f"Correctly caught:             {caught} / {total_wrong} "
          f"({caught/max(total_wrong,1):.0%})")
    print(f"Missed:                       {missed}")
    print(f"False alarms:                 {false_alarm}")
    if total_flagged > 0:
        print(f"Precision:                    {caught/total_flagged:.0%}")
    if total_wrong > 0:
        print(f"Recall:                       {caught/total_wrong:.0%}")

    return {
        'field': target_field,
        'threshold': threshold,
        'caught': caught,
        'missed': missed,
        'false_alarm': false_alarm,
        'precision': caught / max(total_flagged, 1),
        'recall': caught / max(total_wrong, 1),
    }

# Run simulation for each field
sim_results = []
for field in target_fields:
    result = simulate_catch_rate(
        all_models[field], full_dataset, field, feature_cols
    )
    sim_results.append(result)

sim_df = pd.DataFrame(sim_results)
print("\n\nSIMULATION SUMMARY:")
print(sim_df.to_string(index=False))
```

```python
# Try different thresholds to find the best trade-off
for field in target_fields[:1]:  # do the top field
    print(f"\nThreshold analysis for: {field}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'Flagged':<12}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        result = simulate_catch_rate(
            all_models[field], full_dataset, field, feature_cols,
            threshold=thresh
        )
        print(f"{thresh:<12.1f} {result['precision']:<12.0%} "
              f"{result['recall']:<12.0%} "
              f"{result['caught'] + result['false_alarm']:<12}")
```

---

### Step 7: Visualize for Stakeholders

```python
def create_poc_report(results_df, sim_df, all_models, feature_cols,
                      target_fields, output_dir="output"):
    """Generate presentation-ready charts."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: AUC per target field
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(x='target_field', y='auc', kind='bar',
                    ax=ax, color='steelblue', legend=False)
    ax.set_title("Model Accuracy by Target Field (ROC AUC)")
    ax.set_ylabel("AUC (1.0 = perfect, 0.5 = random)")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.7, color='orange', linestyle='--', label='Minimum viable')
    ax.axhline(y=0.85, color='green', linestyle='--', label='Good')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_accuracy.png", dpi=150)
    plt.show()

    # Chart 2: Catch rate per field
    fig, ax = plt.subplots(figsize=(10, 6))
    sim_df.plot(x='field', y='recall', kind='bar',
                ax=ax, color='seagreen', legend=False)
    ax.set_title("What % of Corrections Would the Model Have Caught?")
    ax.set_ylabel("Recall (% of actual errors caught)")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(sim_df['recall']):
        ax.text(i, v + 0.02, f"{v:.0%}", ha='center', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/catch_rate.png", dpi=150)
    plt.show()

    # Chart 3: Feature importance for top field
    top_field = target_fields[0]
    model = all_models[top_field]
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values().tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind='barh', ax=ax, color='coral')
    ax.set_title(f"What Drives Errors in '{top_field}'?")
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_{top_field}.png", dpi=150)
    plt.show()

create_poc_report(results_df, sim_df, all_models, feature_cols, target_fields)
```

---

## 5. POC Project Structure

```
trade-validation-poc/
├── data/                              # Data extracts (DO NOT commit)
│   ├── miss_reporting_trades.csv      # Extract 1: corrected trades
│   └── non_corrected_trades.csv       # Extract 2: normal trades
├── notebooks/
│   ├── 01_load_and_explore.ipynb      # Steps 1 + 4: load, parse, EDA
│   ├── 02_json_to_features.ipynb      # Step 2: JSON flattening tutorial
│   ├── 03_build_labels.ipynb          # Step 3: input_message → labels
│   ├── 04_train_lightgbm.ipynb        # Step 5: model training
│   ├── 05_simulate_and_report.ipynb   # Steps 6 + 7: test & charts
├── output/                            # Charts and reports
│   ├── top_overridden_fields.png
│   ├── model_accuracy.png
│   ├── catch_rate.png
│   └── feature_importance_*.png
├── requirements.txt
├── README.md
└── .gitignore                         # Exclude data/ folder
```

---

## 6. Success Criteria

| Metric | Minimum for POC | Stretch Goal |
|--------|----------------|--------------|
| AUC score | > 0.70 | > 0.85 |
| Recall (errors caught) | > 50% | > 70% |
| Precision (flagged & truly wrong) | > 40% | > 65% |
| Fields where model works | >= 3 of top 5 | All top 5 |

---

## 7. Timeline

| Step | Duration | Who | Output |
|------|----------|-----|--------|
| Data extraction (SQL queries) | 2-3 days | 1 Dev + DBA | CSV files |
| Step 1-2: Load + JSON-to-features | 2 days | 1 Dev | Feature matrix |
| Step 3-4: Labels + exploration | 2 days | 1 Dev | Labeled dataset + EDA charts |
| Step 5: Train LightGBM | 2-3 days | 1 Dev | Trained models + metrics |
| Step 6-7: Simulate + report | 1-2 days | 1 Dev + 1 BA | POC report for stakeholders |
| **Total** | **~2 weeks** | **1 Dev + BA support** | **Proof + presentation** |

---

## 8. What to Present to Management

1. **"The model caught X% of MISS_REPORTING corrections before they happened."**
   Even 50% means cutting manual correction effort significantly.

2. **"These are the fields most often reported incorrectly."**
   The override frequency chart is actionable even without the model.

3. **"These source systems and product types produce the most errors."**
   Feature importance shows WHERE data quality problems originate.

4. **"Estimated Y hours/week saved."**
   Multiply corrections caught by average time-per-correction.

5. **"Here's the roadmap."**
   POC → Model B (eligibility errors) → Production service → Dashboard.

---

## 9. Future Roadmap

```
Phase 1 (this POC)     Phase 2               Phase 3              Production
────────────────────   ────────────────────   ──────────────────   ──────────────────
Model A:               Model B:              Model C:             All models as
MISS_REPORTING         Eligibility Errors    UNDER_REPORTING      a service:
                       FORCE_INELIGIBLE      detection
Python notebooks       OVER_REPORTING_TERM                        FastAPI / Java
Offline / batch        OVER_REPORTING_ERROR                       Live DB connection
CSV data                                                          Solace MQ pipeline
Manual run             Python notebooks      Python notebooks     Automated + retrain
                       Offline / batch       Offline / batch      BA dashboard (web)
```

---

## 10. Common Pitfalls

| Pitfall | What Happens | How to Avoid |
|---------|-------------|--------------|
| **Class imbalance** | Model always says "OK" because 95%+ of trades are fine | `is_unbalance=True` in LightGBM (already in code above) |
| **Data leakage** | Training on data that wouldn't exist at prediction time | Split by date: train on older, test on newer |
| **Too few overrides per field** | Not enough corrections for a specific field | Focus on fields with 100+ overrides; group rare fields |
| **JSON key mismatch** | Corrected and non-corrected trades have different columns | Use `reindex(columns=all_columns)` to align (Step 2.7) |
| **All strings** | Numeric features treated as categories | Use auto-detect or explicit mapping (Step 2.4) |
| **Overfitting** | Great on training, bad on test | Early stopping (already in code); check train vs. test gap |
| **Sparse features** | Many NaN from asset-class-specific keys | LightGBM handles NaN natively; consider per-asset-class models if accuracy is low |
