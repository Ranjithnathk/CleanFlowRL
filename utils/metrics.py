# utils/metrics.py
import numpy as np
import pandas as pd

def missing_fraction(df: pd.DataFrame) -> float:
    total = df.size
    if total == 0:
        return 0.0
    return float(df.isna().sum().sum()) / float(total)

def duplicate_fraction(df: pd.DataFrame) -> float:
    if len(df) <= 1:
        return 0.0
    num_dups = df.duplicated().sum()
    return float(num_dups) / float(len(df))

def outlier_fraction(df: pd.DataFrame, z_thresh: float = 3.0) -> float:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return 0.0

    means = numeric_df.mean()
    stds = numeric_df.std().replace(0, 1e-8)
    z_scores = (numeric_df - means) / stds
    outlier_mask = (z_scores.abs() > z_thresh).any(axis=1)
    return float(outlier_mask.sum()) / float(len(numeric_df))

def normalized_flag(df: pd.DataFrame, range_thresh: float = 1.1) -> float:
    """
    Heuristic: if all numeric cols roughly in [0, 1] (range <= range_thresh),
    treat as normalized.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return 0.0
    ranges = numeric_df.max() - numeric_df.min()
    if (ranges <= range_thresh).all():
        return 1.0
    return 0.0

def compute_data_quality(
    df: pd.DataFrame,
    schema_validated: bool = False,
    z_thresh: float = 3.0,
) -> float:
    """
    Returns a scalar in [0, 1] summarizing dataset quality.

    Components:
      - 1 - missing_fraction
      - 1 - duplicate_fraction
      - 1 - outlier_fraction
      - schema_score (0.5 if not validated, 1.0 if validated)
    """
    miss = missing_fraction(df)
    dups = duplicate_fraction(df)
    outs = outlier_fraction(df, z_thresh=z_thresh)

    schema_score = 1.0 if schema_validated else 0.5

    miss_score = 1.0 - miss
    dups_score = 1.0 - dups
    outs_score = 1.0 - outs

    quality = 0.25 * miss_score + 0.25 * dups_score + 0.25 * outs_score + 0.25 * schema_score
    # clip for safety
    return float(np.clip(quality, 0.0, 1.0))

def categorical_missing_rate(df: pd.DataFrame) -> float:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) == 0:
        return 0.0
    sub = df[cat_cols]
    total = sub.size
    if total == 0:
        return 0.0
    return float(sub.isna().sum().sum()) / float(total)

def categorical_inconsistency_rate(df: pd.DataFrame, rare_thresh: float = 0.02) -> float:
    """
    Heuristic: for each categorical column, treat very rare values and
    obvious placeholder tokens as 'inconsistent' (e.g., 'UNK', 'NA', 'xxx').
    Returns fraction of inconsistent cells among all categorical cells.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return 0.0

    inconsistent_count = 0
    total = 0

    placeholder_tokens = {"unk", "na", "n/a", "none", "null", "xxx", "segment_c"}

    for col in cat_cols:
        series = df[col].astype(str)
        total += len(series)

        # basic placeholder / weird token detection
        lower_vals = series.str.lower().str.strip()
        placeholder_mask = lower_vals.isin(placeholder_tokens)

        # rare category detection
        value_counts = lower_vals.value_counts(dropna=False)
        freqs = value_counts / len(series)
        rare_values = set(value_counts[freqs < rare_thresh].index)
        rare_mask = lower_vals.isin(rare_values)

        inconsistent_mask = placeholder_mask | rare_mask
        inconsistent_count += inconsistent_mask.sum()

    if total == 0:
        return 0.0

    return float(inconsistent_count) / float(total)