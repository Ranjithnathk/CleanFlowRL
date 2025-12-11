import numpy as np
import pandas as pd

def generate_synthetic_dataset(
    n_rows: int = 200,
    missing_prob: float = 0.1,
    dup_prob: float = 0.1,
    outlier_prob: float = 0.05,
    random_state: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # --- Numeric core features (multi-modal) ---
    # Mixture of two Gaussians for each feature (to simulate segments)
    mix_component = rng.integers(0, 2, size=n_rows)
    feature_1 = np.where(
        mix_component == 0,
        rng.normal(loc=40, scale=8, size=n_rows),
        rng.normal(loc=70, scale=12, size=n_rows),
    )
    feature_2 = np.where(
        mix_component == 0,
        rng.normal(loc=90, scale=15, size=n_rows),
        rng.normal(loc=120, scale=25, size=n_rows),
    )
    feature_3 = np.where(
        mix_component == 0,
        rng.normal(loc=-2, scale=4, size=n_rows),
        rng.normal(loc=3, scale=6, size=n_rows),
    )

    df = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
    })

    # --- Categorical feature with noise ---
    segments = rng.choice(["A", "B", "C"], size=n_rows)
    # inject label noise / typos ~5%
    noise_mask = rng.random(n_rows) < 0.05
    segments[noise_mask] = rng.choice(["a", "b", "segment_c", "UNK"], size=noise_mask.sum())
    df["segment"] = segments

    # --- Boolean flag (e.g., is_premium) ---
    df["is_premium"] = rng.random(n_rows) < 0.3

    # --- Inject cell-wise missing values in numeric columns ---
    numeric_cols = ["feature_1", "feature_2", "feature_3"]
    mask_missing = rng.random((n_rows, len(numeric_cols))) < missing_prob
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].mask(mask_missing)

    # --- Block missingness: randomly choose one numeric column to partially blank out ---
    block_col = rng.choice(numeric_cols)
    block_mask = rng.random(n_rows) < (missing_prob * 1.5)
    df.loc[block_mask, block_col] = np.nan

    # --- Inject duplicate rows ---
    n_dups = int(dup_prob * n_rows)
    if n_dups > 0:
        dup_indices = rng.integers(low=0, high=len(df), size=n_dups)
        dup_rows = df.iloc[dup_indices]
        df = pd.concat([df, dup_rows], ignore_index=True)

    # --- Inject strong outliers in numeric features ---
    n_outliers = int(outlier_prob * len(df))
    if n_outliers > 0:
        outlier_indices = rng.integers(low=0, high=len(df), size=n_outliers)
        for idx in outlier_indices:
            # extreme spikes on one random feature
            col = rng.choice(numeric_cols)
            df.loc[idx, col] += rng.normal(loc=0, scale=150)

    return df
