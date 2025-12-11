# agents/outlier_remover.py
import pandas as pd
import numpy as np
from .base_tool import BaseCleaningTool

class OutlierRemover(BaseCleaningTool):
    def __init__(self, z_thresh: float = 3.0, cost: float = 0.3):
        super().__init__(name="remove_outliers", cost=cost)
        self.z_thresh = z_thresh

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        numeric_df = df_new.select_dtypes(include=[np.number])

        if numeric_df.empty or len(df_new) == 0:
            return df_new

        means = numeric_df.mean()
        stds = numeric_df.std().replace(0, 1e-8)
        z_scores = (numeric_df - means) / stds

        mask_keep = ~(z_scores.abs() > self.z_thresh).any(axis=1)
        df_new = df_new.loc[mask_keep].reset_index(drop=True)
        return df_new
