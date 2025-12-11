# agents/imputer.py
import pandas as pd
import numpy as np
from .base_tool import BaseCleaningTool

class MissingValueImputer(BaseCleaningTool):
    def __init__(self, strategy: str = "mean", cost: float = 0.2):
        super().__init__(name="impute_missing", cost=cost)
        self.strategy = strategy

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()

        # Numeric columns (mean/median)
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df_new[col].isna().any():
                if self.strategy == "mean":
                    fill_value = df_new[col].mean()
                elif self.strategy == "median":
                    fill_value = df_new[col].median()
                else:
                    fill_value = df_new[col].mean()
                df_new[col] = df_new[col].fillna(fill_value)

        # Categorical columns --> mode
        cat_cols = df_new.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if df_new[col].isna().any():
                mode_vals = df_new[col].mode(dropna=True)
                if len(mode_vals) > 0:
                    fill_value = mode_vals.iloc[0]
                    df_new[col] = df_new[col].fillna(fill_value)

        return df_new
