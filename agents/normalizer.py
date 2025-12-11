# agents/normalizer.py
import pandas as pd
import numpy as np
from .base_tool import BaseCleaningTool

class Normalizer(BaseCleaningTool):
    def __init__(self, cost: float = 0.2):
        super().__init__(name="normalize", cost=cost)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_min = df_new[col].min()
            col_max = df_new[col].max()
            if pd.isna(col_min) or pd.isna(col_max):
                continue
            if col_max - col_min == 0:
                continue
            df_new[col] = (df_new[col] - col_min) / (col_max - col_min + 1e-8)

        return df_new
