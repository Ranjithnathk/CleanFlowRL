import pandas as pd
from .base_tool import BaseCleaningTool

class CategoricalImputer(BaseCleaningTool):
    """
    Fills missing values in categorical (object/category/bool) columns
    with the most frequent (mode) value.
    """

    def __init__(self, cost: float = 0.2):
        super().__init__(name="categorical_imputer", cost=cost)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        cat_cols = df_new.select_dtypes(include=["object", "category", "bool"]).columns

        for col in cat_cols:
            if df_new[col].isna().any():
                mode = df_new[col].mode(dropna=True)
                if len(mode) > 0:
                    fill_value = mode.iloc[0]
                    df_new[col] = df_new[col].fillna(fill_value)
        return df_new
