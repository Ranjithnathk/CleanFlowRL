import pandas as pd
from .base_tool import BaseCleaningTool

class CategoryNormalizer(BaseCleaningTool):
    """
    Normalizes categorical columns by:
    - converting to lowercase
    - stripping whitespace
    - replacing placeholder tokens like 'UNK', 'NA' with a unified 'unknown'.
    """

    def __init__(self, cost: float = 0.2):
        super().__init__(name="category_normalizer", cost=cost)
        self.placeholder_tokens = {"unk", "na", "n/a", "none", "null", "xxx"}

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        cat_cols = df_new.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            series = df_new[col].astype(str)

            # normalize casing and whitespace
            series = series.str.lower().str.strip()

            # unify placeholder tokens
            series = series.replace(
                list(self.placeholder_tokens),
                "unknown"
            )

            df_new[col] = series

        return df_new
