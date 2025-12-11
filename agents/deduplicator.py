# agents/deduplicator.py
import pandas as pd
from .base_tool import BaseCleaningTool

class Deduplicator(BaseCleaningTool):
    def __init__(self, cost: float = 0.2):
        super().__init__(name="deduplicate", cost=cost)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.drop_duplicates().reset_index(drop=True)
        return df_new
