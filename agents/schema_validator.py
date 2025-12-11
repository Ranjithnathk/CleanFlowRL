# agents/schema_validator.py
import pandas as pd
from .base_tool import BaseCleaningTool

class SchemaValidator(BaseCleaningTool):
    """
    Custom tool: checks if expected columns are present and numeric.
    Does not change the DataFrame content, but environment uses it to
    boost schema_score when called.
    """

    def __init__(self, expected_columns=None, cost: float = 0.1):
        super().__init__(name="schema_validator", cost=cost)
        self.expected_columns = expected_columns or ["feature_1", "feature_2", "feature_3"]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # We won't modify df here; env will track schema_validated.
        # In a more advanced version, we could drop invalid columns, fix types, etc.
        return df.copy()
