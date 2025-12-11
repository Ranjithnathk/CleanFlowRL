from abc import ABC, abstractmethod
import pandas as pd

class BaseCleaningTool(ABC):
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply this cleaning operation and return a new DataFrame."""
        pass
