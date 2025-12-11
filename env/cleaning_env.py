import numpy as np
import pandas as pd
from typing import Dict, Tuple

from agents.imputer import MissingValueImputer
from agents.deduplicator import Deduplicator
from agents.outlier_remover import OutlierRemover
from agents.normalizer import Normalizer
from agents.schema_validator import SchemaValidator
from agents.categorical_imputer import CategoricalImputer
from agents.category_normalizer import CategoryNormalizer
from utils.metrics import (
    compute_data_quality,
    missing_fraction,
    duplicate_fraction,
    outlier_fraction,
    normalized_flag,
    categorical_missing_rate,
    categorical_inconsistency_rate,
)

class CleaningEnv:
    """
    Environment for RL-based data cleaning.

    State: vector of dataset quality metrics:
        [missing_frac, dup_frac, outlier_frac, normalized_flag, schema_score, step_ratio]
    Action: index of cleaning tool or finalize.
    """

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.step_count = 0

        self.tools = [
            MissingValueImputer(),
            Deduplicator(),
            OutlierRemover(),
            Normalizer(),
            SchemaValidator(),
            CategoricalImputer(),
            CategoryNormalizer(),
        ]
        self.num_tool_actions = len(self.tools)
        self.finalize_action = self.num_tool_actions  # last index

        self.df: pd.DataFrame | None = None
        self.state: np.ndarray | None = None

        # tracks whether schema validator has been called at least once this episode
        self.schema_validated = False

    @property
    def state_dim(self) -> int:
        return 8

    @property
    def num_actions(self) -> int:
        return self.num_tool_actions + 1  # including finalize

    def reset(self, df: pd.DataFrame) -> np.ndarray:
        """Reset env with a new raw dataset."""
        self.df = df.copy()
        self.step_count = 0
        self.schema_validated = False
        self.state = self._compute_state()
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply an action:
        - if action == finalize_action: episode ends, final reward
        - else: apply corresponding tool, intermediate reward
        """
        assert self.df is not None, "Environment not reset with a dataset."
        done = False
        info: Dict = {}

        prev_quality = compute_data_quality(self.df, schema_validated=self.schema_validated)

        if action == self.finalize_action:
            # final reward based on final quality & slight penalty for longer sequences
            reward = prev_quality - 0.05 * (self.step_count / max(1, self.max_steps))
            done = True
        else:
            tool = self.tools[action]

            # special handling: if schema validator, flip flag
            if isinstance(tool, SchemaValidator):
                self.schema_validated = True

            prev_df = self.df.copy()
            self.df = tool.apply(self.df)

            new_quality = compute_data_quality(self.df, schema_validated=self.schema_validated)

            # Reward = improvement - tool cost
            reward = (new_quality - prev_quality) - tool.cost

        self.step_count += 1

        if self.step_count >= self.max_steps and not done:
            # force finalize with current quality, small penalty
            final_quality = compute_data_quality(self.df, schema_validated=self.schema_validated)
            reward += final_quality - 0.1  # encourage finishing earlier
            done = True

        self.state = self._compute_state()
        return self.state, float(reward), done, info

    def _compute_state(self) -> np.ndarray:
        assert self.df is not None
        miss = missing_fraction(self.df)
        dups = duplicate_fraction(self.df)
        outs = outlier_fraction(self.df)
        norm = normalized_flag(self.df)
        schema_score = 1.0 if self.schema_validated else 0.5
        step_ratio = self.step_count / self.max_steps

        cat_miss = categorical_missing_rate(self.df)
        cat_inconsistency = categorical_inconsistency_rate(self.df)

        state = np.array(
            [miss, dups, outs, norm, schema_score, step_ratio, cat_miss, cat_inconsistency],
            dtype=float,
        )
        return state
