"""LightGBM-only thermal model: temperature from the current inputs and
their recent history (rolling means), per turbine and target."""

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.thermal.features import normal_target_rows


class LightGBMThermalModel:
    """Fitted on normal rows with a measured target."""

    def __init__(
        self, spec: dict[str, Any], seed: int, target: str, features: list[str]
    ) -> None:
        self.target = target
        self.features = features
        self.rows = normal_target_rows(target)
        self.regressor = LGBMRegressor(
            **spec["lightgbm"], random_state=seed, verbose=-1
        )

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Fit on the normal rows of the training frames."""
        rows = pd.concat(
            [f[self.rows(f)] for f in train_frames.values()], ignore_index=True
        )
        self.regressor.fit(rows[self.features], rows[self.target])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected temperature for every row."""
        return self.regressor.predict(frame[self.features])
