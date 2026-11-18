"""Hybrid models: a fitted base (physics) model plus a learned residual."""

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.modelling import NormalBehaviourModel


class ResidualModel:
    """Base prediction plus a LightGBM model of its residual, per turbine.

    Rows where the base model predicts NaN (outside its domain) are not
    used for fitting and stay NaN.
    """

    def __init__(
        self,
        base: NormalBehaviourModel,
        spec: dict[str, Any],
        seed: int,
        target: str,
    ) -> None:
        self.base = base
        self.target = target
        self.features: list[str] = spec["features"]
        self.regressor = LGBMRegressor(
            **spec["lightgbm"], random_state=seed, verbose=-1
        )

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Fit the residual model; the base model is already fitted."""
        rows = pd.concat(list(train_frames.values()), ignore_index=True)
        residual = rows[self.target] - self.base.predict(rows)
        known = residual.notna()
        self.regressor.fit(rows.loc[known, self.features], residual[known])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Base prediction plus the learned residual for every row."""
        return self.base.predict(frame) + self.regressor.predict(frame[self.features])
