"""Hybrid models: a fitted base (physics) model plus a learned residual."""

from collections.abc import Callable, Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.modelling import NormalBehaviourModel

RowFilter = Callable[[pd.DataFrame], pd.Series]
BASE_COLUMN = "base_prediction"


def all_rows(frame: pd.DataFrame) -> pd.Series:
    """Default row filter: every row."""
    return pd.Series(True, index=frame.index)


class ResidualModel:
    """Base prediction plus a LightGBM model of its residual, per turbine.

    The base model predicts each training frame separately (dynamic models
    need a turbine's contiguous rows). Only rows selected by `rows` with a
    defined base prediction are fitted; where the base predicts NaN (outside
    its domain) the result stays NaN. With `base_as_feature` the residual
    model also sees the base prediction (column `BASE_COLUMN`).
    """

    def __init__(
        self,
        base: NormalBehaviourModel,
        spec: dict[str, Any],
        seed: int,
        target: str,
        rows: RowFilter = all_rows,
        base_as_feature: bool = False,
    ) -> None:
        self.base = base
        self.target = target
        self.rows = rows
        self.features: list[str] = [
            *spec["features"],
            *([BASE_COLUMN] if base_as_feature else []),
        ]
        self.regressor = LGBMRegressor(
            **spec["lightgbm"], random_state=seed, verbose=-1
        )

    def _with_base(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{BASE_COLUMN: self.base.predict(frame)})

    def _residual_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = self._with_base(frame)
        residual = rows[self.target] - rows[BASE_COLUMN]
        used = residual.notna() & self.rows(frame)
        return rows.loc[used, self.features].assign(residual=residual[used])

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Fit the residual model; the base model is already fitted."""
        rows = pd.concat(
            [self._residual_rows(f) for f in train_frames.values()],
            ignore_index=True,
        )
        self.regressor.fit(rows[self.features], rows["residual"])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Base prediction plus the learned residual for every row."""
        rows = self._with_base(frame)
        learned = self.regressor.predict(rows[self.features])
        return rows[BASE_COLUMN].to_numpy() + learned
