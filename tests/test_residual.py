"""Tests for the shared physics-plus-learned-residual model."""

import numpy as np
import pandas as pd
import pytest

from src.models.residual import ResidualModel

SPEC = {"features": ["x"], "lightgbm": {"n_estimators": 100, "min_child_samples": 5}}


class ConstantBase:
    """A fitted base model predicting `value`, NaN where x < 0."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.where(frame["x"] < 0, np.nan, self.value)


def frame(n: int = 200) -> pd.DataFrame:
    x = np.linspace(-1, 1, n)
    return pd.DataFrame({"x": x, "y": 10.0 + 2.0 * (x > 0.5), "keep": x < 0.5})


def test_residual_model_skips_rows_where_the_base_is_undefined() -> None:
    model = ResidualModel(ConstantBase(10.0), SPEC, seed=0, target="y")
    model.fit({"T01": frame()})

    assert np.isnan(model.predict(pd.DataFrame({"x": [-0.5]}))).all()


def test_residual_model_learns_the_residual() -> None:
    model = ResidualModel(ConstantBase(10.0), SPEC, seed=0, target="y")
    model.fit({"T01": frame()})

    assert model.predict(pd.DataFrame({"x": [0.9]})) == pytest.approx([12.0], abs=0.1)


class ColumnBase:
    """A fitted base model predicting column b."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["b"].to_numpy()


def test_residual_model_can_use_the_base_prediction_as_a_feature() -> None:
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"x": 0.0, "b": rng.uniform(0, 10, 500)})
    data["y"] = 2 * data["b"]  # residual = b: only visible through the base
    model = ResidualModel(
        ColumnBase(), SPEC, seed=0, target="y", base_as_feature=True
    ).fit({"T01": data})

    query = pd.DataFrame({"x": [0.0], "b": [5.0]})

    assert model.predict(query) == pytest.approx([10.0], abs=0.3)


def test_residual_model_fits_only_the_selected_rows() -> None:
    model = ResidualModel(
        ConstantBase(10.0), SPEC, seed=0, target="y", rows=lambda f: f["keep"]
    )
    model.fit({"T01": frame()})

    # rows with x > 0.5 (residual 2) were left out, so nothing was learned
    assert model.predict(pd.DataFrame({"x": [0.9]})) == pytest.approx([10.0], abs=0.1)
