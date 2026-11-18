"""Tests for the LightGBM-only thermal model."""

import numpy as np
import pandas as pd
import pytest

from src.models.thermal.baseline import LightGBMThermalModel

SPEC = {"lightgbm": {"n_estimators": 100, "min_child_samples": 5}}


def frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Temperature = 20 + 0.01 * power on normal rows; faulty rows run
    20 degC hotter."""
    rng = np.random.default_rng(seed)
    power = rng.uniform(0, 2000, n)
    normal = rng.random(n) > 0.3
    return pd.DataFrame(
        {
            "power": power,
            "is_normal": normal,
            "temp": 20 + 0.01 * power + np.where(normal, 0, 20),
        }
    )


def test_baseline_learns_temperature_from_the_inputs() -> None:
    model = LightGBMThermalModel(SPEC, seed=0, target="temp", features=["power"])
    model.fit({"KLM_T01": frame()})

    assert model.predict(pd.DataFrame({"power": [1000.0]})) == pytest.approx(
        [30.0], abs=0.5
    )


def test_baseline_ignores_rows_that_are_not_normal() -> None:
    model = LightGBMThermalModel(SPEC, seed=0, target="temp", features=["power"])
    data = frame()
    model.fit({"KLM_T01": data})

    hot = ~data["is_normal"]
    error = model.predict(data[hot]) - data.loc[hot, "temp"]

    assert error.mean() == pytest.approx(-20.0, abs=0.5)
