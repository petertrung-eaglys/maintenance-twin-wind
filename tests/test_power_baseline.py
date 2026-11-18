"""Tests for the LightGBM baseline power model on a synthetic turbine."""

import numpy as np
import pandas as pd
import pytest

from src.models.power_baseline import LightGBMPowerModel, train_turbine

RATED = 2050.0
SPLIT = {"train_end": "2021-12-31 23:50", "calibration_end": "2021-12-31 23:50"}
SEED = 42
SPEC = {
    "model_class": "PowerModel",
    "component": "generator",
    "variant": "baseline",
    "inputs": ["wind_speed", "wind_direction", "ambient_temp"],
    "features": [
        "wind_speed",
        "wind_dir_sin",
        "wind_dir_cos",
        "ambient_temp",
        "air_density",
    ],
    "target": "power",
    "monotone_increasing": ["wind_speed"],
    "lightgbm": {"n_estimators": 100},
}


def synthetic_turbine(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Two years of rows with a logistic power curve and a little noise."""
    rng = np.random.default_rng(seed)
    wind = rng.uniform(3, 20, n)
    temp = rng.normal(10, 5, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", "2022-12-31", periods=n, tz="UTC"),
            "wind_speed": wind,
            "wind_direction": rng.uniform(0, 360, n),
            "ambient_temp": temp,
            "air_density": 101325.0 / (287.05 * (temp + 273.15)),
            "power": RATED / (1 + np.exp(-(wind - 8.0))) + rng.normal(0, 20, n),
            "is_normal": rng.uniform(size=n) > 0.1,
        }
    )


@pytest.fixture(scope="module")
def trained() -> tuple:
    return train_turbine(synthetic_turbine(), "T01", SPEC, SPLIT, RATED, SEED)


def test_train_turbine_counts_normal_rows_per_period(trained: tuple) -> None:
    frame = synthetic_turbine()
    in_train = frame["timestamp"] <= pd.Timestamp("2021-12-31 23:50", tz="UTC")
    _, metrics, _ = trained

    assert (metrics["n_train"], metrics["n_test"]) == (
        int((frame["is_normal"] & in_train).sum()),
        int((frame["is_normal"] & ~in_train).sum()),
    )


def test_train_turbine_learns_the_power_curve(trained: tuple) -> None:
    _, metrics, _ = trained

    assert metrics["nrmse"] < 0.03  # noise alone is 20 kW, about 1 % of rated


def test_train_turbine_predicts_every_row(trained: tuple) -> None:
    _, _, expected = trained

    assert len(expected) == len(synthetic_turbine())


def test_expected_power_never_drops_with_wind_speed(trained: tuple) -> None:
    model, _, _ = trained
    grid = pd.DataFrame(
        {
            "wind_speed": np.linspace(0, 30, 301),
            "wind_dir_sin": 0.3,
            "wind_dir_cos": -0.9,
            "ambient_temp": 5.0,
            "air_density": 1.25,
        }
    )

    assert np.all(np.diff(model.predict(grid)) >= 0)


def test_fit_accepts_several_turbines() -> None:
    frames = {
        tid: synthetic_turbine(500, seed).assign(wind_dir_sin=0.0, wind_dir_cos=1.0)
        for tid, seed in (("T01", 1), ("T02", 2))
    }

    model = LightGBMPowerModel(SPEC, SEED).fit(frames)

    assert len(model.predict(frames["T02"])) == 500
