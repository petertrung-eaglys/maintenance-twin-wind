"""Tests for the fleet-relative and load-filtered residual scores."""

import numpy as np
import pandas as pd

from src.health.scores import fleet_relative, needed_columns, smoothed_residual


def times(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq="10min", tz="UTC")


def test_smoothed_residual_can_require_a_minimum_power() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": times(3),
            "is_normal": True,
            "power": [100.0, 900.0, 1500.0],
            "temp": [60.0, 52.0, 54.0],
            "expected": [50.0, 50.0, 50.0],
        }
    )

    smoothed = smoothed_residual(frame, "temp", "expected", "1h", 1, min_power=500)

    # the low-load row is left out
    assert smoothed.tolist() == [2.0, 3.0]


def test_fleet_relative_removes_what_the_farm_shares() -> None:
    index = times(2)
    shared = np.array([1.0, 3.0])  # e.g. a warm day for the whole farm
    residuals = {
        "T01": pd.Series(shared + [0.0, 5.0], index=index),  # T01 heats up
        "T02": pd.Series(shared, index=index),
        "T03": pd.Series(shared, index=index),
        "T04": pd.Series(shared, index=index),
    }

    relative = fleet_relative(residuals, min_turbines=3)

    assert (relative["T01"].tolist(), relative["T02"].tolist()) == (
        [0.0, 5.0],
        [0.0, 0.0],
    )


def test_fleet_relative_needs_enough_other_turbines() -> None:
    index = times(2)
    residuals = {
        "T01": pd.Series([1.0, 1.0], index=index),
        "T02": pd.Series([1.0, np.nan], index=index),
        "T03": pd.Series([1.0, np.nan], index=index),
    }

    relative = fleet_relative(residuals, min_turbines=2)

    # at the second timestamp T01 has no other turbine to compare with
    assert relative["T01"].index.tolist() == [index[0]]


def test_needed_columns_skip_yaw_and_include_power() -> None:
    models = pd.DataFrame(
        {
            "model_class": ["ThermalModel", "YawModel"],
            "target": ["stator_temp", "power"],
            "variant": ["pinn", "physics"],
        }
    )

    assert needed_columns(models) == [
        "timestamp",
        "is_normal",
        "power",
        "stator_temp",
        "expected_stator_temp_pinn",
    ]
