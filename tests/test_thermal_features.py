"""Tests for the shared thermal model inputs."""

import pandas as pd

from src.models.thermal.features import (
    add_history_features,
    feature_columns,
    normal_target_rows,
    target_spec,
    turbine_key,
)


def test_turbine_key_is_unique_across_farms() -> None:
    assert (turbine_key("KLM", "T01"), turbine_key("PEN", "T01")) == (
        "KLM_T01",
        "PEN_T01",
    )


def test_feature_columns_are_inputs_then_rolling_means() -> None:
    thermal = {"history_inputs": ["power"], "history_windows_min": [60, 360]}

    assert feature_columns(thermal) == [
        "power",
        "power_mean_60min",
        "power_mean_360min",
    ]


def test_history_means_use_a_time_window_across_gaps() -> None:
    times = pd.to_datetime(
        ["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 02:00"], utc=True
    )
    frame = pd.DataFrame({"timestamp": times, "power": [100.0, 200.0, 600.0]})

    out = add_history_features(frame, ["power"], [30])

    # the 02:00 row has no other row within 30 minutes
    assert out["power_mean_30min"].tolist() == [100.0, 150.0, 600.0]


def test_normal_target_rows_need_normal_and_a_measured_target() -> None:
    frame = pd.DataFrame(
        {"is_normal": [True, True, False], "stator_temp": [60.0, None, 61.0]}
    )

    assert normal_target_rows("stator_temp")(frame).tolist() == [True, False, False]


def test_target_spec_sets_target_component_and_qualifier() -> None:
    spec = target_spec({"variant": "physics"}, "gear_oil_temp", "gearbox")

    assert spec == {
        "variant": "physics",
        "target": "gear_oil_temp",
        "component": "gearbox",
        "qualifier": "gear_oil_temp",
    }
