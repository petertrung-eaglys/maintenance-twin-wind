"""Tests for the thermal training helpers."""

import numpy as np
import pandas as pd
import pytest

from src.models.residual import BASE_COLUMN
from src.models.thermal.train import (
    VARIANTS,
    hybrid_model,
    parse_targets,
    prepare,
    scored_rows,
    selected_targets,
    thermal_columns,
    training_period,
    variant_specs,
)

THERMAL = {"history_inputs": ["power"], "history_windows_min": [20]}
SPEC = pd.Series({"rotor_diameter_m": 92.0, "rated_power_kw": 2050.0})


def test_prepare_sorts_and_adds_key_specs_and_history() -> None:
    times = pd.to_datetime(["2022-01-01 00:10", "2022-01-01 00:00"], utc=True)
    frame = pd.DataFrame({"timestamp": times, "power": [300.0, 100.0]})

    out = prepare(frame, "KLM_T01", SPEC, THERMAL)

    assert out[
        ["power", "power_mean_20min", "turbine_key", "rotor_diameter_m"]
    ].values.tolist() == [
        [100.0, 100.0, "KLM_T01", 92.0],
        [300.0, 200.0, "KLM_T01", 92.0],
    ]


def test_scored_rows_need_test_period_normal_target_and_all_predictions() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-12-31", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"],
                utc=True,
            ),
            "is_normal": [True, True, False, True, True],
            "temp": [50.0, 50.0, 50.0, np.nan, 50.0],
        }
    )
    outputs = {
        "physics": np.array([1.0, 1.0, 1.0, 1.0, np.nan]),
        "pinn": np.ones(5),
    }

    rows = scored_rows(frame, "temp", pd.Timestamp("2022-01-01", tz="UTC"), outputs)

    assert rows.tolist() == [False, True, False, False, False]


def test_thermal_columns_list_each_needed_column_once() -> None:
    thermal = {
        "history_inputs": ["power", "nacelle_temp"],
        "reference": "nacelle_temp",
        "targets": {"stator_temp": "generator"},
    }

    assert thermal_columns(thermal) == [
        "timestamp",
        "is_normal",
        "power",
        "nacelle_temp",
        "stator_temp",
    ]


def test_training_period_keeps_rows_up_to_and_including_train_end() -> None:
    times = pd.date_range("2021-12-31 23:40", periods=3, freq="10min", tz="UTC")
    frame = pd.DataFrame({"timestamp": times})

    kept = training_period(frame, pd.Timestamp("2021-12-31 23:50", tz="UTC"))

    assert kept["timestamp"].tolist() == list(times[:2])


TARGETS = {"gear_oil_temp": "gearbox", "front_bearing_temp": "main_bearing"}


def test_selected_targets_default_to_all_configured_targets() -> None:
    assert selected_targets(TARGETS, None) == TARGETS


def test_selected_targets_keep_only_the_requested_ones() -> None:
    assert selected_targets(TARGETS, ["front_bearing_temp"]) == {
        "front_bearing_temp": "main_bearing"
    }


def test_selected_targets_reject_an_unknown_target() -> None:
    with pytest.raises(ValueError, match="stator_tmp"):
        selected_targets(TARGETS, ["stator_tmp"])


def test_parse_targets_reads_the_command_line() -> None:
    assert parse_targets(["--targets", "a", "b"]) == ["a", "b"]


def test_parse_targets_is_none_without_the_option() -> None:
    assert parse_targets([]) is None


def test_variant_specs_give_every_variant_the_target_as_qualifier() -> None:
    models = {f"thermal_{v}": {"variant": v} for v in VARIANTS}

    specs = variant_specs(models, "stator_temp", "generator")

    assert list(specs) == list(VARIANTS)
    assert specs["pinn"] == {
        "variant": "pinn",
        "target": "stator_temp",
        "component": "generator",
        "qualifier": "stator_temp",
    }


def test_hybrid_model_fits_normal_target_rows_with_optional_base_feature() -> None:
    spec = {"lightgbm": {}, "base_as_feature": True}
    frame = pd.DataFrame({"is_normal": [True, False, True], "t": [1.0, 2.0, None]})

    model = hybrid_model(None, spec, 0, "t", ["power_mean_20min"])

    assert model.features == ["power_mean_20min", BASE_COLUMN]
    assert model.rows(frame).tolist() == [True, False, False]
