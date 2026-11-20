"""Tests for the dashboard's data access."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.dashboard.data import (
    component_scores,
    detail_columns,
    farm_normal,
    read_columns,
    with_model_ids,
)
from src.graph.graph_utils import add_model, get_models

METRICS = {"rmse": 1.0, "mae": 1.0, "nrmse": 1.0, "n_train": 6, "n_test": 6}
CONFIG = {
    "alerts": {
        "window": "1h",
        "min_samples": 1,
        "min_power_kw": {"ThermalModel": 0.0},
        "fleet_relative": False,
        "fleet_min_turbines": 1,
        "direction": {"ThermalModel": "high"},
    },
    "dashboard": {"health_resample": "1h"},
}


def turbine_frame(offset: float) -> pd.DataFrame:
    timestamps = pd.date_range("2022-01-01", periods=6, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps[::-1],  # unsorted on disk
            "is_normal": True,
            "power": 1000.0,
            "gear_oil_temp": 50.0 + offset,
            "expected_gear_oil_temp_physics": 50.0,
        }
    )


def test_read_columns_skips_missing_columns_and_sorts_by_time(tmp_path) -> None:
    path = tmp_path / "T01.parquet"
    turbine_frame(0.0).to_parquet(path)

    frame = read_columns(path, ["timestamp", "power", "not_there"])

    assert frame.columns.tolist() == ["timestamp", "power"]
    assert frame["timestamp"].is_monotonic_increasing


def test_farm_normal_stacks_every_turbine(tmp_path) -> None:
    paths = {"T01": tmp_path / "a.parquet", "T02": tmp_path / "b.parquet"}
    for path in paths.values():
        turbine_frame(0.0).to_parquet(path)

    normal = farm_normal(paths)

    assert normal.columns.tolist() == ["timestamp", "is_normal", "turbine_id"]
    assert normal.groupby("turbine_id").size().tolist() == [6, 6]


def test_detail_columns_list_power_and_one_temperature_with_variants() -> None:
    columns = detail_columns(["physics"], "stator_temp", ["pinn"])

    assert columns == [
        "timestamp",
        "is_normal",
        "wind_speed",
        "power",
        "expected_power_physics",
        "stator_temp",
        "expected_stator_temp_pinn",
    ]
    assert "stator_temp" not in detail_columns(["physics"], None, [])


def test_component_scores_score_every_turbines_models(graph, tmp_path) -> None:
    for tid in ("T01", "T02"):
        add_model(
            graph,
            "ThermalModel",
            "KLM",
            tid,
            "gearbox",
            "physics",
            ["power"],
            "gear_oil_temp",
            METRICS,
            datetime(2026, 11, 20, tzinfo=UTC),
            qualifier="gear_oil_temp",
        )
    paths = {"T01": tmp_path / "T01.parquet", "T02": tmp_path / "T02.parquet"}
    turbine_frame(2.0).to_parquet(paths["T01"])
    turbine_frame(0.0).to_parquet(paths["T02"])
    models = with_model_ids(get_models(graph, "KLM"))

    scores = component_scores(CONFIG, models, paths)

    assert sorted(scores) == [
        "ThermalModel_KLM_T01_gear_oil_temp_physics",
        "ThermalModel_KLM_T02_gear_oil_temp_physics",
    ]
    assert scores["ThermalModel_KLM_T01_gear_oil_temp_physics"].tolist() == (
        pytest.approx([2.0])
    )  # one hourly value
