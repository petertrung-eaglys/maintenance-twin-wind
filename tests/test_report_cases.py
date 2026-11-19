"""Tests for the report's fault case-study helpers."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.graph.graph_utils import add_model
from src.report.cases import (
    OTHER,
    case_events,
    case_figure,
    case_scores,
    event_types,
    resampled_mean,
)

TRAINED_AT = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
METRICS = {"rmse": 1.0, "mae": 1.0, "nrmse": 1.0, "n_train": 6, "n_test": 6}
ALERTS = {
    "window": "1h",
    "min_samples": 1,
    "min_power_kw": {"ThermalModel": 0.0},
    "fleet_relative": False,
    "fleet_min_turbines": 1,
    "direction": {"ThermalModel": "high"},
}
STYLE = {
    "colors": {
        "grid": "#e4e3df",
        "muted": "#52514e",
        "text": "#0b0b0b",
        "variants": {"physics": "#2a78d6", "baseline": "#eb6834"},
    },
    "case_resample": "1D",
    "max_event_types": 1,
}


def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turbine_id": ["T01", "T01", "T01", "T02", "T01"],
            "component": ["gearbox", "gearbox", "gearbox", "gearbox", "pitch"],
            "start": pd.to_datetime(
                ["2020-12-01", "2021-02-01", "2021-03-01", "2021-02-01", "2021-02-01"],
                utc=True,
            ),
            "message": ["Old", "Particle", "Particle", "Particle", "Pitch"],
        }
    )


def turbine_frame(offset: float) -> pd.DataFrame:
    timestamps = pd.date_range("2022-01-01", periods=6, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "is_normal": True,
            "power": 1000.0,
            "gear_oil_temp": 50.0 + offset,
            "expected_gear_oil_temp_physics": 50.0,
            "expected_gear_oil_temp_baseline": 49.0,
        }
    )


def test_case_scores_return_each_thermal_variant_of_the_case_turbine(
    graph, tmp_path
) -> None:
    for tid in ("T01", "T02"):
        for variant in ("physics", "baseline"):
            add_model(
                graph,
                "ThermalModel",
                "KLM",
                tid,
                "gearbox",
                variant,
                ["power"],
                "gear_oil_temp",
                METRICS,
                TRAINED_AT,
                qualifier="gear_oil_temp",
            )
    paths = {"T01": tmp_path / "T01.parquet", "T02": tmp_path / "T02.parquet"}
    turbine_frame(2.0).to_parquet(paths["T01"])
    turbine_frame(0.0).to_parquet(paths["T02"])
    case = {"turbine_id": "T01", "target": "gear_oil_temp"}

    scores = case_scores({"alerts": ALERTS}, graph, "KLM", paths, case)

    assert sorted(scores) == ["baseline", "physics"]
    assert scores["physics"].tolist() == pytest.approx([2.0] * 6)
    assert scores["baseline"].tolist() == pytest.approx([3.0] * 6)


def test_case_events_keep_one_turbine_and_component_from_the_start() -> None:
    chosen = case_events(
        events(), "T01", "gearbox", pd.Timestamp("2021-01-01", tz="UTC")
    )

    assert chosen.index.tolist() == [1, 2]


def test_event_types_group_rare_messages_as_other() -> None:
    messages = pd.Series(["a", "b", "a", "c"])

    assert event_types(messages, max_types=1).tolist() == ["a", OTHER, "a", OTHER]


def test_resampled_mean_averages_each_period_on_naive_utc_time() -> None:
    index = pd.date_range("2022-01-01 22:00", periods=4, freq="h", tz="Europe/London")
    score = pd.Series([1.0, 3.0, 5.0, 7.0], index=index)

    mean = resampled_mean(score, "1D")

    assert mean.index.tz is None
    assert mean.tolist() == [2.0, 6.0]


def test_case_figure_labels_variants_in_order_and_counts_event_types() -> None:
    index = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
    scores = {
        "baseline": pd.Series(1.0, index=index),
        "physics": pd.Series(2.0, index=index),
    }
    chosen = case_events(events(), "T01", "gearbox", index[0])
    chosen = pd.concat([chosen, chosen.assign(message="Rare")])
    calibration = (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))

    fig = case_figure(scores, chosen, calibration, STYLE, "case")

    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["physics", "baseline", "Particle (2)", f"{OTHER} (2)"]
