"""Tests for the dashboard's model-comparison tables."""

import pandas as pd

from src.dashboard.tables import (
    EVALUATION_COLUMNS,
    drift_view,
    evaluation_view,
    turbine_metrics,
    wake_view,
)

VARIANTS = {"physics": "#1", "baseline": "#2", "hybrid": "#3"}


def test_turbine_metrics_pivot_one_farm_and_class_in_variant_order() -> None:
    metrics = pd.DataFrame(
        {
            "farm_code": ["KLM", "KLM", "KLM", "PEN"],
            "model_class": ["PowerModel", "PowerModel", "PitchModel", "PowerModel"],
            "turbine_id": ["T01", "T01", "T01", "T01"],
            "target": ["power", "power", "pitch", "power"],
            "variant": ["hybrid", "physics", "physics", "physics"],
            "nrmse": [0.035, 0.038, 1.0, 0.03],
        }
    )

    table = turbine_metrics(metrics, "KLM", "PowerModel", "nrmse", VARIANTS)

    assert table.columns.tolist() == ["physics", "hybrid"]
    assert table.loc[("power", "T01")].tolist() == [0.038, 0.035]


def test_evaluation_view_keeps_one_farm_and_scope() -> None:
    row = dict.fromkeys(EVALUATION_COLUMNS, 0)
    summary = [
        row | {"farm_code": "KLM", "scope": "component", "variant": "a"},
        row | {"farm_code": "KLM", "scope": "signal", "variant": "b"},
        row | {"farm_code": "PEN", "scope": "component", "variant": "c"},
    ]

    view = evaluation_view(summary, "KLM", "component")

    assert view.columns.tolist() == EVALUATION_COLUMNS
    assert view["variant"].tolist() == ["a"]


def test_drift_view_shows_months_of_one_farm() -> None:
    drift = [
        {
            "farm_code": "KLM",
            "turbine_id": "T04",
            "target": "gear_oil_temp",
            "variant": "physics",
            "start": "2023-07-01T00:00:00.000Z",
            "end": "2024-12-01T00:00:00.000Z",
            "peak": 5.2,
            "model": "x",
        },
        {
            "farm_code": "PEN",
            "turbine_id": "T15",
            "target": "gear_oil_temp",
            "variant": "physics",
            "start": "2024-05-01T00:00:00.000Z",
            "end": "2024-12-01T00:00:00.000Z",
            "peak": 7.0,
            "model": "y",
        },
    ]

    view = drift_view(drift, "KLM")

    assert view.to_dict("records") == [
        {
            "turbine_id": "T04",
            "target": "gear_oil_temp",
            "variant": "physics",
            "start": "2023-07",
            "end": "2024-12",
            "peak": 5.2,
        }
    ]


def test_wake_view_pivots_bias_per_turbine_and_variant() -> None:
    wake = {
        "k": {"KLM": 0.04},
        "bias": [
            {
                "farm_code": "KLM",
                "turbine_id": "T01",
                "variant": "wake",
                "bias_waked": 2.0,
                "bias_free": 1.0,
                "waked_share": 0.3,
            },
            {
                "farm_code": "PEN",
                "turbine_id": "T01",
                "variant": "wake",
                "bias_waked": 9.0,
                "bias_free": 9.0,
                "waked_share": 0.3,
            },
        ],
    }

    view = wake_view(wake, "KLM")

    assert view.loc["T01", ("bias_waked", "wake")] == 2.0
    assert len(view) == 1
