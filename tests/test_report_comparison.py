"""Tests for the report's model-comparison and cross-site helpers."""

import pandas as pd
import pytest

from src.report.comparison import (
    comparison_figure,
    cross_site_figure,
    cross_site_panels,
    cross_site_table,
    in_order,
    metric_frame,
    signal_label,
    variant_means,
)

VARIANTS = ["physics", "baseline", "hybrid", "pinn"]
COLORS = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))


def record(farm: str, tid: str, variant: str, rmse: float) -> dict:
    return {
        "farm_code": farm,
        "turbine_id": tid,
        "target": "stator_temp",
        "variant": variant,
        "rmse": rmse,
    }


def summary() -> pd.DataFrame:
    rows = [
        ("KLM", "PEN", "power", "baseline", 60.0, 200.0),
        ("KLM", "PEN", "power", "physics", 70.0, 90.0),
        ("KLM", "PEN", "stator_temp", "physics", 4.0, 5.0),
        ("KLM", "PEN", "gear_oil_temp", "physics", 2.0, 3.0),
        ("KLM", "PEN", "stator_temp", "pinn", 2.0, 6.0),
    ]
    return pd.DataFrame(
        rows, columns=["train", "test", "signal", "variant", "rmse_in", "rmse_cross"]
    )


def test_signal_label_drops_the_unit_suffix_and_underscores() -> None:
    assert signal_label("generator_bearing_rear_temp") == "generator bearing rear"
    assert signal_label("power") == "power"


def test_metric_frame_takes_the_model_class_from_the_model_id() -> None:
    metrics = {
        "PowerModel_KLM_T01_baseline": {"rmse": 1.0},
        "ThermalModel_KLM_T01_stator_temp_pinn": {"rmse": 2.0},
    }

    assert metric_frame(metrics)["model_class"].tolist() == [
        "PowerModel",
        "ThermalModel",
    ]


def test_variant_means_average_turbines_in_the_configured_variant_order() -> None:
    frame = pd.DataFrame(
        [
            record("KLM", "T01", "pinn", 1.0),
            record("KLM", "T02", "pinn", 3.0),
            record("KLM", "T01", "physics", 4.0),
        ]
    ).assign(model_class="ThermalModel")

    table = variant_means(frame, "ThermalModel", "rmse", VARIANTS)

    assert table.columns.tolist() == ["physics", "pinn"]
    assert table.loc[("stator_temp", "KLM")].tolist() == [4.0, 2.0]


def test_in_order_sorts_rows_by_variant_and_skips_missing_ones() -> None:
    table = pd.DataFrame({"x": [1, 2]}, index=["hybrid", "physics"])

    assert in_order(table, COLORS).index.tolist() == ["physics", "hybrid"]


def test_cross_site_panels_split_power_and_average_the_temperatures() -> None:
    panels = cross_site_panels(summary())

    assert panels["power"].loc[("KLM → PEN", "baseline"), "rmse_cross"] == 200.0
    physics = panels["thermal"].loc[("KLM → PEN", "physics")]
    assert physics.tolist() == [pytest.approx(3.0), pytest.approx(4.0)]


def test_cross_site_table_shows_in_and_cross_rmse_with_a_dash_if_missing() -> None:
    table = cross_site_table(summary(), VARIANTS)

    row = table.loc[("KLM", "PEN", "power")]
    assert row["physics"] == "70.00 → 90.00"
    assert row["pinn"] == "–"


def test_cross_site_figure_keeps_the_variant_order_in_every_panel() -> None:
    fig = cross_site_figure(cross_site_panels(summary()), COLORS, "#52514e")

    labels = [[t.get_text() for t in ax.get_xticklabels()] for ax in fig.axes]
    assert labels == [["physics", "baseline"], ["physics", "pinn"]]


def test_comparison_legend_lists_every_variant_in_the_configured_order() -> None:
    power = pd.DataFrame(
        {"physics": [3.8], "wake": [7.2]},
        index=pd.MultiIndex.from_tuples([("power", "KLM")], names=["target", "f"]),
    )
    pitch = pd.DataFrame(
        {"physics": [1.0]},
        index=pd.MultiIndex.from_tuples([("pitch", "KLM")], names=["target", "f"]),
    )
    thermal = pd.DataFrame(
        {"physics": [3.0], "pinn": [2.0]},
        index=pd.MultiIndex.from_tuples(
            [("stator_temp", "KLM")], names=["target", "f"]
        ),
    )
    colors = COLORS | {"wake": "#008300"}

    fig = comparison_figure(power, pitch, thermal, colors)

    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == ["physics", "pinn", "wake"]
