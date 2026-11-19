"""Tests for the report's wake-alert helpers."""

import numpy as np
import pandas as pd
import pytest

from src.report.wake import (
    ALL_ROWS,
    alert_time,
    comparable_rows,
    wake_alert_figure,
    waked_rows,
    waked_share_table,
)

COLORS = {"physics": "#2a78d6", "wake": "#008300"}


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=4, freq="h", tz="UTC"),
            "expected_power_wake": [50.0, 100.0, np.nan, 90.0],
            "expected_power_freestream": [100.0, 100.0, 100.0, 0.0],
        }
    )


def test_comparable_rows_need_both_variants_and_free_stream_power() -> None:
    assert comparable_rows(frame()).tolist() == [True, True, False, False]


def test_waked_rows_expect_clearly_less_than_free_stream_power() -> None:
    assert waked_rows(frame().iloc[:2], 0.95).tolist() == [True, False]


def test_alert_time_counts_rows_and_waked_rows_inside_each_alert() -> None:
    rows = frame().iloc[:2]
    alerts = pd.DataFrame(
        {
            "variant": ["physics"],
            "start": pd.to_datetime(["2022-01-01 00:00"], utc=True),
            "end": pd.to_datetime(["2022-01-01 00:30"], utc=True),
        }
    )

    records = alert_time(rows, alerts, np.array([True, False]))

    assert records == [
        {"variant": ALL_ROWS, "rows": 2, "waked": 1},
        {"variant": "physics", "rows": 1, "waked": 1},
    ]


def test_waked_share_table_orders_variants_and_converts_rows_to_hours() -> None:
    records = pd.DataFrame(
        {
            "farm_code": ["KLM"] * 3,
            "variant": ["wake", ALL_ROWS, "physics"],
            "rows": [6, 12, 6],
            "waked": [0, 6, 6],
        }
    )

    table = waked_share_table(records, ["physics", "wake", "pinn"])

    assert table.index.tolist() == [
        ("KLM", ALL_ROWS),
        ("KLM", "physics"),
        ("KLM", "wake"),
    ]
    assert table.loc[("KLM", ALL_ROWS)].tolist() == [2.0, 1.0, 0.5]
    assert table.loc[("KLM", "physics"), "waked_share"] == pytest.approx(1.0)


def test_wake_alert_figure_draws_one_bar_per_variant_and_the_base_rate() -> None:
    table = pd.DataFrame(
        {
            "hours": [2.0, 1.0, 1.0],
            "waked_hours": [1.0, 1.0, 0.0],
            "waked_share": [0.5, 1.0, 0.0],
        },
        index=pd.MultiIndex.from_tuples(
            [("KLM", ALL_ROWS), ("KLM", "physics"), ("KLM", "wake")],
            names=["farm_code", "variant"],
        ),
    )

    fig = wake_alert_figure(table, COLORS, "#52514e")

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["physics", "wake"]
    assert ax.lines[0].get_ydata()[0] == 0.5
