"""Tests for the report's data-overview helpers."""

import numpy as np
import pandas as pd
import pytest

from src.report.coverage import (
    availability,
    coverage_figure,
    monthly_normal_share,
    report_months,
    year_ticks,
    yearly_counts,
)
from src.report.style import sequential_map

COLORS = {
    "grid": "#e4e3df",
    "sequential": ["#cde2fb", "#0d366b"],
}


def ten_minute_rows(start: str, days: int, normal: bool) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=days * 144, freq="10min", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "is_normal": normal})


def test_monthly_normal_share_counts_normal_slots_and_zero_for_empty_months() -> None:
    frame = pd.concat(
        [
            ten_minute_rows("2020-01-01", 2, True),
            ten_minute_rows("2020-01-10", 1, False),
        ]
    )
    months = report_months([2020])

    share = monthly_normal_share(frame, months, interval_minutes=10)

    assert len(share) == 12
    assert share.iloc[0] == pytest.approx(2 / 31)
    assert share.iloc[1:].eq(0).all()


def test_availability_pools_rows_with_a_value_over_turbines() -> None:
    one = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-06-01", "2021-06-01"], utc=True),
            "a": [1.0, np.nan],
        }
    )
    two = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-07-01"] * 3, utc=True),
            "a": [np.nan, 2.0, 3.0],
        }
    )

    share = availability([yearly_counts(one, ["a"]), yearly_counts(two, ["a"])])

    assert share.loc["a", 2020] == pytest.approx(3 / 4)
    assert share.loc["a", 2021] == 0.0


def test_year_ticks_mark_each_january() -> None:
    positions, labels = year_ticks(pd.period_range("2019-11", "2021-02", freq="M"))

    assert positions == [2, 14]
    assert labels == ["2020", "2021"]


def test_coverage_figure_draws_one_heatmap_per_farm() -> None:
    months = report_months([2020])
    shares = {
        "KLM": pd.DataFrame(0.5, index=["T01", "T02"], columns=months),
        "PEN": pd.DataFrame(1.0, index=["T01"], columns=months),
    }

    fig = coverage_figure(shares, sequential_map(COLORS), COLORS)

    heatmaps = [ax for ax in fig.axes if ax.images]
    assert [ax.get_ylabel() for ax in heatmaps] == ["KLM", "PEN"]
    assert heatmaps[0].images[0].get_array().shape == (2, 12)
