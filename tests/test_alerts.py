"""Tests for residual-based alerts on synthetic series."""

import numpy as np
import pandas as pd
import pytest

from src.health.alerts import (
    alert_periods,
    calibration_threshold,
    drift_periods,
    merge_periods,
    yaw_alert_periods,
    yearly_means,
)
from src.health.scores import problem_score, smoothed_residual

TRAIN_END = pd.Timestamp("2020-12-31 23:50", tz="UTC")
CAL_END = pd.Timestamp("2021-12-31 23:50", tz="UTC")


def times(start: str, n: int, freq: str = "10min") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def test_smoothed_residual_uses_normal_rows_with_both_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": times("2022-01-01", 4),
            "is_normal": [True, False, True, True],
            "temp": [50.0, 99.0, 52.0, np.nan],
            "expected_temp_pinn": [48.0, 50.0, 48.0, 48.0],
        }
    )

    smoothed = smoothed_residual(frame, "temp", "expected_temp_pinn", "1h", 1)

    # rows 1 (not normal) and 3 (no measurement) are left out
    assert smoothed.tolist() == [2.0, 3.0]


def test_problem_score_is_larger_when_worse() -> None:
    residual = pd.Series([-3.0, 2.0])

    assert (
        problem_score(residual, "low").tolist(),
        problem_score(residual, "high").tolist(),
        problem_score(residual, "both").tolist(),
    ) == ([3.0, -2.0], [-3.0, 2.0], [3.0, 2.0])


def test_threshold_comes_from_the_calibration_period_only() -> None:
    index = pd.DatetimeIndex(
        times("2020-06-01", 1, "D").append(times("2021-01-01", 100, "D"))
    )
    score = pd.Series([1000.0, *np.arange(100.0)], index=index)

    threshold = calibration_threshold(score, TRAIN_END, CAL_END, quantile=0.5)

    # the 2020 outlier is ignored: median of 0..99
    assert threshold == pytest.approx(49.5)


def test_alert_periods_need_the_minimum_duration() -> None:
    score = pd.Series([0, 5, 5, 0, 5, 5, 5, 5, 0], index=times("2022-01-01", 9, "h"))

    periods = alert_periods(score, 1.0, pd.Timedelta("6h"), pd.Timedelta("2h"))

    assert periods.values.tolist() == [
        [
            pd.Timestamp("2022-01-01 04:00", tz="UTC"),
            pd.Timestamp("2022-01-01 07:00", tz="UTC"),
            5.0,
        ]
    ]


def test_alert_periods_split_at_long_gaps() -> None:
    index = times("2022-01-01", 3, "h").append(times("2022-01-03", 3, "h"))
    score = pd.Series(5.0, index=index)

    periods = alert_periods(score, 1.0, pd.Timedelta("6h"), pd.Timedelta("1h"))

    assert len(periods) == 2


def test_alert_periods_report_the_peak_score() -> None:
    score = pd.Series([2.0, 7.0, 3.0], index=times("2022-01-01", 3, "h"))

    periods = alert_periods(score, 1.0, pd.Timedelta("6h"), pd.Timedelta("1h"))

    assert periods["peak"].tolist() == [7.0]


def test_yaw_alerts_on_identified_windows_that_drift() -> None:
    windows = pd.DataFrame(
        {
            "window_start": times("2022-01-01", 4, "30D"),
            "window_end": times("2022-03-31", 4, "30D"),
            "offset_deg": [5.0, 11.0, 12.0, 11.5],
            "identified": [True, True, True, False],
        }
    )

    periods = yaw_alert_periods(windows, static_offset=5.0, drift_deg=4.0, step_days=30)

    # windows 2 and 3 drift; window 4 is not identified
    assert periods[["start", "end"]].values.tolist() == [
        [windows["window_start"][1], windows["window_end"][2]]
    ]


def test_merge_periods_join_alerts_closer_than_the_gap() -> None:
    periods = pd.DataFrame(
        {
            "start": [
                pd.Timestamp(d, tz="UTC")
                for d in ("2022-01-01", "2022-01-05", "2022-02-01")
            ],
            "end": [
                pd.Timestamp(d, tz="UTC")
                for d in ("2022-01-02", "2022-01-06", "2022-02-02")
            ],
            "peak": [2.0, 5.0, 3.0],
        }
    )

    merged = merge_periods(periods, pd.Timedelta(days=7))

    assert merged.values.tolist() == [
        [
            pd.Timestamp("2022-01-01", tz="UTC"),
            pd.Timestamp("2022-01-06", tz="UTC"),
            5.0,
        ],
        [
            pd.Timestamp("2022-02-01", tz="UTC"),
            pd.Timestamp("2022-02-02", tz="UTC"),
            3.0,
        ],
    ]


def test_merge_periods_keeps_no_alerts_empty() -> None:
    empty = pd.DataFrame(columns=["start", "end", "peak"])

    assert merge_periods(empty, pd.Timedelta(days=7)).empty


def test_yearly_means_of_the_score() -> None:
    score = pd.Series([1.0, 3.0, 10.0], index=times("2021-12-31 23:00", 3, "30min"))

    assert yearly_means(score) == {2021: 2.0, 2022: 10.0}


def monthly_score(values: list[float]) -> pd.Series:
    """Daily scores from January 2021, constant within each calendar month."""
    months = pd.date_range("2021-01-01", periods=len(values), freq="MS", tz="UTC")
    parts = [
        pd.Series(value, index=pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D"))
        for m, value in zip(months, values, strict=True)
    ]
    return pd.concat(parts)


CALIBRATION_END = pd.Timestamp("2021-12-31 23:50", tz="UTC")


def test_drift_periods_need_enough_months_above_the_calibration_mean() -> None:
    # 2021-ish calibration around 0, then 2 months at +2 and later 3 at +2
    score = monthly_score([0.0] * 12 + [2.0, 2.0, 0.0, 2.0, 2.0, 2.0])
    train_end = pd.Timestamp("2020-12-31 23:50", tz="UTC")

    periods = drift_periods(score, train_end, CALIBRATION_END, margin=1.0, months=3)

    assert len(periods) == 1


def test_drift_periods_end_after_the_last_month() -> None:
    score = monthly_score([0.0] * 12 + [3.0, 3.0, 3.0])
    train_end = pd.Timestamp("2020-12-31 23:50", tz="UTC")

    periods = drift_periods(score, train_end, CALIBRATION_END, margin=1.0, months=3)

    assert (periods["peak"].tolist(), periods["end"].iloc[0] > score.index[-1]) == (
        [3.0],
        True,
    )
