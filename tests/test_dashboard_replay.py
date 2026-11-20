"""Tests for the dashboard's replay helpers."""

import pandas as pd
import pytest

from src.dashboard.replay import (
    ALERT,
    NORMAL,
    NOT_NORMAL,
    active_alerts,
    advance,
    alerts_overlapping,
    clamp,
    farm_statuses,
    map_center,
    period_of,
    replay_start,
    row_at,
    sample_rows,
    turbine_status,
    value_of,
    window,
)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2022-01-01 00:00", "2022-01-01 00:10", "2022-01-01 00:20"], utc=True
            ),
            "is_normal": [True, False, True],
            "power": [1.0, 2.0, 3.0],
        }
    )


def alerts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turbine_id": ["T01", "T02"],
            "model": ["m1", "m2"],
            "start": [ts("2022-01-01 00:10"), ts("2022-01-02")],
            "end": [ts("2022-01-01 00:20"), ts("2022-01-03")],
        }
    )


def test_row_at_takes_the_latest_row_not_after_ts() -> None:
    assert row_at(frame(), ts("2022-01-01 00:15"))["power"] == 2.0
    assert row_at(frame(), ts("2022-01-01 00:20"))["power"] == 3.0
    assert row_at(frame(), ts("2021-12-31")) is None


def test_value_of_is_none_without_a_row_or_the_column() -> None:
    row = frame().iloc[0]

    assert (value_of(row, "power"), value_of(row, "x"), value_of(None, "power")) == (
        1.0,
        None,
        None,
    )


def test_active_alerts_include_the_period_bounds() -> None:
    assert active_alerts(alerts(), ts("2022-01-01 00:20"))["model"].tolist() == ["m1"]
    assert active_alerts(alerts(), ts("2022-01-01 12:00")).empty


def test_turbine_status_is_alert_first_then_normal_flag() -> None:
    normal, not_normal = frame().iloc[0], frame().iloc[1]

    assert turbine_status(not_normal, alerts().iloc[:1]) == ALERT
    assert turbine_status(normal, alerts().iloc[:0]) == NORMAL
    assert turbine_status(not_normal, alerts().iloc[:0]) == NOT_NORMAL
    assert turbine_status(None, alerts().iloc[:0]) == NOT_NORMAL


def test_farm_statuses_use_each_turbines_own_rows_and_alerts() -> None:
    normal = pd.concat(
        [frame().assign(turbine_id="T01"), frame().assign(turbine_id="T02")]
    )

    statuses = farm_statuses(normal, alerts(), ts("2022-01-01 00:15"))

    assert statuses == {"T01": ALERT, "T02": NOT_NORMAL}


def test_window_keeps_the_last_days_up_to_the_end() -> None:
    rows = window(frame(), ts("2022-01-01 00:10"), days=0)

    assert rows.empty
    assert len(window(frame(), ts("2022-01-01 00:10"), days=1)) == 2


def test_alerts_overlapping_a_window() -> None:
    chosen = alerts_overlapping(alerts(), ts("2022-01-01 00:15"), ts("2022-01-02"))

    assert chosen["model"].tolist() == ["m1", "m2"]


def test_sample_rows_caps_reproducibly_in_the_original_order() -> None:
    big = pd.DataFrame({"x": range(100)})

    sample = sample_rows(big, 10, seed=1)

    assert len(sample) == 10
    assert sample.index.is_monotonic_increasing
    assert sample.equals(sample_rows(big, 10, seed=1))
    assert sample_rows(big, 200).equals(big)


def test_clamp_and_advance_stay_inside_the_data() -> None:
    first, last = ts("2022-01-01"), ts("2022-01-02")

    assert clamp(ts("2021-01-01"), first, last) == first
    assert advance(first, 6, 10, last) == (ts("2022-01-01 01:00"), True)
    assert advance(ts("2022-01-01 23:30"), 6, 10, last) == (last, False)


def test_replay_starts_at_the_first_sample_of_the_test_period() -> None:
    start = replay_start(ts("2016-01-01"), ts("2024-12-31"), ts("2021-12-31 23:50"), 10)

    assert start == ts("2022-01-01 00:00")


def test_period_of_names_the_split() -> None:
    train_end, calibration_end = ts("2020-12-31 23:50"), ts("2021-12-31 23:50")

    periods = [
        period_of(ts(t), train_end, calibration_end)
        for t in ("2020-06-01", "2021-06-01", "2022-06-01")
    ]

    assert periods == ["training", "calibration", "test"]


def test_map_center_is_the_mean_position() -> None:
    turbines = pd.DataFrame({"lat": [52.0, 52.2], "long": [-1.0, -0.8]})

    assert map_center(turbines) == pytest.approx({"lat": 52.1, "lon": -0.9})
