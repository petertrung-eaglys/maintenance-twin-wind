"""Tests for the cleaning rules, derived signals and stop-event masking."""

import numpy as np
import pandas as pd
import pytest

from src.clean import (
    add_cleaning_columns,
    air_density,
    normal_mask,
    stop_active,
    yaw_misalignment,
)

CLEANING = {
    "required": ["wind_speed", "wind_direction", "ambient_temp", "power"],
    "min_wind_speed": 3.0,
    "max_wind_speed": 25.0,
    "max_pitch_deg": 5.0,
    "min_power_kw": 0.0,
    "standard_pressure_pa": 101325.0,
}
RATED = 12.0
NAN = float("nan")
TEN_MIN = pd.Timedelta(minutes=10)


def row(**overrides: float) -> dict:
    """A normal row, with selected values overridden."""
    base = {
        "wind_speed": 8.0,
        "wind_direction": 200.0,
        "nacelle_direction": 190.0,
        "ambient_temp": 10.0,
        "power": 1000.0,
        "pitch": 0.0,
    }
    return {**base, **overrides}


def mask_of(*rows: dict) -> list[bool]:
    return normal_mask(pd.DataFrame(rows), CLEANING, RATED).tolist()


def test_typical_operating_row_is_normal() -> None:
    assert mask_of(row()) == [True]


def test_missing_required_signal_is_not_normal() -> None:
    assert mask_of(row(ambient_temp=NAN), row(power=NAN)) == [False, False]


def test_wind_speed_outside_range_is_not_normal() -> None:
    assert mask_of(row(wind_speed=2.9), row(wind_speed=25.1), row(wind_speed=3.0)) == [
        False,
        False,
        True,
    ]


def test_non_positive_power_is_not_normal() -> None:
    assert mask_of(row(power=0.0), row(power=-5.0)) == [False, False]


def test_pitch_rule_applies_below_rated_wind_only() -> None:
    below, above = row(wind_speed=8.0, pitch=10.0), row(wind_speed=15.0, pitch=10.0)

    assert mask_of(below, above) == [False, True]


def test_missing_pitch_is_accepted() -> None:
    assert mask_of(row(pitch=NAN)) == [True]


def test_air_density_at_standard_conditions() -> None:
    assert air_density(pd.Series([15.0]), 101325.0)[0] == pytest.approx(1.225, abs=1e-3)


def test_yaw_misalignment_wraps_to_plus_minus_180() -> None:
    wind = pd.Series([10.0, 350.0, 200.0, 0.0])
    nacelle = pd.Series([350.0, 10.0, 190.0, 180.0])

    assert yaw_misalignment(wind, nacelle).tolist() == [20.0, -20.0, 10.0, -180.0]


def times(*clock: str) -> pd.Series:
    return pd.Series(pd.to_datetime([f"2022-01-01 {c}" for c in clock], utc=True))


def stops(*spans: tuple[str, str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.to_datetime([f"2022-01-01 {s}" for s, _ in spans], utc=True),
            "end": pd.to_datetime(
                [f"2022-01-01 {e}" if e else None for _, e in spans], utc=True
            ),
        }
    )


def test_stop_active_marks_rows_overlapping_a_stop() -> None:
    rows = times("00:00", "00:10", "00:20", "00:30", "00:40")

    active = stop_active(rows, stops(("00:15", "00:30")), TEN_MIN)

    assert active.tolist() == [False, True, True, True, False]


def test_stop_without_end_marks_the_row_containing_its_start() -> None:
    rows = times("00:00", "00:10", "00:20")

    active = stop_active(rows, stops(("00:12", None)), TEN_MIN)

    assert active.tolist() == [False, True, False]


def test_stop_active_without_stops_is_all_false() -> None:
    assert stop_active(times("00:00"), stops(), TEN_MIN).tolist() == [False]


def test_add_cleaning_columns_combines_rules_service_date_and_stops() -> None:
    frame = pd.DataFrame([row(), row(), row(), row(power=0.0)]).assign(
        timestamp=times("00:00", "00:10", "00:20", "00:30")
    )
    commissioned = pd.Timestamp("2022-01-01 00:10", tz="UTC")
    stopped = np.array([False, False, True, False])

    cleaned = add_cleaning_columns(frame, CLEANING, RATED, commissioned, stopped)

    assert (
        cleaned["is_normal"].tolist(),
        cleaned["yaw_misalignment"].iloc[0],
        "air_density" in cleaned,
    ) == ([False, True, False, False], 10.0, True)
