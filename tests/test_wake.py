"""Tests for the wake-aware power helpers."""

import numpy as np
import pandas as pd
import pytest

from src.models.physics_power import PhysicsPowerModel
from src.models.wake import (
    FarmWind,
    WakeTable,
    bias_records,
    calibrate_k,
    direction_bias,
    expected_power,
    farm_direction,
    free_stream,
    local_xy,
    pooled_gap,
    turbine_views,
    wake_table,
)

WAKE = {
    "turbulence_intensity": 0.1,
    "rotor_average_points": 4,
    "wd_step_deg": 90,
    "ws_range": [6, 8],
    "free_ratio": 0.98,
    "bias_wind_speed": [5.0, 10.0],
}
PHYSICS = {
    "reference_air_density": 1.225,
    "cut_in_wind_speed": 3.0,
    "cut_out_wind_speed": 25.0,
}


def table() -> WakeTable:
    """Two turbines, wd 0/90/180/270, ws 6/7/8: turbine 1 is waked (0.7)
    when the wind comes from the north (wd 0), everything else is free."""
    ratio = np.ones((2, 4, 3))
    ratio[1, 0, :] = 0.7
    return WakeTable(
        ["T01", "T02"],
        np.array([0.0, 90.0, 180.0, 270.0]),
        np.array([6.0, 7, 8]),
        ratio,
    )


def test_local_xy_gives_metres_around_the_mean_position() -> None:
    east, north = local_xy(pd.Series([52.0, 52.001]), pd.Series([-1.0, -1.0]))

    assert east == pytest.approx([0.0, 0.0])
    assert north[1] - north[0] == pytest.approx(111.2, abs=0.1)
    assert north.sum() == pytest.approx(0.0, abs=1e-6)


def test_lookup_uses_the_nearest_point_wraps_directions_and_clips_speeds() -> None:
    ratio = table().lookup(
        np.array([1, 1, 1, 0, 1]),
        np.array([350.0, 80.0, 10.0, 0.0, np.nan]),
        np.array([7.0, 7.0, 30.0, 7.0, 7.0]),
    )

    assert ratio[:4].tolist() == [0.7, 1.0, 0.7, 1.0]
    assert np.isnan(ratio[4])


def test_wake_table_wakes_only_the_downwind_turbine() -> None:
    turbines = pd.DataFrame(
        {
            "turbine_id": ["T01", "T02"],
            "lat": [52.0, 52.0 - 5 * 92 / 111_200],  # T02 5 D south of T01
            "long": [-1.0, -1.0],
            "rotor_diameter_m": [92.0, 92.0],
            "hub_height_m": [78.5, 78.5],
            "rated_power_kw": [2050.0, 2050.0],
        }
    )

    result = wake_table(turbines, WAKE, k=0.04)

    northerly = result.ratio[:, 0, :]  # wind from the north
    assert northerly[0] == pytest.approx(1.0)
    assert (northerly[1] < 0.8).all()
    assert result.ratio[:, 1, :] == pytest.approx(1.0)  # from the east


def test_farm_direction_is_the_circular_mean_of_valid_turbines() -> None:
    directions = np.array([[350.0, 10.0, 180.0], [90.0, 0.0, 0.0]])
    valid = np.array([[True, True, False], [False, False, False]])

    wd = farm_direction(directions, valid)

    assert wd[0] == pytest.approx(0.0, abs=1e-9) or wd[0] == pytest.approx(360.0)
    assert np.isnan(wd[1])


def test_free_stream_averages_the_turbines_free_in_that_direction() -> None:
    speeds = np.array([[8.0, 6.0], [8.0, 7.0], [8.0, 7.0]])
    directions = np.array([[0.0, 0.0], [90.0, 90.0], [90.0, 90.0]])
    normal = np.array([[True, True], [True, True], [False, False]])

    wd, free_ws = free_stream(speeds, directions, normal, table(), 0.98)

    assert wd[1] == pytest.approx(90.0)
    assert free_ws[:2].tolist() == [8.0, 7.5]  # T02 waked from the north
    assert np.isnan(free_ws[2])  # no normal turbine


def test_expected_power_is_nan_where_the_wind_speed_is_missing() -> None:
    physics = PhysicsPowerModel.from_dict(
        PHYSICS,
        {"wind_speed_normalised": [4.0, 12.0], "cp": [0.4, 0.4], "count": [1, 1]},
    )
    frame = pd.DataFrame(
        {
            "wind_speed": [0.0, 0.0],
            "air_density": [1.225, 1.225],
            "rotor_diameter_m": [92.0, 92.0],
            "rated_power_kw": [2050.0, 2050.0],
        }
    )

    power = expected_power(physics, frame, np.array([8.0, np.nan]))

    area = np.pi * 92.0**2 / 4
    assert power[0] == pytest.approx(0.5 * 1.225 * area * 0.4 * 8.0**3 / 1000)
    assert np.isnan(power[1])


def test_direction_bias_is_the_relative_sum_of_residuals() -> None:
    power = pd.Series([90.0, 110.0, 50.0])
    expected = np.array([100.0, 100.0, 100.0])

    bias = direction_bias(power, expected, np.array([True, True, False]))

    assert bias == pytest.approx(0.0)


def test_bias_records_split_waked_and_free_rows_below_rated() -> None:
    frame = pd.DataFrame(
        {"power": [80.0, 100.0, 100.0, 0.0], "free_wind_speed": [7.0, 7.0, 7.0, 15.0]}
    )
    outputs = {"wake": np.array([80.0, 100.0, 100.0, 50.0])}
    test = pd.Series([True, True, True, True])
    ratio = np.array([0.7, 1.0, 1.0, 0.7])

    records = bias_records(frame, outputs, test, ratio, WAKE)

    assert records == [
        {"variant": "wake", "bias_waked": 0.0, "bias_free": 0.0, "waked_share": 1 / 3}
    ]


def test_physics_curve_round_trips_through_to_dict() -> None:
    saved = {"wind_speed_normalised": [4.0, 5.0], "cp": [0.3, 0.4], "count": [20, 30]}

    model = PhysicsPowerModel.from_dict(PHYSICS, saved)

    assert model.to_dict() == saved


def test_wider_wakes_recover_faster_downwind() -> None:
    turbines = pd.DataFrame(
        {
            "turbine_id": ["T01", "T02"],
            "lat": [52.0, 52.0 - 5 * 92 / 111_200],
            "long": [-1.0, -1.0],
            "rotor_diameter_m": [92.0, 92.0],
            "hub_height_m": [78.5, 78.5],
            "rated_power_kw": [2050.0, 2050.0],
        }
    )

    narrow = wake_table(turbines, WAKE, k=0.03).ratio[1, 0, 1]
    wide = wake_table(turbines, WAKE, k=0.10).ratio[1, 0, 1]

    assert narrow < wide < 1.0


def test_calibrate_k_finds_where_the_gap_crosses_zero() -> None:
    k = calibrate_k(lambda k: 0.1 - k, [0.02, 0.2], tolerance=1e-4)

    assert k == pytest.approx(0.1, abs=1e-4)


def test_calibrate_k_returns_the_closer_end_without_a_crossing() -> None:
    assert calibrate_k(lambda k: 1.0 - k, [0.02, 0.2], 1e-4) == 0.2
    assert calibrate_k(lambda k: -1.0 - k, [0.02, 0.2], 1e-4) == 0.02


def test_pooled_gap_weights_turbines_by_their_expected_power() -> None:
    parts = [
        (  # waked rows 10 % low, free rows exact
            pd.Series([90.0, 100.0]),
            np.array([100.0, 100.0]),
            np.array([True, True]),
            np.array([True, False]),
        ),
        (  # one more waked row, exact, with three times the expected power
            pd.Series([300.0]),
            np.array([300.0]),
            np.array([True]),
            np.array([True]),
        ),
    ]

    assert pooled_gap(parts) == pytest.approx(-2.5)  # -10 / 400


def test_turbine_views_add_free_stream_speed_and_each_turbines_ratio() -> None:
    timestamps = pd.date_range("2022-01-01", periods=2, freq="10min", tz="UTC")
    frames = {
        tid: pd.DataFrame({"timestamp": timestamps, "power": [1.0, 2.0]})
        for tid in ("T01", "T02")
    }
    wind = FarmWind(
        pd.Index(timestamps),
        np.array([[8.0, 6.0], [7.0, 7.0]]),
        np.array([[0.0, 0.0], [90.0, 90.0]]),
        np.ones((2, 2), bool),
    )

    views = turbine_views(frames, wind, table(), free_ratio=0.98)

    view, ratio = views["T02"]
    assert view["free_wind_speed"].tolist() == [8.0, 7.0]
    assert ratio.tolist() == [0.7, 1.0]
