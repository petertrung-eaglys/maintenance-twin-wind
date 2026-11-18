"""Tests for the static yaw misalignment model on synthetic turbines."""

import numpy as np
import pandas as pd
import pytest

from src.yaw import (
    YawModel,
    cosine_power_ratio,
    fit_offset,
    offset_windows,
    peak_drop,
)

PHYSICS = {"reference_air_density": 1.225, "bin_width": 0.5, "min_bin_count": 5}
YAW = {
    "wind_speed_range": [5.0, 10.0],
    "max_misalignment_deg": 25.0,
    "bin_width_deg": 1.0,
    "min_bin_count": 20,
    "min_bins": 8,
    "edge_bins": 3,
    "min_peak_drop": 0.05,
    "window_days": 90,
    "step_days": 30,
}
EXPONENT = 2.0


def synthetic(
    offset: float, n: int = 40000, start: str = "2020-01-01", seed: int = 0
) -> pd.DataFrame:
    """A year of rows whose power drops as cos^2 of the true misalignment,
    measured with a vane offset (the controller centres the measured angle)."""
    rng = np.random.default_rng(seed)
    wind = rng.uniform(4, 11, n)
    gamma = rng.normal(0, 8, n)
    loss = cosine_power_ratio(gamma, 1.0, offset, EXPONENT)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="10min", tz="UTC"),
            "wind_speed": wind,
            "air_density": 1.225,
            "yaw_misalignment": gamma,
            "power": 2.0 * wind**3 * loss * rng.normal(1, 0.02, n),
        }
    )


def test_cosine_power_ratio_peaks_at_the_offset() -> None:
    ratio = cosine_power_ratio(np.array([0.0, 60.0, 180.0]), 1.0, 0.0, 1.0)

    assert ratio == pytest.approx([1.0, 0.5, 0.0])


def test_fit_offset_recovers_offset_and_exponent() -> None:
    frame = synthetic(offset=6.0)
    ratio = frame["power"] / (2.0 * frame["wind_speed"] ** 3)

    fit = fit_offset(ratio, frame["yaw_misalignment"], YAW)

    assert (fit["offset_deg"], fit["exponent"]) == pytest.approx((6.0, 2.0), abs=0.5)


def test_fit_offset_is_identified_for_a_visible_peak() -> None:
    frame = synthetic(offset=6.0)
    ratio = frame["power"] / (2.0 * frame["wind_speed"] ** 3)

    assert fit_offset(ratio, frame["yaw_misalignment"], YAW)["identified"]


def test_fit_offset_is_not_identified_without_a_peak() -> None:
    gamma = pd.Series(np.linspace(-20, 20, 4000))
    rising = 1 + 0.004 * gamma  # power rises across the whole range

    assert not fit_offset(rising, gamma, YAW)["identified"]


def test_peak_drop_takes_the_smaller_side_relative_to_the_gain() -> None:
    bins = pd.DataFrame({"y": [0.8, 0.9, 1.0, 0.95, 0.85]})

    assert peak_drop(bins, gain=1.0, edge_bins=2) == pytest.approx(0.1)


def test_peak_drop_is_negative_when_the_data_rise_past_the_peak() -> None:
    bins = pd.DataFrame({"y": [0.9, 1.0, 1.1]})

    assert peak_drop(bins, gain=1.0, edge_bins=1) < 0


def test_fit_offset_ignores_rows_without_a_ratio() -> None:
    frame = synthetic(offset=6.0)
    ratio = frame["power"] / (2.0 * frame["wind_speed"] ** 3)
    outside = frame["yaw_misalignment"] > 15  # whole bins of NaN ratios

    fit = fit_offset(ratio.mask(outside), frame["yaw_misalignment"], YAW)

    assert fit["offset_deg"] == pytest.approx(6.0, abs=1.0)


def test_fit_offset_needs_enough_bins() -> None:
    gamma = pd.Series(np.repeat([0.0, 1.0, 2.0], 100))

    assert fit_offset(pd.Series(1.0, index=gamma.index), gamma, YAW) is None


def test_yaw_model_finds_the_static_offset() -> None:
    model = YawModel(PHYSICS, YAW).fit({"T01": synthetic(offset=-5.0)})

    assert model.offset["offset_deg"] == pytest.approx(-5.0, abs=0.5)


def test_yaw_model_predicts_power_lost_to_misalignment() -> None:
    frame = synthetic(offset=0.0)
    model = YawModel(PHYSICS, YAW).fit({"T01": frame})
    query = frame.iloc[:2].assign(wind_speed=8.0, yaw_misalignment=[0.0, 30.0])

    aligned, misaligned = model.predict(query)

    assert misaligned / aligned == pytest.approx(np.cos(np.deg2rad(30)) ** 2, abs=0.03)


def test_yaw_model_is_undefined_outside_the_wind_speed_range() -> None:
    frame = synthetic(offset=0.0)
    model = YawModel(PHYSICS, YAW).fit({"T01": frame})
    query = frame.iloc[:2].assign(wind_speed=[4.0, 12.0])

    assert np.isnan(model.predict(query)).all()


def test_offset_windows_track_a_drift() -> None:
    before = synthetic(offset=2.0, n=26000, start="2020-01-01")
    after = synthetic(offset=10.0, n=26000, start="2020-07-01", seed=1)
    model = YawModel(PHYSICS, YAW).fit({"T01": before})

    windows = offset_windows(pd.concat([before, after]), model, YAW)
    first, last = windows["offset_deg"].iloc[[0, -1]]

    assert (first, last) == pytest.approx((2.0, 10.0), abs=1.0)


def test_offset_windows_columns() -> None:
    frame = synthetic(offset=2.0, n=26000)
    model = YawModel(PHYSICS, YAW).fit({"T01": frame})

    windows = offset_windows(frame, model, YAW)

    assert list(windows.columns) == [
        "window_start",
        "window_end",
        "offset_deg",
        "exponent",
        "peak_drop",
        "identified",
        "n",
    ]
