"""Tests for the method of bins and the physics and hybrid power models."""

import numpy as np
import pandas as pd
import pytest

from src.binned import binned_means, interpolate
from src.physics_power import (
    BETZ_LIMIT,
    HybridPowerModel,
    PhysicsPowerModel,
    normalised_wind_speed,
    rotor_area,
)

PHYSICS = {
    "reference_air_density": 1.225,
    "bin_width": 0.5,
    "min_bin_count": 5,
    "max_power_fraction": 0.95,
    "cut_in_wind_speed": 3.0,
    "cut_out_wind_speed": 25.0,
}
HYBRID = {
    "features": ["wind_speed", "ambient_temp"],
    "lightgbm": {"n_estimators": 50},
}
RATED = 2050.0
CP = 0.45  # constant power coefficient of the synthetic turbines


def synthetic(diameter: float, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """A turbine whose power follows 1/2 rho A Cp v^3, capped at rated."""
    rng = np.random.default_rng(seed)
    wind = rng.uniform(3.5, 20, n)
    density = rng.normal(1.225, 0.03, n)
    power = 0.5 * density * rotor_area(diameter) * CP * wind**3 / 1000
    return pd.DataFrame(
        {
            "wind_speed": wind,
            "air_density": density,
            "ambient_temp": rng.normal(10, 5, n),
            "power": np.minimum(power, RATED),
            "rotor_diameter_m": diameter,
            "rated_power_kw": RATED,
        }
    )


def test_binned_means_per_bin_with_minimum_count() -> None:
    x = pd.Series([0.1, 0.2, 0.3, 0.6, 1.1, 1.2])
    y = pd.Series([1.0, 2.0, 3.0, 9.0, 4.0, 6.0])

    curve = binned_means(x, y, 0.5, min_count=2)

    assert curve[["y", "count"]].values.tolist() == [[2.0, 3], [5.0, 2]]


def test_interpolate_is_linear_and_flat_past_the_ends() -> None:
    curve = pd.DataFrame({"x": [1.0, 3.0], "y": [10.0, 30.0]})

    assert interpolate(curve, [0.0, 2.0, 5.0]).tolist() == [10.0, 20.0, 30.0]


def test_normalised_wind_speed_follows_cube_root_of_density_ratio() -> None:
    v = normalised_wind_speed(pd.Series([10.0]), pd.Series([1.225 * 8]), 1.225)

    assert v.iloc[0] == pytest.approx(20.0)


def test_physics_recovers_the_power_coefficient_below_rated() -> None:
    model = PhysicsPowerModel(PHYSICS).fit({"T01": synthetic(92.0)})
    below_rated = model.cp_curve[model.cp_curve["x"] < 8]

    assert below_rated["y"].to_numpy() == pytest.approx(CP, rel=0.02)


def test_physics_curve_stays_below_betz() -> None:
    model = PhysicsPowerModel(PHYSICS).fit({"T01": synthetic(92.0)})

    assert model.max_cp() < BETZ_LIMIT


def test_physics_transfers_to_another_rotor_size() -> None:
    model = PhysicsPowerModel(PHYSICS).fit({"T01": synthetic(92.0)})
    other = synthetic(82.0, seed=1)

    error = model.predict(other) - other["power"]

    assert np.sqrt(np.mean(error**2)) < 0.01 * RATED


def test_physics_is_zero_outside_cut_in_cut_out_and_capped() -> None:
    model = PhysicsPowerModel(PHYSICS).fit({"T01": synthetic(92.0)})
    frame = synthetic(92.0, n=3).assign(wind_speed=[2.0, 18.0, 30.0], air_density=1.225)

    assert model.predict(frame).tolist() == [0.0, RATED, 0.0]


def test_hybrid_learns_what_physics_misses() -> None:
    frame = synthetic(92.0)
    shifted = frame.assign(  # a temperature effect the physics cannot see
        power=np.clip(frame["power"] - 5 * frame["ambient_temp"], 0, RATED)
    )
    physics = PhysicsPowerModel(PHYSICS).fit({"T01": shifted})
    hybrid = HybridPowerModel(physics, HYBRID, seed=42).fit({"T01": shifted})

    def rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - shifted["power"]) ** 2)))

    assert rmse(hybrid.predict(shifted)) < 0.5 * rmse(physics.predict(shifted))


def test_hybrid_is_clipped_to_rated_power() -> None:
    frame = synthetic(92.0)
    physics = PhysicsPowerModel(PHYSICS).fit({"T01": frame})
    hybrid = HybridPowerModel(physics, HYBRID, seed=42).fit({"T01": frame})

    predicted = hybrid.predict(frame)

    assert (predicted.min() >= 0, predicted.max() <= RATED) == (True, True)


def test_physics_cp_curve_excludes_power_limited_bins() -> None:
    model = PhysicsPowerModel(PHYSICS).fit({"T01": synthetic(92.0)})

    # rated is reached near 10.4 m/s for this rotor; later bins are capped
    assert model.cp_curve["x"].max() < 10.5


def test_physics_cp_curve_stops_at_first_power_limited_bin() -> None:
    frame = synthetic(92.0)
    storm = frame["wind_speed"] > 17  # storm control lowers power again
    derated = frame.assign(power=frame["power"].where(~storm, 0.5 * RATED))

    model = PhysicsPowerModel(PHYSICS).fit({"T01": derated})

    assert model.cp_curve["x"].max() < 10.5
