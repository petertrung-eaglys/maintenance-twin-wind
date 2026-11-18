"""Tests for the physics and hybrid pitch models on synthetic turbines."""

import numpy as np
import pandas as pd
import pytest

from src.pitch import HybridPitchModel, PhysicsPitchModel, above_rated

PHYSICS = {
    "reference_air_density": 1.225,
    "bin_width": 0.5,
    "min_bin_count": 5,
}
HYBRID = {
    "features": ["wind_speed", "ambient_temp"],
    "lightgbm": {"n_estimators": 50},
}
RATED_WIND = 12.0


def synthetic(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Pitch 0 below rated, then rising 2 deg per m/s above it."""
    rng = np.random.default_rng(seed)
    wind = rng.uniform(4, 24, n)
    return pd.DataFrame(
        {
            "wind_speed": wind,
            "air_density": 1.225,
            "ambient_temp": rng.normal(10, 5, n),
            "pitch": np.where(wind >= RATED_WIND, 2 * (wind - RATED_WIND), 0.0),
        }
    )


def test_above_rated_uses_measured_wind_speed() -> None:
    frame = pd.DataFrame({"wind_speed": [11.9, 12.0, 20.0]})

    assert above_rated(frame, RATED_WIND).tolist() == [False, True, True]


def test_physics_pitch_follows_the_binned_curve_above_rated() -> None:
    frame = synthetic()
    model = PhysicsPitchModel(PHYSICS, RATED_WIND).fit({"T01": frame})
    query = frame.iloc[:3].assign(wind_speed=[13.0, 16.0, 20.0])

    assert model.predict(query) == pytest.approx([2.0, 8.0, 16.0], abs=0.3)


def test_physics_pitch_is_undefined_below_rated() -> None:
    model = PhysicsPitchModel(PHYSICS, RATED_WIND).fit({"T01": synthetic()})
    query = synthetic(n=1).assign(wind_speed=8.0)

    assert np.isnan(model.predict(query)).all()


def test_physics_pitch_ignores_rows_without_pitch() -> None:
    frame = synthetic()
    gaps = frame.assign(pitch=frame["pitch"].where(frame.index % 2 == 0))

    model = PhysicsPitchModel(PHYSICS, RATED_WIND).fit({"T01": gaps})

    assert model.curve["y"].notna().all()


def test_hybrid_pitch_learns_what_physics_misses() -> None:
    frame = synthetic()
    shifted = frame.assign(  # a temperature effect the binned curve cannot see
        pitch=frame["pitch"]
        + np.where(frame["wind_speed"] >= RATED_WIND, 0.3, 0) * frame["ambient_temp"]
    )
    physics = PhysicsPitchModel(PHYSICS, RATED_WIND).fit({"T01": shifted})
    hybrid = HybridPitchModel(physics, HYBRID, seed=42).fit({"T01": shifted})
    above = above_rated(shifted, RATED_WIND)

    def rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred[above] - shifted.loc[above, "pitch"]) ** 2)))

    assert rmse(hybrid.predict(shifted)) < 0.5 * rmse(physics.predict(shifted))


def test_hybrid_pitch_is_undefined_below_rated() -> None:
    frame = synthetic()
    physics = PhysicsPitchModel(PHYSICS, RATED_WIND).fit({"T01": frame})
    hybrid = HybridPitchModel(physics, HYBRID, seed=42).fit({"T01": frame})

    below = ~above_rated(frame, RATED_WIND)

    assert np.isnan(hybrid.predict(frame)[below]).all()
