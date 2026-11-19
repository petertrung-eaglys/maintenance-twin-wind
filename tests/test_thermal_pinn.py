"""Tests for the thermal PINN on synthetic turbines."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.thermal.pinn import ThermalPINN, ode_residual

THERMAL = {
    "reference": "nacelle_temp",
    "power_scale_kw": 1000.0,
    "rotor_speed_scale_rpm": 15.0,
    "device": "cpu",
    "pinn": {
        "hidden": 32,
        "layers": 2,
        "embedding_dim": 2,
        "ode_weight": 0.1,
        "steps": 400,
        "batch_per_turbine": 128,
        "learning_rate": 0.01,
    },
}
FEATURES = ["power", "rotor_diameter_m"]
OFFSET = {"KLM_T01": 0.0, "PEN_T01": 5.0}  # PEN_T01 runs 5 degC hotter


def synthetic(key: str, n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Temperature following power quasi-statically; power drifts slowly,
    so temperature changes at a physical rate (about 1 degC per hour)."""
    rng = np.random.default_rng(seed)
    power = np.clip(1000 + np.cumsum(rng.normal(0, 20, n)), 0, 2000)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC"),
            "turbine_key": key,
            "power": power,
            "rotor_speed": 10.0,
            "nacelle_temp": 20.0,
            "rotor_diameter_m": 92.0 if key.startswith("KLM") else 82.0,
            "is_normal": True,
            "temp": 30 + 0.01 * power + OFFSET[key],
        }
    )


def fitted(seed: int = 0) -> tuple[ThermalPINN, dict[str, pd.DataFrame]]:
    frames = {k: synthetic(k, seed=i) for i, k in enumerate(OFFSET)}
    model = ThermalPINN("temp", THERMAL, 10, FEATURES, seed=seed).fit(frames)
    return model, frames


def test_ode_residual_is_zero_when_temperature_follows_the_heat_balance() -> None:
    raw = torch.tensor([[0.1, 0.0, 0.0, 0.0, -20.0]])  # c = 0.1, h0 = ln 2
    h0 = float(np.log(2.0))
    temp = torch.tensor([30.0])
    following = temp + 0.1 - h0 * (temp - 20.0)

    residual = ode_residual(
        raw,
        temp,
        following,
        power=torch.tensor([0.0]),
        speed=torch.tensor([0.0]),
        reference=torch.tensor([20.0]),
        steps_per_hour=6.0,
    )

    assert residual.item() == pytest.approx(0.0, abs=1e-5)


def test_pinn_learns_each_turbines_temperature() -> None:
    model, frames = fitted()

    rmse = [
        np.sqrt(np.mean((model.predict(f) - f["temp"]) ** 2)) for f in frames.values()
    ]

    assert max(rmse) < 1.5


def test_pinn_is_reproducible_with_a_seed() -> None:
    first, frames = fitted(seed=3)
    second, _ = fitted(seed=3)
    frame = frames["KLM_T01"]

    assert np.array_equal(first.predict(frame), second.predict(frame))


def test_pinn_predicts_nan_where_features_are_missing() -> None:
    model, frames = fitted()
    frame = frames["KLM_T01"].iloc[:3].assign(power=[1000.0, np.nan, 500.0])

    assert np.isnan(model.predict(frame)).tolist() == [False, True, False]


def test_pinn_rejects_an_unknown_turbine() -> None:
    model, frames = fitted()

    with pytest.raises(KeyError):
        model.predict(frames["KLM_T01"].assign(turbine_key="PEN_T15"))


def test_pinn_can_predict_an_unseen_turbine_with_the_mean_embedding() -> None:
    frames = {k: synthetic(k, seed=i) for i, k in enumerate(OFFSET)}
    model = ThermalPINN(
        "temp", THERMAL, 10, FEATURES, seed=0, unseen_turbines="mean"
    ).fit(frames)
    unseen = frames["KLM_T01"].assign(turbine_key="KLM_T09")

    assert np.isfinite(model.predict(unseen)).all()
