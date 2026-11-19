"""Tests for the grey-box thermal model on synthetic turbines."""

import numpy as np
import pandas as pd
import pytest

from src.models.thermal.physics import (
    ThermalPhysicsModel,
    coefficients,
    cooler,
    simulate,
    steady_state,
)

THERMAL = {
    "reference": "nacelle_temp",
    "power_scale_kw": 1000.0,
    "rotor_speed_scale_rpm": 15.0,
    "device": "cpu",
    "physics": {
        "sequence_steps": 48,
        "warmup_steps": 3,
        "iterations": 300,
        "learning_rate": 0.05,
        "batch_sequences": 64,
    },
}
TRUE = {
    "KLM_T01": {"c": 0.05, "a": 0.2, "b": 0.1, "h0": 0.02, "h1": 0.03},
    "KLM_T02": {"c": 0.05, "a": 0.2, "b": 0.1, "h0": 0.05, "h1": 0.03},
}
# Without its cooler this turbine would reach ~40 degC above the nacelle at
# full power; the cooler holds it near 35 degC.
COOLED = {
    "c": 0.3,
    "a": 0.4,
    "b": 0.1,
    "h0": 0.02,
    "h1": 0.03,
    "h2": 0.2,
    "set_point": 35.0,
}


def synthetic(
    key: str, days: int = 40, seed: int = 0, params: dict | None = None
) -> pd.DataFrame:
    """A turbine whose temperature follows the heat balance exactly."""
    params = TRUE[key] if params is None else params
    rng = np.random.default_rng(seed)
    n = days * 144
    power = np.clip(np.cumsum(rng.normal(0, 60, n)) % 4000 - 1000, 0, 2050)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC"),
            "turbine_key": key,
            "power": power,
            "rotor_speed": 8 + 6 * power / 2050,
            "nacelle_temp": 20 + 5 * np.sin(np.arange(n) * 2 * np.pi / 144),
            "is_normal": True,
        }
    )
    alpha, beta = coefficients(frame, params, THERMAL)
    start = np.r_[30.0, np.full(n - 1, np.nan)]
    restart = np.r_[True, np.zeros(n - 1, dtype=bool)]
    temp = simulate(alpha, beta, start, restart, warmup=0, cooler=cooler(params))
    return frame.assign(temp=temp)


def test_simulate_restarts_from_the_measurement_after_a_gap() -> None:
    out = simulate(
        alpha=np.full(4, 0.5),
        beta=np.full(4, 10.0),
        measured=np.array([0.0, 99.0, 50.0, 99.0]),
        restart=np.array([True, False, True, False]),
        warmup=0,
    )

    assert out.tolist() == [0.0, 10.0, 50.0, 35.0]


def test_simulate_hides_the_warm_up_and_restarts_after_missing_inputs() -> None:
    out = simulate(
        alpha=np.array([0.5, np.nan, 0.5, 0.5]),
        beta=np.full(4, 10.0),
        measured=np.array([0.0, 99.0, 40.0, 99.0]),
        restart=np.array([True, False, False, False]),
        warmup=1,
    )

    # row 2 restarts from its measurement (row 1 had no inputs)
    assert np.isnan(out[[0, 2]]).all() and out[[1, 3]].tolist() == [10.0, 30.0]


def test_steady_state_balances_heat_and_cooling() -> None:
    params = {"c": 0.1, "a": 0.2, "b": 0.0, "h0": 0.1, "h1": 0.1}

    # (0.1 + 0.2 * 1) / (0.1 + 0.1 * 1) = 1.5 K above the nacelle
    assert steady_state(
        params, power_mw=1.0, speed=1.0, reference=20.0
    ) == pytest.approx(21.5)


def test_free_run_converges_to_the_steady_state() -> None:
    frame = synthetic("KLM_T01").assign(
        power=1000.0, rotor_speed=15.0, nacelle_temp=20.0
    )
    model = ThermalPhysicsModel("temp", THERMAL, interval_minutes=10)
    model.params = {"KLM_T01": TRUE["KLM_T01"]}

    final = model.predict(frame)[-1]

    expected = steady_state(TRUE["KLM_T01"], power_mw=1.0, speed=1.0, reference=20.0)
    assert final == pytest.approx(expected, abs=1e-6)


def test_fit_recovers_each_turbines_behaviour() -> None:
    frames = {k: synthetic(k, seed=i) for i, k in enumerate(TRUE)}
    model = ThermalPhysicsModel("temp", THERMAL, interval_minutes=10, seed=0)

    model.fit(frames)

    rmse = [
        np.sqrt(np.nanmean((model.predict(f) - f["temp"]) ** 2))
        for f in frames.values()
    ]
    assert max(rmse) < 0.3


def test_fit_keeps_parameters_per_turbine() -> None:
    frames = {k: synthetic(k, seed=i) for i, k in enumerate(TRUE)}
    model = ThermalPhysicsModel("temp", THERMAL, interval_minutes=10, seed=0)

    model.fit(frames)

    # T02 cools faster: a lower steady state at the same operating point
    t01, t02 = (
        steady_state(model.params[k], power_mw=1.0, speed=1.0, reference=20.0)
        for k in TRUE
    )
    assert t02 < t01


def test_predict_rejects_an_unknown_turbine() -> None:
    model = ThermalPhysicsModel("temp", THERMAL, interval_minutes=10)
    model.params = {"KLM_T01": TRUE["KLM_T01"]}

    with pytest.raises(KeyError):
        model.predict(synthetic("KLM_T01", days=1).assign(turbine_key="PEN_T09"))


def test_simulate_cools_harder_above_the_set_point() -> None:
    out = simulate(
        alpha=np.ones(3),
        beta=np.zeros(3),
        measured=np.array([60.0, np.nan, np.nan]),
        restart=np.array([True, False, False]),
        warmup=0,
        cooler=(0.5, 50.0),
    )

    assert out.tolist() == [60.0, 55.0, 52.5]


def test_steady_state_with_a_cooler_stays_near_the_set_point() -> None:
    params = {"c": 0.3, "a": 0.0, "b": 0.0, "h0": 0.2, "h1": 0.0}
    cooled = params | {"h2": 0.8, "set_point": 21.0}

    # without the cooler 21.5; with it (0.3 + 0.2*20 + 0.8*21) / (0.2 + 0.8)
    assert steady_state(cooled, 0.0, 0.0, reference=20.0) == pytest.approx(21.1)


def test_cooler_defaults_to_none() -> None:
    assert cooler({"h0": 0.1}) == (0.0, float("inf"))


def test_fit_learns_a_regulated_temperature() -> None:
    frame = synthetic("KLM_T01", params=COOLED)
    model = ThermalPhysicsModel("temp", THERMAL, interval_minutes=10, seed=0)

    model.fit({"KLM_T01": frame})

    rmse = np.sqrt(np.nanmean((model.predict(frame) - frame["temp"]) ** 2))
    assert rmse < 0.5


def test_pooled_fit_shares_one_parameter_set_for_unseen_turbines() -> None:
    frames = {
        "KLM_T01": synthetic("KLM_T01", params=TRUE["KLM_T01"]),
        "KLM_T02": synthetic("KLM_T02", seed=1, params=TRUE["KLM_T01"]),
    }
    model = ThermalPhysicsModel(
        "temp", THERMAL, interval_minutes=10, seed=0, pooled=True
    ).fit(frames)
    unseen = synthetic("KLM_T01", seed=2).assign(turbine_key="PEN_T05")

    rmse = np.sqrt(np.nanmean((model.predict(unseen) - unseen["temp"]) ** 2))

    assert rmse < 0.3
