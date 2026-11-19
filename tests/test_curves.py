"""Tests for detection-vs-false-alarm curves."""

import pandas as pd
import pytest

from src.health.curves import lift, threshold_alerts

TRAIN_END = pd.Timestamp("2020-12-31 23:50", tz="UTC")
CAL_END = pd.Timestamp("2021-12-31 23:50", tz="UTC")
ALERTS = {
    "max_gap_hours": 6,
    "min_duration_hours": 1,
    "merge_gap_days": 7,
}
MODEL = pd.Series(
    {
        "model": "kt#ThermalModel_KLM_T01_stator_temp_pinn",
        "farm_code": "KLM",
        "turbine_id": "T01",
        "component_key": "generator",
        "model_class": "ThermalModel",
        "target": "stator_temp",
        "variant": "pinn",
    }
)


def score() -> pd.Series:
    """Hourly score: calibration year 0..9 repeating, then a 3 h spike of 20
    in the test period (and one in calibration that must not count)."""
    calibration = pd.date_range("2021-01-01", "2021-12-31", freq="h", tz="UTC")
    test = pd.date_range("2022-03-01", periods=48, freq="h", tz="UTC")
    values = pd.Series(range(len(calibration)), index=calibration) % 10.0
    values.iloc[100:103] = 20.0
    later = pd.Series(0.0, index=test)
    later.iloc[10:13] = 20.0
    return pd.concat([values, later])


def test_threshold_alerts_keep_test_period_alerts_only() -> None:
    alerts = threshold_alerts(
        {"ThermalModel_KLM_T01_stator_temp_pinn": score()},
        pd.DataFrame([MODEL]),
        ALERTS,
        (TRAIN_END, CAL_END),
        quantile=0.999,
    )

    assert alerts[["turbine_id", "target", "variant", "start"]].values.tolist() == [
        ["T01", "stator_temp", "pinn", pd.Timestamp("2022-03-01 10:00", tz="UTC")]
    ]


def test_threshold_alerts_skip_models_without_a_score() -> None:
    alerts = threshold_alerts(
        {}, pd.DataFrame([MODEL]), ALERTS, (TRAIN_END, CAL_END), quantile=0.99
    )

    assert alerts.empty


def test_lift_is_mean_detection_above_chance_over_quantiles() -> None:
    curve = pd.DataFrame(
        {
            "detection_rate": [0.5, 0.3],
            "chance_detection_rate": [0.4, 0.3],
        }
    )

    assert lift(curve) == pytest.approx(0.05)
