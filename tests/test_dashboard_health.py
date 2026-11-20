"""Tests for the dashboard's component health index."""

import numpy as np
import pandas as pd
import pytest

from src.dashboard.health import (
    ALERT,
    OK,
    WATCH,
    compact,
    component_level,
    farm_health,
    health_index,
    health_level,
    health_table,
    score_at,
)

TS = pd.Timestamp("2022-01-01 12:00", tz="UTC")


def score(values: list[float]) -> pd.Series:
    index = pd.date_range("2022-01-01 10:00", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index)


def models() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["a", "b", "c"],
            "turbine_id": ["T01", "T01", "T02"],
            "target": ["gear_oil_temp"] * 3,
            "variant": ["physics", "hybrid", "physics"],
        }
    )


def test_score_at_takes_the_latest_value_not_after_ts() -> None:
    series = score([1.0, 2.0, 3.0, 4.0])

    assert score_at(series, TS) == 3.0
    assert np.isnan(score_at(series, TS - pd.Timedelta(days=1)))


def test_health_index_needs_a_positive_threshold() -> None:
    assert health_index(1.5, 2.0) == 0.75
    assert np.isnan(health_index(1.5, 0.0))


def test_health_level_bands() -> None:
    levels = [health_level(i, 0.8) for i in (0.5, 0.8, 1.0, np.nan)]

    assert levels == [OK, WATCH, ALERT, OK]


def test_health_table_puts_the_worst_model_first_and_keeps_missing_scores() -> None:
    scores = {"a": score([0.0, 0.0, 1.0]), "b": score([0.0, 0.0, 3.0])}
    thresholds = {"a": 2.0, "b": 2.0, "c": 2.0}

    table = health_table(models(), scores, thresholds, TS, watch_ratio=0.8)

    assert table["model_id"].tolist() == ["b", "a", "c"]
    assert table["level"].tolist() == [ALERT, OK, OK]
    assert table.iloc[0]["health_index"] == 1.5
    assert np.isnan(table.iloc[2]["score"])


def test_component_level_is_the_worst_model() -> None:
    table = pd.DataFrame({"health_index": [0.2, 0.9, np.nan]})

    assert component_level(table, 0.8) == (0.9, WATCH)


def test_farm_health_gives_each_turbine_its_worst_model() -> None:
    scores = {"a": score([0.0, 0.0, 1.0]), "b": score([0.0, 0.0, 3.0])}
    thresholds = {"a": 2.0, "b": 2.0, "c": 2.0}

    overview = farm_health(models(), scores, thresholds, TS, watch_ratio=0.8)

    assert overview["turbine_id"].tolist() == ["T01", "T02"]
    assert overview["health_index"].iloc[0] == 1.5
    assert overview["level"].tolist() == [ALERT, OK]


def test_compact_keeps_the_last_value_per_period() -> None:
    index = pd.date_range("2022-01-01", periods=12, freq="10min", tz="UTC")
    series = pd.Series(np.arange(12.0), index=index)

    hourly = compact(series, "1h")

    assert hourly.tolist() == pytest.approx([5.0, 11.0])
