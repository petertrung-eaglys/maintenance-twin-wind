"""Component health index: each model's problem score at the replay time
divided by its alert threshold (both as in `make alerts`)."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

OK = "ok"
WATCH = "watch"
ALERT = "alert"


def score_at(score: pd.Series, ts: pd.Timestamp) -> float:
    """The latest score at or before ts (NaN if there is none)."""
    position = score.index.searchsorted(ts, side="right")
    return float(score.iloc[position - 1]) if position > 0 else np.nan


def health_index(score: float, threshold: float) -> float:
    """Score as a share of the alert threshold (NaN without a positive
    threshold): 1 means at the threshold."""
    return score / threshold if threshold > 0 else np.nan


def health_level(index: float, watch_ratio: float) -> str:
    """ok, watch (from watch_ratio) or alert (from 1); NaN counts as ok."""
    if index >= 1:
        return ALERT
    return WATCH if index >= watch_ratio else OK


def health_table(
    models: pd.DataFrame,
    scores: Mapping[str, pd.Series],
    thresholds: Mapping[str, float],
    ts: pd.Timestamp,
    watch_ratio: float,
) -> pd.DataFrame:
    """One row per model (rows: model_id, target, variant): score at ts,
    threshold, health index and level, worst first."""
    rows = []
    for model in models.itertuples():
        score = scores.get(model.model_id, pd.Series(dtype=float))
        value = score_at(score, ts)
        threshold = thresholds.get(model.model_id, np.nan)
        index = health_index(value, threshold)
        rows.append(
            {
                "model_id": model.model_id,
                "target": model.target,
                "variant": model.variant,
                "score": value,
                "threshold": threshold,
                "health_index": index,
                "level": health_level(index, watch_ratio),
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "target",
            "variant",
            "score",
            "threshold",
            "health_index",
            "level",
        ],
    )
    return table.sort_values("health_index", ascending=False, na_position="last")


def component_level(table: pd.DataFrame, watch_ratio: float) -> tuple[float, str]:
    """The component's index (its worst model) and level."""
    worst = table["health_index"].max()
    return worst, health_level(worst, watch_ratio)


def farm_health(
    models: pd.DataFrame,
    scores: Mapping[str, pd.Series],
    thresholds: Mapping[str, float],
    ts: pd.Timestamp,
    watch_ratio: float,
) -> pd.DataFrame:
    """Component index and level per turbine (its worst model) at ts;
    `models` also needs turbine_id."""
    rows = []
    for turbine_id, group in models.groupby("turbine_id"):
        table = health_table(group, scores, thresholds, ts, watch_ratio)
        index, level = component_level(table, watch_ratio)
        rows.append({"turbine_id": turbine_id, "health_index": index, "level": level})
    return pd.DataFrame(rows, columns=["turbine_id", "health_index", "level"])


def compact(score: pd.Series, rule: str) -> pd.Series:
    """A score kept at a coarser resolution (last value per period), to
    bound the dashboard's memory."""
    return score.resample(rule).last().dropna()
