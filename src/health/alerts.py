"""Residual-based component health alerts (spec §7.10).

For every model with a prediction column: residual = measured - expected
on normal rows, smoothed with a rolling time-window mean, turned into a
problem score (larger = worse: low power, high temperature, either side
for pitch). The threshold is a high quantile of the score in the
calibration year, which no model was fitted on. Exceedances longer than a
minimum duration become alerts, and a model's alerts less than
`merge_gap_days` apart are merged into one; alerts are raised after
train_end. Yaw models alert when an identified rolling-window offset
drifts from the static offset.

Alerts go into the graph (kt:Alert), replacing each model's previous
alerts. models/alerts.json holds per model its threshold, alert count and
yearly mean score (the drift KPI used by evaluate.py).
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph, URIRef

from src.config import farm_names, farm_processed_dir, load_config
from src.graph.graph_utils import (
    add_alert,
    component_iri,
    get_models,
    load_graph,
    remove_alerts,
    save_graph,
)
from src.models.modelling import prediction_column, split_times, turbine_paths

logger = logging.getLogger(__name__)

PERIOD_COLUMNS = ["start", "end", "peak"]
IDENTITY = ["farm_code", "turbine_id", "model_class", "target", "variant"]


def smoothed_residual(
    frame: pd.DataFrame, target: str, column: str, window: str, min_samples: int
) -> pd.Series:
    """Rolling time-window mean of measured - expected over normal rows
    with both values, indexed by timestamp (windows with fewer than
    `min_samples` rows dropped)."""
    rows = frame["is_normal"] & frame[target].notna() & frame[column].notna()
    residual = pd.Series(
        (frame.loc[rows, target] - frame.loc[rows, column]).to_numpy(),
        index=pd.DatetimeIndex(frame.loc[rows, "timestamp"]),
    )
    return residual.rolling(window, min_periods=min_samples).mean().dropna()


def problem_score(residual: pd.Series, direction: str) -> pd.Series:
    """Residual turned so that larger is worse: `low` (e.g. power below
    expected), `high` (temperature above) or `both` (either side)."""
    if direction == "low":
        return -residual
    if direction == "high":
        return residual
    if direction == "both":
        return residual.abs()
    raise ValueError(f"unknown alert direction {direction!r}")


def calibration_threshold(
    score: pd.Series,
    train_end: pd.Timestamp,
    calibration_end: pd.Timestamp,
    quantile: float,
) -> float:
    """Quantile of the score in the calibration period (NaN if empty)."""
    inside = (score.index > train_end) & (score.index <= calibration_end)
    return float(score[inside].quantile(quantile))


def alert_periods(
    score: pd.Series,
    threshold: float,
    max_gap: pd.Timedelta,
    min_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Runs of score > threshold (split where rows are more than max_gap
    apart) lasting at least min_duration: start, end and peak score."""
    flag = pd.Series(score.to_numpy() > threshold, index=score.index)
    times = pd.Series(score.index, index=score.index)
    new_run = (flag != flag.shift()) | (times.diff() > max_gap)
    runs = pd.DataFrame({"time": times, "score": score, "run": new_run.cumsum()})[flag]
    periods = runs.groupby("run").agg(
        start=("time", "min"), end=("time", "max"), peak=("score", "max")
    )
    long_enough = periods["end"] - periods["start"] >= min_duration
    return periods[long_enough].reset_index(drop=True)[PERIOD_COLUMNS]


def merge_periods(periods: pd.DataFrame, gap: pd.Timedelta) -> pd.DataFrame:
    """Join alert periods that start less than `gap` after the previous one
    ended (one alert per problem, not one per threshold crossing)."""
    if periods.empty:
        return periods
    ordered = periods.sort_values("start", ignore_index=True)
    previous_end = ordered["end"].shift()
    group = (previous_end.isna() | (ordered["start"] - previous_end >= gap)).cumsum()
    merged = ordered.groupby(group).agg(
        start=("start", "min"), end=("end", "max"), peak=("peak", "max")
    )
    return merged.reset_index(drop=True)[PERIOD_COLUMNS]


def yearly_means(score: pd.Series) -> dict[int, float]:
    """Mean score per calendar year (the drift KPI)."""
    means = score.groupby(score.index.year).mean()
    return {int(year): float(value) for year, value in means.items()}


def yaw_alert_periods(
    windows: pd.DataFrame, static_offset: float, drift_deg: float, step_days: int
) -> pd.DataFrame:
    """Runs of consecutive identified windows whose offset is more than
    drift_deg from the static offset; from the first window's start to
    the last window's end, peak = largest drift."""
    known = windows[windows["identified"]]
    starts = pd.DatetimeIndex(known["window_start"])
    drift = pd.Series(
        (known["offset_deg"] - static_offset).abs().to_numpy(), index=starts
    )
    ends = pd.Series(known["window_end"].to_numpy(), index=starts)
    periods = alert_periods(
        drift, drift_deg, pd.Timedelta(days=step_days), pd.Timedelta(0)
    )
    return periods.assign(end=ends.loc[periods["end"]].to_numpy())


def write_alerts(
    g: Graph, model: URIRef, component: URIRef, periods: pd.DataFrame
) -> int:
    """Replace the model's alerts in the graph with these periods."""
    remove_alerts(g, model)
    for number, period in enumerate(periods.itertuples(), start=1):
        add_alert(g, model, component, number, period.start, period.end)
    return len(periods)


def residual_alerts(
    frame: pd.DataFrame, model: pd.Series, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merged alert periods (after train_end) of one residual model, with
    its threshold and yearly mean score."""
    settings = config["alerts"]
    train_end, calibration_end = split_times(config["train_test_split"])
    column = prediction_column(model.target, model.variant)
    smoothed = smoothed_residual(
        frame, model.target, column, settings["window"], settings["min_samples"]
    )
    score = problem_score(smoothed, settings["direction"][model.model_class])
    threshold = calibration_threshold(
        score, train_end, calibration_end, settings["quantile"]
    )
    periods = alert_periods(
        score[score.index > train_end],
        threshold,
        pd.Timedelta(hours=settings["max_gap_hours"]),
        pd.Timedelta(hours=settings["min_duration_hours"]),
    )
    merged = merge_periods(periods, pd.Timedelta(days=settings["merge_gap_days"]))
    return merged, {"threshold": threshold, "yearly_mean_score": yearly_means(score)}


def yaw_alerts(
    windows: pd.DataFrame, model: pd.Series, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drift alerts (windows starting after train_end) of an identified yaw
    model; none when its static offset was not identified."""
    fit = json.loads(Path(model.model_file).read_text())
    if not fit["identified"]:
        return pd.DataFrame(columns=PERIOD_COLUMNS), {"threshold": float("nan")}
    train_end, _ = split_times(config["train_test_split"])
    later = windows[windows["window_start"] > train_end]
    drift = config["alerts"]["yaw_drift_deg"]
    periods = yaw_alert_periods(
        later, fit["offset_deg"], drift, config["yaw"]["step_days"]
    )
    gap = pd.Timedelta(days=config["alerts"]["merge_gap_days"])
    return merge_periods(periods, gap), {"threshold": drift}


def turbine_alerts(
    g: Graph,
    config: dict[str, Any],
    models: pd.DataFrame,
    frame: pd.DataFrame,
    windows: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Alerts of every model of one turbine; returns a record per model."""
    records = {}
    for model in models.itertuples():
        if model.model_class == "YawModel":
            periods, info = yaw_alerts(windows, model, config)
        else:
            periods, info = residual_alerts(frame, model, config)
        component = component_iri(
            model.farm_code, model.turbine_id, model.component_key
        )
        count = write_alerts(g, URIRef(model.model), component, periods)
        identity = {key: getattr(model, key) for key in IDENTITY}
        records[model.model.split("#")[-1]] = identity | info | {"alerts": count}
    return records


def needed_columns(models: pd.DataFrame) -> list[str]:
    """Parquet columns that the residual models of one turbine need."""
    residual = models[models["model_class"] != "YawModel"]
    columns = ["timestamp", "is_normal", *residual["target"]]
    columns += [prediction_column(m.target, m.variant) for m in residual.itertuples()]
    return list(dict.fromkeys(columns))


def main() -> None:
    """Raise alerts for every model of every turbine."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    records = {}
    for farm in farm_names(config):
        code = config["farms"][farm]["code"]
        models = get_models(g, code)
        offsets = pd.read_parquet(
            farm_processed_dir(config, farm) / "yaw_offsets.parquet"
        )
        for tid, path in turbine_paths(g, code).items():
            mine = models[models["turbine_id"] == tid]
            frame = pd.read_parquet(path, columns=needed_columns(mine))
            windows = offsets[offsets["turbine_id"] == tid]
            records |= turbine_alerts(g, config, mine, frame, windows)
            logger.info("%s %s: alerts for %d models", code, tid, len(mine))
    save_graph(g, ontology)
    out = Path(config["paths"]["models"]) / "alerts.json"
    out.write_text(json.dumps(records, indent=2, sort_keys=True))
    summary = (
        pd.DataFrame(records)
        .T.astype({"alerts": int})
        .groupby(["farm_code", "model_class", "variant"])["alerts"]
        .agg(["sum", "mean"])
    )
    print("Alerts after train_end (sum over turbines and targets, mean per model):")
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
