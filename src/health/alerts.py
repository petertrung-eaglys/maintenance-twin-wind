"""Residual-based component health alerts (spec §7.10).

For every model with a prediction column, the problem score of
scores.py (fleet-relative, temperatures under load only; larger = worse)
is compared with a threshold: a high quantile of the score in the
calibration year, which no model was fitted on. Exceedances longer than a
minimum duration become alerts, and a model's alerts less than
`merge_gap_days` apart are merged into one; alerts are raised after
train_end. Yaw models alert when an identified rolling-window offset
drifts from the static offset. Drift alerts flag months in a row whose
mean score stays above the calibration year's by a margin.

Alerts go into the graph (kt:Alert), replacing each model's previous
alerts. models/alerts.json holds per model its threshold, alert count and
yearly mean score (the drift KPI used by evaluate.py);
models/drift_alerts.json the drift alerts.
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
from src.health.scores import farm_scores
from src.models.modelling import split_times, turbine_paths

logger = logging.getLogger(__name__)

PERIOD_COLUMNS = ["start", "end", "peak"]
IDENTITY = ["farm_code", "turbine_id", "model_class", "target", "variant"]
MIN_MONTH_DAYS, MAX_MONTH_DAYS = 28, 31


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
    score: pd.Series, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merged alert periods (after train_end) of one model's score, with
    its threshold and yearly mean score."""
    settings = config["alerts"]
    train_end, calibration_end = split_times(config["train_test_split"])
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


def drift_periods(
    score: pd.Series,
    train_end: pd.Timestamp,
    calibration_end: pd.Timestamp,
    margin: float,
    months: int,
) -> pd.DataFrame:
    """Runs of at least `months` consecutive months (after train_end) whose
    mean score exceeds the calibration year's mean by more than `margin`:
    start of the first month, end of the last, peak = largest excess."""
    monthly = score.resample("MS").mean().dropna()
    calibration = monthly[
        (monthly.index > train_end) & (monthly.index <= calibration_end)
    ]
    excess = monthly[monthly.index > train_end] - calibration.mean()
    periods = alert_periods(
        excess,
        margin,
        pd.Timedelta(days=MAX_MONTH_DAYS),
        pd.Timedelta(days=MIN_MONTH_DAYS * (months - 1)),
    )
    return periods.assign(end=periods["end"] + pd.offsets.MonthBegin(1))


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


def farm_alerts(
    g: Graph,
    config: dict[str, Any],
    models: pd.DataFrame,
    scores: dict[str, pd.Series],
    windows: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Alerts of every model of one farm, written to the graph; returns a
    record per model and the drift alerts."""
    drift = config["alerts"]["drift"]
    train_end, calibration_end = split_times(config["train_test_split"])
    records, drifts = {}, []
    for model in models.itertuples():
        model_id = model.model.split("#")[-1]
        identity = {key: getattr(model, key) for key in IDENTITY}
        if model.model_class == "YawModel":
            mine = windows[windows["turbine_id"] == model.turbine_id]
            periods, info = yaw_alerts(mine, model, config)
        else:
            score = scores.get(model_id, pd.Series(dtype=float))
            periods, info = residual_alerts(score, config)
            sustained = drift_periods(
                score,
                train_end,
                calibration_end,
                drift["margin"][model.model_class],
                drift["months"],
            )
            drifts += [
                {"model": model_id, **identity, "component": model.component_key}
                | {"start": p.start, "end": p.end, "peak": p.peak}
                for p in sustained.itertuples()
            ]
        component = component_iri(
            model.farm_code, model.turbine_id, model.component_key
        )
        count = write_alerts(g, URIRef(model.model), component, periods)
        records[model_id] = identity | info | {"alerts": count}
    return records, drifts


def main() -> None:
    """Raise alerts for every model of every turbine."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    records, drifts = {}, []
    for farm in farm_names(config):
        code = config["farms"][farm]["code"]
        models = get_models(g, code)
        windows = pd.read_parquet(
            farm_processed_dir(config, farm) / "yaw_offsets.parquet"
        )
        scores = farm_scores(
            config, models, turbine_paths(g, code), config["alerts"]["fleet_relative"]
        )
        farm_records, farm_drifts = farm_alerts(g, config, models, scores, windows)
        records |= farm_records
        drifts += farm_drifts
        logger.info("%s: alerts for %d models", code, len(farm_records))
    save_graph(g, ontology)
    models_dir = Path(config["paths"]["models"])
    (models_dir / "alerts.json").write_text(
        json.dumps(records, indent=2, sort_keys=True)
    )
    drift_frame = pd.DataFrame(drifts)
    (models_dir / "drift_alerts.json").write_text(
        drift_frame.to_json(orient="records", date_format="iso", indent=2)
    )
    summary = (
        pd.DataFrame(records)
        .T.astype({"alerts": int})
        .groupby(["farm_code", "model_class", "variant"])["alerts"]
        .agg(["sum", "mean"])
    )
    print("Alerts after train_end (sum over turbines and targets, mean per model):")
    print(summary.round(1).to_string())
    print(f"\nDrift alerts (sustained {config['alerts']['drift']['months']}+ months):")
    if not drift_frame.empty:
        columns = [
            "farm_code",
            "turbine_id",
            "target",
            "variant",
            "start",
            "end",
            "peak",
        ]
        shown = drift_frame[columns].assign(
            start=drift_frame["start"].dt.strftime("%Y-%m"),
            end=drift_frame["end"].dt.strftime("%Y-%m"),
            peak=drift_frame["peak"].round(2),
        )
        print(shown.to_string(index=False))


if __name__ == "__main__":
    main()
