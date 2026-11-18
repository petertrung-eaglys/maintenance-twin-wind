"""Evaluate alerts against the fault events (spec §7.10).

Test period: after calibration_end. Fault events of one turbine and
component less than `merge_gap_days` apart form one episode. An episode is
detected by a model when one of its alerts on the same turbine and
component starts at most `lookback_days` before the episode starts; the
lead time is that of the earliest such alert. An alert that precedes no
episode is a false alarm. Chance detection is what alerts at the same rate,
placed at random, would detect: 1 - exp(-rate * lookback).

Matched alerts are linked to their nearest episode's first event with
kt:precedes and kt:leadTimeHours; the summary goes to
models/evaluation.json and is printed per farm, component, target and
variant. The drift KPI (each later year's mean problem score minus the
calibration year's, from models/alerts.json) reports slow changes that
event-timed matching cannot credit.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdflib import URIRef

from src.config import farm_names, load_config
from src.graph.graph_utils import (
    add_precedes,
    clear_precedes,
    get_alerts,
    get_fault_events,
    get_models,
    load_graph,
    save_graph,
)
from src.models.modelling import split_times, turbine_paths

logger = logging.getLogger(__name__)

GROUP = ["farm_code", "component_key", "model_class", "target", "variant"]
LOCATION = ["farm_code", "turbine_id", "component_key"]
IDENTITY = ["farm_code", "turbine_id", "model_class", "target", "variant"]
DAYS_PER_YEAR = 365.25
MATCH_COLUMNS = ["alert", "event", "turbine_id", "component_key", "lead_hours"]


def location(frame: pd.DataFrame) -> list[str]:
    """The location columns a frame has (farm_code is optional)."""
    return [c for c in LOCATION if c in frame.columns]


def merge_episodes(events: pd.DataFrame, gap: pd.Timedelta) -> pd.DataFrame:
    """One row per episode: events of one turbine and component, each less
    than `gap` after the previous one. Keeps the first event's id and
    start, and the number of events."""
    keys = location(events)
    ordered = events.sort_values([*keys, "start"], ignore_index=True)
    previous = ordered.groupby(keys)["start"].shift()
    episode = (previous.isna() | (ordered["start"] - previous >= gap)).cumsum()
    grouped = ordered.groupby(episode)
    return (
        grouped[[*keys, "event", "start"]]
        .first()
        .assign(n_events=grouped.size())
        .reset_index(drop=True)
    )


def match_alerts(
    alerts: pd.DataFrame, episodes: pd.DataFrame, lookback: pd.Timedelta
) -> pd.DataFrame:
    """Every (alert, episode) pair of one location where the alert starts
    at most `lookback` before the episode; lead time in hours."""
    if alerts.empty or episodes.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    keys = [c for c in location(alerts) if c in episodes.columns]
    pairs = alerts[[*keys, "alert", "start"]].merge(
        episodes[[*keys, "event", "start"]], on=keys, suffixes=("_alert", "")
    )
    lead = pairs["start"] - pairs["start_alert"]
    within = (lead >= pd.Timedelta(0)) & (lead <= lookback)
    return (
        pairs[within]
        .assign(lead_hours=lead[within] / pd.Timedelta(hours=1))
        .sort_values(["alert", "lead_hours"], ignore_index=True)[MATCH_COLUMNS]
    )


def nearest_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """For each alert, only the episode it is closest to."""
    return (
        matches.sort_values("lead_hours")
        .drop_duplicates("alert")
        .sort_values("alert", ignore_index=True)
    )


def chance_detection(
    alerts_per_turbine: np.ndarray, test_days: float | np.ndarray, lookback_days: float
) -> np.ndarray:
    """Probability that alerts at this rate, placed at random, start within
    the lookback window before an event."""
    rate = np.asarray(alerts_per_turbine, dtype=float) / test_days
    return 1 - np.exp(-rate * lookback_days)


def summarise(
    alerts: pd.DataFrame,
    episodes: pd.DataFrame,
    turbine_days: pd.Series,
    lookback: pd.Timedelta,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Detections, lead times and false alarms per group (farm, component,
    model class, target, variant). `turbine_days`: test days per (farm,
    turbine)."""
    records = []
    for group in groups.itertuples(index=False):
        farm, component = group.farm_code, group.component_key
        mine = alerts
        for key, value in zip(GROUP, group, strict=True):
            mine = mine[mine[key] == value]
        reach = episodes[
            (episodes["farm_code"] == farm) & (episodes["component_key"] == component)
        ]
        pairs = match_alerts(mine, reach, lookback)
        lead = pairs.groupby("event")["lead_hours"].max()
        days = turbine_days.loc[farm]
        per_turbine = (
            mine.groupby("turbine_id").size().reindex(days.index, fill_value=0)
        )
        chance = chance_detection(
            per_turbine.loc[reach["turbine_id"]].to_numpy(),
            days.loc[reach["turbine_id"]].to_numpy(),
            lookback / pd.Timedelta(days=1),
        )
        false_alarms = int((~mine["alert"].isin(pairs["alert"])).sum())
        turbine_years = days.sum() / DAYS_PER_YEAR
        records.append(
            dict(zip(GROUP, group, strict=True))
            | {
                "episodes": len(reach),
                "detected": int(pairs["event"].nunique()),
                "detection_rate": pairs["event"].nunique() / len(reach)
                if len(reach)
                else np.nan,
                "chance_detection_rate": float(chance.mean()) if len(reach) else np.nan,
                "lead_hours_median": float(lead.median()) if len(lead) else np.nan,
                "lead_hours_p25": float(lead.quantile(0.25)) if len(lead) else np.nan,
                "lead_hours_p75": float(lead.quantile(0.75)) if len(lead) else np.nan,
                "alerts": len(mine),
                "false_alarms": false_alarms,
                "false_alarms_per_turbine_year": false_alarms / turbine_years,
            }
        )
    return pd.DataFrame(records)


def drift_table(
    records: Mapping[str, dict[str, Any]], calibration_year: int
) -> pd.DataFrame:
    """Drift KPI per model: each later year's mean problem score minus the
    calibration year's (positive = worse: hotter, or less power), and the
    largest of them. Models without a calibration-year score are left out."""
    rows = []
    for model_id, record in records.items():
        yearly = record.get("yearly_mean_score") or {}
        base = yearly.get(str(calibration_year))
        later = {
            year: round(value - base, 3)
            for year, value in yearly.items()
            if base is not None and int(year) > calibration_year
        }
        if later:
            identity = {key: record[key] for key in IDENTITY}
            rows.append(
                {"model": model_id, **identity, "calibration_mean": base, **later}
                | {"max_drift": max(later.values())}
            )
    return pd.DataFrame(rows)


def days_after(paths: dict[str, Path], test_after: pd.Timestamp) -> pd.Series:
    """Days of data after `test_after` per turbine (from the last
    timestamp; only that column is read)."""
    last = {
        tid: pq.read_table(path, columns=["timestamp"])["timestamp"].to_pandas().max()
        for tid, path in paths.items()
    }
    return pd.Series(
        {
            tid: max((t - test_after) / pd.Timedelta(days=1), 0.0)
            for tid, t in last.items()
        }
    )


def main() -> None:
    """Match test-period alerts to fault episodes, link and summarise."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    settings = config["evaluation"]
    lookback = pd.Timedelta(days=settings["lookback_days"])
    gap = pd.Timedelta(days=settings["merge_gap_days"])
    _, calibration_end = split_times(config["train_test_split"])
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    codes = [config["farms"][farm]["code"] for farm in farm_names(config)]
    events = get_fault_events(g)
    episodes = merge_episodes(events[events["start"] > calibration_end], gap)
    alerts = get_alerts(g)
    alerts = alerts[alerts["start"] > calibration_end]
    days = pd.concat(
        {code: days_after(turbine_paths(g, code), calibration_end) for code in codes}
    )
    groups = get_models(g)[GROUP].drop_duplicates().sort_values(GROUP)
    summary = summarise(alerts, episodes, days, lookback, groups)
    clear_precedes(g)
    links = nearest_matches(match_alerts(alerts, episodes, lookback))
    for link in links.itertuples():
        add_precedes(g, URIRef(link.alert), URIRef(link.event), link.lead_hours)
    save_graph(g, ontology)
    records = json.loads((Path(config["paths"]["models"]) / "alerts.json").read_text())
    drift = drift_table(records, calibration_end.year)
    write_evaluation(config, summary, episodes, drift)
    print_summary(summary)
    print_drift(drift)
    logger.info("Linked %d alerts to fault events in %s", len(links), ontology)


def write_evaluation(
    config: dict[str, Any],
    summary: pd.DataFrame,
    episodes: pd.DataFrame,
    drift: pd.DataFrame,
) -> None:
    """models/evaluation.json: settings, episode counts, the summary and
    the drift table."""
    out = Path(config["paths"]["models"]) / "evaluation.json"
    counts = episodes.groupby(["farm_code", "component_key"]).agg(
        episodes=("event", "size"), events=("n_events", "sum")
    )
    payload = {
        "settings": config["evaluation"]
        | {"alerts": config["alerts"], "split": config["train_test_split"]},
        "episodes": counts.reset_index().to_dict("records"),
        "summary": json.loads(summary.to_json(orient="records")),
        "drift": json.loads(drift.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=2))


def print_summary(summary: pd.DataFrame) -> None:
    """Acceptance table per farm, component, target and variant."""
    table = summary.assign(
        detected=summary["detected"].astype(str)
        + "/"
        + summary["episodes"].astype(str),
        rate=summary["detection_rate"].round(2),
        chance=summary["chance_detection_rate"].round(2),
        lead_days=(summary["lead_hours_median"] / 24).round(1),
        fa_per_ty=summary["false_alarms_per_turbine_year"].round(1),
    )[[*GROUP, "detected", "rate", "chance", "lead_days", "alerts", "fa_per_ty"]]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.drop(columns="model_class").to_string(index=False))


def print_drift(drift: pd.DataFrame, top: int = 10) -> None:
    """The largest drifts of the thermal (degC) and power (kW) models."""
    for model_class, unit in (("ThermalModel", "degC"), ("PowerModel", "kW")):
        rows = drift[drift["model_class"] == model_class].nlargest(top, "max_drift")
        years = sorted(c for c in drift.columns if c.isdigit())
        print(f"\nLargest drift vs calibration year, {model_class} ({unit}):")
        print(rows[[*IDENTITY[:2], "target", "variant", *years]].to_string(index=False))


if __name__ == "__main__":
    main()
