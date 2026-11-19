"""Fault case studies: a turbine's fleet-relative residual score per
thermal variant over time, with the real fault events of the component."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import date2num
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory
from rdflib import Graph

from src.graph.graph_utils import get_models
from src.health.scores import farm_scores

OTHER = "other"
EVENT_MARKERS = ["v", "o", "s", "D"]  # one per message type, then `other`
EVENT_ROW = 0.04  # event markers sit this far up the axes (axes fraction)


def case_scores(
    config: dict[str, Any],
    g: Graph,
    farm_code: str,
    paths: Mapping[str, Path],
    case: Mapping[str, str],
) -> dict[str, pd.Series]:
    """Problem score per variant of the case's thermal target and turbine
    (fleet-relative as configured, so every turbine of the farm is read)."""
    models = get_models(g, farm_code)
    target = models[
        (models["model_class"] == "ThermalModel") & (models["target"] == case["target"])
    ]
    scores = farm_scores(config, target, paths, config["alerts"]["fleet_relative"])
    mine = target[target["turbine_id"] == case["turbine_id"]]
    return {m.variant: scores[m.model.split("#")[-1]] for m in mine.itertuples()}


def case_events(
    events: pd.DataFrame, turbine_id: str, component: str, start: pd.Timestamp
) -> pd.DataFrame:
    """Labelled events of one turbine and component from `start` on."""
    return events[
        (events["turbine_id"] == turbine_id)
        & (events["component"] == component)
        & (events["start"] >= start)
    ]


def event_types(messages: pd.Series, max_types: int) -> pd.Series:
    """Each message, or `other` outside the `max_types` most frequent."""
    top = messages.value_counts().index[:max_types]
    return messages.where(messages.isin(top), OTHER)


def resampled_mean(score: pd.Series, rule: str) -> pd.Series:
    """Mean per `rule` period (e.g. `7D`) of a timestamp-indexed score, on
    naive UTC time."""
    return score.tz_convert(None).resample(rule).mean()


def case_figure(
    scores: Mapping[str, pd.Series],
    events: pd.DataFrame,
    calibration: tuple[pd.Timestamp, pd.Timestamp],
    style: Mapping[str, Any],
    title: str,
) -> Figure:
    """Resampled mean score per variant (calibration year shaded) with the
    events as markers along the bottom, one marker shape per type."""
    colors = style["colors"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axvspan(*calibration, color=colors["grid"], alpha=0.6, linewidth=0)
    ax.axhline(0, color=colors["muted"], linewidth=1)
    rule = style["case_resample"]
    for variant in colors["variants"]:
        if variant in scores:
            mean = resampled_mean(scores[variant], rule)
            ax.plot(mean.index, mean, color=colors["variants"][variant], label=variant)
    row = blended_transform_factory(ax.transData, ax.transAxes)
    times = events["start"].dt.tz_convert(None)
    types = event_types(events["message"], style["max_event_types"])
    labels = [*types.value_counts().index.drop(OTHER, errors="ignore"), OTHER]
    for label, marker in zip(labels, EVENT_MARKERS, strict=False):
        chosen = types == label
        if chosen.any():
            ax.scatter(
                times[chosen],
                [EVENT_ROW] * int(chosen.sum()),
                transform=row,
                marker=marker,
                s=30,
                facecolors="none",
                edgecolors=colors["text"],
                linewidths=1,
                label=f"{label} ({int(chosen.sum())})",
            )
    ax.annotate(
        "calibration year",
        (date2num(calibration[0]), 0.95),
        xycoords=row,
        xytext=(4, 0),
        textcoords="offset points",
        color=colors["muted"],
        va="top",
    )
    ax.set_ylabel(f"Fleet-relative residual, °C ({rule} mean)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return fig
