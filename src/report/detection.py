"""Event-timed detection against chance (from models/curves.json)."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def detection_points(
    curves: pd.DataFrame, scope: str, fleet: bool, min_episodes: int
) -> pd.DataFrame:
    """Curve rows of one scope and scoring with at least `min_episodes`
    test episodes (every threshold quantile)."""
    return curves[
        (curves["scope"] == scope)
        & (curves["fleet"] == fleet)
        & (curves["episodes"] >= min_episodes)
    ]


def mean_lift(points: pd.DataFrame) -> float:
    """Mean detection rate minus chance over all points."""
    return float((points["detection_rate"] - points["chance_detection_rate"]).mean())


def detection_figure(
    points: pd.DataFrame,
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    muted: str,
) -> Figure:
    """Detection rate vs chance per model group and threshold; on the
    diagonal a model detects no more than randomly placed alerts would."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=muted, linewidth=1, linestyle="--")
    ax.annotate(
        "chance", (0.82, 0.86), color=muted, rotation=45, ha="center", va="center"
    )
    for variant in colors:
        rows = points[points["variant"] == variant]
        ax.scatter(
            rows["chance_detection_rate"],
            rows["detection_rate"],
            s=36,
            color=colors[variant],
            marker=markers[variant],
            edgecolors=plt.rcParams["axes.facecolor"],  # ring on overlaps
            linewidths=1,
            label=variant,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Chance detection rate (alerts placed at random)")
    ax.set_ylabel("Detection rate (alert ≤ 30 days before an episode)")
    ax.set_title(f"Detection vs chance, mean lift {mean_lift(points):+.2f}", loc="left")
    ax.legend(loc="upper left")
    return fig
