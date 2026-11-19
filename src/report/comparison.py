"""Model comparison: test error per variant (in-site) and the cross-site
experiment (trained on the other farm)."""

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

IN_SITE_ALPHA = 0.35  # lighter bar: trained on the same farm


def signal_label(signal: str) -> str:
    """Short axis label of a signal: `gear_oil_temp` -> `gear oil`."""
    return signal.removesuffix("_temp").replace("_", " ")


def in_order(table: pd.DataFrame, order: Mapping[str, Any]) -> pd.DataFrame:
    """Rows of a variant-indexed table in the configured variant order."""
    return table.reindex([v for v in order if v in table.index])


def metric_frame(metrics: Mapping[str, dict[str, Any]]) -> pd.DataFrame:
    """metrics.json records as rows, with the model class from the model id."""
    return pd.DataFrame(list(metrics.values())).assign(
        model_class=[model_id.split("_")[0] for model_id in metrics]
    )


def variant_means(
    frame: pd.DataFrame, model_class: str, value: str, variants: list[str]
) -> pd.DataFrame:
    """Mean of `value` over turbines per target and farm (rows) and variant
    (columns, in the given order) for one model class."""
    table = frame[frame["model_class"] == model_class].pivot_table(
        index=["target", "farm_code"], columns="variant", values=value, aggfunc="mean"
    )
    return table[[v for v in variants if v in table.columns]]


def grouped_bars(
    ax: Axes, table: pd.DataFrame, colors: Mapping[str, str], alpha: float = 1.0
) -> None:
    """One bar group per row of `table`, one bar per column (variant)."""
    count = len(table.columns)
    width = 0.8 / count
    for i, variant in enumerate(table.columns):
        offset = (i - (count - 1) / 2) * width
        ax.bar(
            np.arange(len(table)) + offset,
            table[variant],
            width * 0.9,  # surface gap between neighbouring bars
            color=colors[variant],
            alpha=alpha,
            label=variant,
        )
    ax.set_xticks(range(len(table)), table.index)


def comparison_figure(
    power: pd.DataFrame,
    pitch: pd.DataFrame,
    thermal: pd.DataFrame,
    colors: Mapping[str, str],
) -> Figure:
    """Test error per variant: power nRMSE and pitch RMSE per farm, thermal
    RMSE per temperature (mean over both farms)."""
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.6), width_ratios=[2, 2, 7], constrained_layout=True
    )
    panels = [
        (power.droplevel("target"), "Power nRMSE (%)"),
        (pitch.droplevel("target"), "Pitch RMSE (°)"),
        (
            thermal.groupby(level="target", sort=False).mean().rename(signal_label),
            "Temperature RMSE (°C), both farms",
        ),
    ]
    for ax, (table, title) in zip(axes, panels, strict=True):
        grouped_bars(ax, table, colors)
        ax.set_title(title, loc="left")
        ax.set_xlabel("")
    axes[2].tick_params(axis="x", labelrotation=20)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels))
    return fig


def cross_site_panels(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mean in-site and cross-site RMSE per direction and variant, for
    power and for the mean over the temperatures."""
    direction = summary["train"] + " → " + summary["test"]
    frame = summary.assign(direction=direction)
    is_power = frame["signal"] == "power"
    return {
        group: rows.groupby(["direction", "variant"], sort=False)[
            ["rmse_in", "rmse_cross"]
        ].mean()
        for group, rows in (("power", frame[is_power]), ("thermal", frame[~is_power]))
    }


def cross_site_figure(
    panels: Mapping[str, pd.DataFrame], colors: Mapping[str, str], muted: str
) -> Figure:
    """Rows: direction; columns: power, temperatures. Per variant a light
    bar (trained on the same farm) next to a solid one (other farm)."""
    directions = panels["power"].index.unique("direction")
    fig, axes = plt.subplots(
        len(directions), 2, figsize=(10, 6), squeeze=False, constrained_layout=True
    )
    units = {"power": "Power: RMSE (kW)", "thermal": "Temperatures, mean: RMSE (°C)"}
    for row, direction in enumerate(directions):
        for col, (group, table) in enumerate(panels.items()):
            ax = axes[row, col]
            data = in_order(table.loc[direction], colors)
            x = np.arange(len(data))
            bar_colors = [colors[v] for v in data.index]
            ax.bar(
                x - 0.2, data["rmse_in"], 0.36, color=bar_colors, alpha=IN_SITE_ALPHA
            )
            ax.bar(x + 0.2, data["rmse_cross"], 0.36, color=bar_colors)
            ax.set_xticks(x, data.index)
            ax.set_title(f"{units[group]}, trained {direction}", loc="left")
    legend = [
        Patch(color=muted, alpha=IN_SITE_ALPHA, label="trained on the test farm"),
        Patch(color=muted, label="trained on the other farm"),
    ]
    fig.legend(handles=legend, loc="outside upper center", ncols=2)
    return fig


def cross_site_table(summary: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """`in → cross` RMSE per direction and signal (rows) and variant."""
    cell = summary.assign(
        cell=summary["rmse_in"].map("{:.2f}".format)
        + " → "
        + summary["rmse_cross"].map("{:.2f}".format)
    )
    table = cell.pivot_table(
        index=["train", "test", "signal"],
        columns="variant",
        values="cell",
        aggfunc="first",
    )
    return table[[v for v in variants if v in table.columns]].fillna("–")
