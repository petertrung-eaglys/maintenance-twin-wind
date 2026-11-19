"""Data overview: normal-row coverage per turbine and month, and signal
availability per farm and year."""

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

MINUTES_PER_DAY = 24 * 60


def report_months(years: list[int]) -> pd.PeriodIndex:
    """Every month of the configured years."""
    return pd.period_range(f"{years[0]}-01", f"{years[-1]}-12", freq="M")


def monthly_normal_share(
    frame: pd.DataFrame, months: pd.PeriodIndex, interval_minutes: int
) -> pd.Series:
    """Share of each month's sampling slots that hold a normal row (0 for a
    month without data)."""
    normal = frame.loc[frame["is_normal"], "timestamp"].dt.tz_convert(None)
    counts = normal.dt.to_period("M").value_counts()
    slots = months.days_in_month * MINUTES_PER_DAY / interval_minutes
    return pd.Series(counts.reindex(months, fill_value=0).to_numpy() / slots, months)


def yearly_counts(frame: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Per year: rows (`rows`) and rows with a value, per signal."""
    years = frame["timestamp"].dt.year.rename("year")
    counts = frame[signals].notna().groupby(years).sum()
    return counts.assign(rows=years.value_counts())


def availability(counts: list[pd.DataFrame]) -> pd.DataFrame:
    """Share of rows with a value, signals x years, pooled over turbines."""
    total = pd.concat(counts).groupby(level=0).sum()
    return total.drop(columns="rows").div(total["rows"], axis=0).T


def year_ticks(months: pd.PeriodIndex) -> tuple[list[int], list[str]]:
    """Positions and labels of each January in a month axis."""
    starts = [i for i, m in enumerate(months) if m.month == 1]
    return starts, [str(months[i].year) for i in starts]


def coverage_figure(
    shares: Mapping[str, pd.DataFrame], cmap: Colormap, colors: dict[str, Any]
) -> Figure:
    """Heatmap per farm: turbines x months, share of normal rows."""
    heights = [len(s) for s in shares.values()]
    fig, axes = plt.subplots(
        len(shares),
        1,
        figsize=(9, 0.22 * sum(heights) + 1.6),
        height_ratios=heights,
        sharex=True,
        squeeze=False,
    )
    for ax, (farm, share) in zip(axes[:, 0], shares.items(), strict=True):
        image = ax.imshow(share.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(share)), share.index)
        ax.set_ylabel(farm)
        ax.grid(False)
        positions, labels = year_ticks(pd.PeriodIndex(share.columns))
        ax.set_xticks(positions, labels)
    axes[0, 0].set_title("Share of 10-minute slots with a normal row, per month")
    bar = fig.colorbar(image, ax=axes[:, 0], fraction=0.03, pad=0.02)
    bar.outline.set_edgecolor(colors["grid"])
    bar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return fig


def availability_figure(
    shares: Mapping[str, pd.DataFrame], cmap: Colormap, colors: dict[str, Any]
) -> Figure:
    """Heatmap per farm: signals x years, share of rows with a value."""
    fig, axes = plt.subplots(
        1, len(shares), figsize=(10, 4.2), sharey=True, squeeze=False
    )
    for ax, (farm, share) in zip(axes[0], shares.items(), strict=True):
        image = ax.imshow(share.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(share.shape[1]), [str(y) for y in share.columns])
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_yticks(range(len(share)), share.index)
        ax.set_title(farm)
        ax.grid(False)
    fig.suptitle("Share of rows with a value, per signal and year", fontweight="bold")
    bar = fig.colorbar(image, ax=axes[0], fraction=0.03, pad=0.02)
    bar.outline.set_edgecolor(colors["grid"])
    bar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return fig
