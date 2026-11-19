"""Wake-aware power (§7.13): how much of each power variant's alert time
falls while the turbine is waked, against how often it is waked at all."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.models.modelling import prediction_column

ALL_ROWS = "all test rows"
ROWS_PER_HOUR = 6  # 10-minute rows
HEADROOM = 1.3  # y-axis room above the bars for the legend


def comparable_rows(frame: pd.DataFrame) -> pd.Series:
    """Rows where both wake-model variants predict (a free turbine exists)
    and the free-stream turbine would produce power."""
    wake = frame[prediction_column("power", "wake")]
    return wake.notna() & frame[prediction_column("power", "freestream")].gt(0)


def waked_rows(frame: pd.DataFrame, power_ratio: float) -> np.ndarray:
    """Rows where the wake variant expects less than `power_ratio` of the
    free-stream power (the turbine is in a wake)."""
    wake = frame[prediction_column("power", "wake")].to_numpy()
    free = frame[prediction_column("power", "freestream")].to_numpy()
    return wake < power_ratio * free


def alert_time(
    frame: pd.DataFrame, alerts: pd.DataFrame, waked: np.ndarray
) -> list[dict[str, object]]:
    """Normal rows and waked rows inside each alert (one record per alert,
    `variant` from the alert), plus one `all test rows` record for the
    turbine; `frame` holds the turbine's normal test rows."""
    times = frame["timestamp"].to_numpy("datetime64[ns]")
    records = [{"variant": ALL_ROWS, "rows": len(frame), "waked": int(waked.sum())}]
    for alert in alerts.itertuples():
        inside = (times >= alert.start.to_datetime64()) & (
            times <= alert.end.to_datetime64()
        )
        records.append(
            {
                "variant": alert.variant,
                "rows": int(inside.sum()),
                "waked": int(waked[inside].sum()),
            }
        )
    return records


def waked_share_table(records: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """Alert hours, waked alert hours and waked share per farm and variant
    (`all test rows` first, then the variants in the given order)."""
    total = records.groupby(["farm_code", "variant"])[["rows", "waked"]].sum()
    table = pd.DataFrame(
        {
            "hours": total["rows"] / ROWS_PER_HOUR,
            "waked_hours": total["waked"] / ROWS_PER_HOUR,
            "waked_share": total["waked"] / total["rows"],
        }
    )
    order = [ALL_ROWS, *variants]
    return table.reindex(
        [(farm, v) for farm in table.index.unique("farm_code") for v in order],
    ).dropna()


def wake_alert_figure(
    table: pd.DataFrame, colors: Mapping[str, str], muted: str
) -> Figure:
    """Per farm: waked share of each variant's alert time, with the share of
    all test rows that are waked as a dashed line."""
    farms = table.index.unique("farm_code")
    fig, axes = plt.subplots(
        1,
        len(farms),
        figsize=(10, 3.4),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for ax, farm in zip(axes[0], farms, strict=True):
        rows = table.loc[farm]
        base = rows.loc[ALL_ROWS, "waked_share"]
        variants = rows.drop(index=ALL_ROWS)
        x = np.arange(len(variants))
        ax.bar(
            x, variants["waked_share"], 0.7, color=[colors[v] for v in variants.index]
        )
        ax.axhline(
            base,
            color=muted,
            linewidth=1,
            linestyle="--",
            label=f"waked share of all test rows ({base:.0%})",
        )
        ax.set_ylim(0, max(variants["waked_share"].max(), base) * HEADROOM)
        ax.legend(loc="upper right")
        ax.set_xticks(x, variants.index)
        ax.set_title(f"{farm}: share of power-alert time while waked", loc="left")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return fig
