"""Shared figure style and output helpers for the report."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no display

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def apply_style(colors: dict[str, Any]) -> None:
    """Recessive axes and grid, text in ink colours, on the chart surface."""
    plt.rcParams.update(
        {
            "figure.facecolor": colors["surface"],
            "axes.facecolor": colors["surface"],
            "savefig.facecolor": colors["surface"],
            "axes.edgecolor": colors["grid"],
            "axes.labelcolor": colors["muted"],
            "axes.titlecolor": colors["text"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": colors["grid"],
            "grid.linewidth": 0.8,
            "text.color": colors["text"],
            "xtick.color": colors["muted"],
            "ytick.color": colors["muted"],
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def sequential_map(colors: dict[str, Any]) -> LinearSegmentedColormap:
    """One-hue light-to-dark colour map for magnitudes (0 = lightest)."""
    return LinearSegmentedColormap.from_list("sequential", colors["sequential"])


def save(fig: Figure, path: Path, dpi: int) -> Path:
    """Write a figure as PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    """A DataFrame as a GitHub Markdown table (index as the first columns),
    numbers rounded to `digits`."""
    flat = frame.reset_index()
    cells = [
        [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row]
        for row in flat.itertuples(index=False)
    ]
    header = [str(c) for c in flat.columns]
    lines = [header, ["---"] * len(header), *cells]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)
