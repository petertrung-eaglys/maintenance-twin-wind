"""Method of bins (IEC 61400-12-1 style): mean response per x bin."""

import numpy as np
import pandas as pd


def binned_means(
    x: pd.Series, y: pd.Series, width: float, min_count: int
) -> pd.DataFrame:
    """Mean x and mean y per bin of `width`, for bins with at least
    `min_count` rows, sorted by x. Columns: x, y, count."""
    bins = np.floor(x / width)
    grouped = pd.DataFrame({"x": x, "y": y}).groupby(bins)
    curve = grouped.mean().assign(count=grouped.size())
    return curve[curve["count"] >= min_count].sort_values("x").reset_index(drop=True)


def interpolate(curve: pd.DataFrame, x: pd.Series | np.ndarray) -> np.ndarray:
    """Linear interpolation of the curve's y at x (held constant past the ends)."""
    return np.interp(np.asarray(x, dtype=float), curve["x"], curve["y"])
