"""Replay helpers: the state of each turbine at the replay time, chart
windows and the play clock (adapted from the Kelmarsh repo's dashboard)."""

import pandas as pd

NORMAL = "normal"
NOT_NORMAL = "not operating normally"
ALERT = "alert"


def map_center(turbines: pd.DataFrame) -> dict[str, float]:
    """Mean turbine position; plotly's scatter_map does not centre itself."""
    return {"lat": float(turbines["lat"].mean()), "lon": float(turbines["long"].mean())}


def row_at(frame: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """The latest row at or before ts, or None if there is none (frame
    sorted by timestamp)."""
    position = frame["timestamp"].searchsorted(ts, side="right")
    return None if position == 0 else frame.iloc[position - 1]


def value_of(row: pd.Series | None, column: str) -> object:
    """A row's value, None without a row or the column."""
    return None if row is None or column not in row else row[column]


def active_alerts(alerts: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """The alerts whose period contains ts."""
    return alerts[(alerts["start"] <= ts) & (ts <= alerts["end"])]


def turbine_status(row: pd.Series | None, alerts_now: pd.DataFrame) -> str:
    """`alert` with an active alert, else normal or not operating normally."""
    if not alerts_now.empty:
        return ALERT
    return NORMAL if row is not None and bool(row["is_normal"]) else NOT_NORMAL


def farm_statuses(
    normal: pd.DataFrame, alerts: pd.DataFrame, ts: pd.Timestamp
) -> dict[str, str]:
    """Status per turbine at ts; `normal` holds timestamp, turbine_id and
    is_normal of every turbine of the farm."""
    now = active_alerts(alerts, ts)
    return {
        tid: turbine_status(row_at(rows, ts), now[now["turbine_id"] == tid])
        for tid, rows in normal.groupby("turbine_id")
    }


def window(frame: pd.DataFrame, end: pd.Timestamp, days: int) -> pd.DataFrame:
    """Rows in the last `days` days up to and including end."""
    start = end - pd.Timedelta(days=days)
    return frame[(frame["timestamp"] > start) & (frame["timestamp"] <= end)]


def alerts_overlapping(
    alerts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Alerts whose period overlaps [start, end]."""
    return alerts[(alerts["start"] <= end) & (alerts["end"] >= start)]


def sample_rows(frame: pd.DataFrame, max_rows: int, seed: int = 0) -> pd.DataFrame:
    """At most max_rows rows: a reproducible random sample when there are more."""
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).sort_index()


def clamp(ts: pd.Timestamp, first: pd.Timestamp, last: pd.Timestamp) -> pd.Timestamp:
    """ts limited to [first, last]."""
    return min(max(ts, first), last)


def advance(
    ts: pd.Timestamp, samples: int, interval_minutes: int, last: pd.Timestamp
) -> tuple[pd.Timestamp, bool]:
    """Move ts forward by samples; return the new time and whether to keep playing."""
    following = ts + pd.Timedelta(minutes=samples * interval_minutes)
    return (following, True) if following < last else (last, False)


def replay_start(
    first: pd.Timestamp, last: pd.Timestamp, after: pd.Timestamp, interval_minutes: int
) -> pd.Timestamp:
    """First sample after `after` (the start of the test period), within
    the data."""
    return clamp(after + pd.Timedelta(minutes=interval_minutes), first, last)


def period_of(
    ts: pd.Timestamp, train_end: pd.Timestamp, calibration_end: pd.Timestamp
) -> str:
    """Which split ts falls in: training, calibration or test."""
    if ts <= train_end:
        return "training"
    return "calibration" if ts <= calibration_end else "test"
