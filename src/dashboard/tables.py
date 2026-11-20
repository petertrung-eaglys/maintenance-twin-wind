"""Model-comparison tables of one farm, from models/*.json (the report's
helpers do the averaging)."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.report.comparison import in_order

EVALUATION_COLUMNS = [
    "component_key",
    "model_class",
    "target",
    "variant",
    "episodes",
    "detected",
    "detection_rate",
    "chance_detection_rate",
    "lead_hours_median",
    "false_alarms_per_turbine_year",
]


def turbine_metrics(
    metrics: pd.DataFrame,
    farm_code: str,
    model_class: str,
    value: str,
    variants: Mapping[str, Any],
) -> pd.DataFrame:
    """`value` per turbine and target (rows) and variant (columns, in the
    configured order) for one farm and model class; metric_frame rows."""
    rows = metrics[
        (metrics["farm_code"] == farm_code) & (metrics["model_class"] == model_class)
    ]
    table = rows.pivot_table(
        index=["target", "turbine_id"], columns="variant", values=value
    )
    return in_order(table.T, variants).T


def evaluation_view(
    summary: list[dict[str, Any]], farm_code: str, scope: str
) -> pd.DataFrame:
    """Detection, chance and false alarms per model group of one farm and
    evaluation scope."""
    frame = pd.DataFrame(summary)
    chosen = frame[(frame["farm_code"] == farm_code) & (frame["scope"] == scope)]
    return chosen[EVALUATION_COLUMNS].reset_index(drop=True)


def drift_view(drift_alerts: list[dict[str, Any]], farm_code: str) -> pd.DataFrame:
    """Drift alerts of one farm (turbine, model, period, peak)."""
    frame = pd.DataFrame(
        drift_alerts,
        columns=[
            "farm_code",
            "turbine_id",
            "target",
            "variant",
            "start",
            "end",
            "peak",
        ],
    )
    chosen = frame[frame["farm_code"] == farm_code].drop(columns="farm_code")
    return chosen.assign(
        start=pd.to_datetime(chosen["start"]).dt.strftime("%Y-%m"),
        end=pd.to_datetime(chosen["end"]).dt.strftime("%Y-%m"),
    ).reset_index(drop=True)


def wake_view(wake: Mapping[str, Any], farm_code: str) -> pd.DataFrame:
    """Waked and free residual bias (%) per turbine and power variant."""
    frame = pd.DataFrame(wake["bias"])
    chosen = frame[frame["farm_code"] == farm_code]
    return chosen.pivot_table(
        index="turbine_id", columns="variant", values=["bias_waked", "bias_free"]
    )
