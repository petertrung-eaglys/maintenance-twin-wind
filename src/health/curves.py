"""Detection versus false alarms across alert thresholds (README §7.10).

A single threshold compares models at different alert rates, and a model
that alerts more also detects more by chance. Here every power, pitch and
thermal model's test-period alerts are recomputed for several threshold
quantiles (all set on the calibration year, so nothing is tuned on the
test period), with raw and fleet-relative scores, and evaluated in both
scopes. The curve summary is the lift: detection rate minus chance,
averaged over the quantiles. Writes models/curves.json.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import farm_names, load_config
from src.graph.graph_utils import get_fault_events, get_models, load_graph
from src.health.alerts import alert_periods, calibration_threshold, merge_periods
from src.health.evaluate import GROUP, days_after, evaluate_scopes
from src.health.scores import farm_scores, residual_models
from src.models.modelling import split_times, turbine_paths

logger = logging.getLogger(__name__)

ALERT_COLUMNS = [*GROUP, "turbine_id", "alert", "start"]
CURVE_KEY = ["scope", *GROUP, "fleet"]


def threshold_alerts(
    scores: dict[str, pd.Series],
    models: pd.DataFrame,
    settings: dict[str, Any],
    split: tuple[pd.Timestamp, pd.Timestamp],
    quantile: float,
) -> pd.DataFrame:
    """Test-period alerts of every scored model at one threshold quantile."""
    train_end, calibration_end = split
    rows = []
    for model in residual_models(models).itertuples():
        model_id = model.model.split("#")[-1]
        score = scores.get(model_id)
        if score is None or score.empty:
            continue
        threshold = calibration_threshold(score, train_end, calibration_end, quantile)
        periods = merge_periods(
            alert_periods(
                score[score.index > train_end],
                threshold,
                pd.Timedelta(hours=settings["max_gap_hours"]),
                pd.Timedelta(hours=settings["min_duration_hours"]),
            ),
            pd.Timedelta(days=settings["merge_gap_days"]),
        )
        test = periods[periods["start"] > calibration_end]
        identity = {key: getattr(model, key) for key in GROUP}
        rows += [
            identity
            | {"turbine_id": model.turbine_id, "alert": f"{model_id}_{i}", "start": s}
            for i, s in enumerate(test["start"])
        ]
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def lift(curve: pd.DataFrame) -> float:
    """Mean of detection rate minus chance over a curve's points."""
    return float((curve["detection_rate"] - curve["chance_detection_rate"]).mean())


def main() -> None:
    """Curves for every model, raw and fleet-relative, both scopes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    settings, evaluation = config["alerts"], config["evaluation"]
    split = split_times(config["train_test_split"])
    lookback = pd.Timedelta(days=evaluation["lookback_days"])
    gap = pd.Timedelta(days=evaluation["merge_gap_days"])
    g = load_graph(config["paths"]["ontology"])
    events = get_fault_events(g)
    events = events[events["start"] > split[1]]
    codes = {farm: config["farms"][farm]["code"] for farm in farm_names(config)}
    days = pd.concat(
        {code: days_after(turbine_paths(g, code), split[1]) for code in codes.values()}
    )
    models = get_models(g)
    groups = residual_models(models)[GROUP].drop_duplicates()
    points = []
    for fleet in (False, True):
        alerts = {q: [] for q in evaluation["curve_quantiles"]}
        for farm, code in codes.items():
            mine = models[models["farm_code"] == code]
            scores = farm_scores(config, mine, turbine_paths(g, code), fleet)
            for quantile, found in alerts.items():
                found.append(threshold_alerts(scores, mine, settings, split, quantile))
            logger.info("%s scored (fleet-relative: %s)", farm, fleet)
        for quantile, found in alerts.items():
            summary = evaluate_scopes(
                pd.concat(found, ignore_index=True),
                events,
                days,
                lookback,
                gap,
                groups,
                evaluation["signal_events"],
            )
            points.append(summary.assign(quantile=quantile, fleet=fleet))
    curves = pd.concat(points, ignore_index=True)
    out = Path(config["paths"]["models"]) / "curves.json"
    out.write_text(json.dumps(json.loads(curves.to_json(orient="records")), indent=2))
    print_curves(curves)


def print_curves(curves: pd.DataFrame, min_episodes: int = 5) -> None:
    """Lift per curve (rows with at least `min_episodes` episodes), raw vs
    fleet-relative, with the false-alarm range covered."""
    known = curves[curves["episodes"] >= min_episodes]
    table = (
        known.groupby(CURVE_KEY)
        .apply(
            lambda c: pd.Series(
                {
                    "episodes": int(c["episodes"].iloc[0]),
                    "lift": lift(c),
                    "fa_min": c["false_alarms_per_turbine_year"].min(),
                    "fa_max": c["false_alarms_per_turbine_year"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    wide = table.pivot_table(
        index=["scope", "farm_code", "component_key", "target", "variant", "episodes"],
        columns="fleet",
        values="lift",
    ).rename(columns={False: "lift_raw", True: "lift_fleet"})
    print("Lift = detection rate - chance, mean over threshold quantiles")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(wide.round(3).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
