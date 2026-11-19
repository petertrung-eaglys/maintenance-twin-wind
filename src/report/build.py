"""Report figures (docs/figures/) and the Markdown tables of docs/report.md
(spec §7.12). Reads the processed data, models/*.json and the graph;
writes only figures. The tables are printed (acceptance output) to be
pasted into the report."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph

from src.config import farm_names, farm_processed_dir, load_config
from src.experiments.cross_site import summarise
from src.graph.graph_utils import get_alerts, load_graph
from src.models.modelling import split_times, turbine_paths
from src.report.cases import case_events, case_figure, case_scores
from src.report.comparison import (
    comparison_figure,
    cross_site_figure,
    cross_site_panels,
    cross_site_table,
    metric_frame,
    signal_label,
    variant_means,
)
from src.report.coverage import (
    availability,
    availability_figure,
    coverage_figure,
    monthly_normal_share,
    report_months,
    yearly_counts,
)
from src.report.detection import detection_figure, detection_points, mean_lift
from src.report.style import apply_style, markdown_table, save, sequential_map
from src.report.wake import (
    alert_time,
    comparable_rows,
    wake_alert_figure,
    waked_rows,
    waked_share_table,
)

logger = logging.getLogger(__name__)

PERCENT = 100


def read_json(path: Path) -> Any:
    """A JSON file's content."""
    return json.loads(path.read_text())


def data_overview(g: Graph, config: dict[str, Any], out: Path) -> None:
    """Normal-row coverage and signal availability figures (one turbine
    read at a time, only the needed columns)."""
    report = config["report"]
    months = report_months(config["years"])
    signals = [*config["signals"], *config["component_signals"]]
    shares, available = {}, {}
    for farm in farm_names(config):
        code = config["farms"][farm]["code"]
        rows, counts = {}, []
        for tid, path in turbine_paths(g, code).items():
            frame = pd.read_parquet(path, columns=["timestamp", "is_normal", *signals])
            rows[tid] = monthly_normal_share(
                frame, months, config["sampling"]["interval_minutes"]
            )
            counts.append(yearly_counts(frame, signals))
        shares[code] = pd.DataFrame(rows).T
        available[code] = availability(counts)
    cmap = sequential_map(report["colors"])
    save(
        coverage_figure(shares, cmap, report["colors"]),
        out / "data_coverage.png",
        report["dpi"],
    )
    save(
        availability_figure(available, cmap, report["colors"]),
        out / "signal_availability.png",
        report["dpi"],
    )


def model_comparison(config: dict[str, Any], out: Path) -> dict[str, pd.DataFrame]:
    """In-site comparison figure; returns its tables."""
    report = config["report"]
    variants = list(report["colors"]["variants"])
    frame = metric_frame(read_json(Path(config["paths"]["models"]) / "metrics.json"))
    tables = {
        "Power: test nRMSE (%)": variant_means(frame, "PowerModel", "nrmse", variants)
        * PERCENT,
        "Pitch above rated: test RMSE (°)": variant_means(
            frame, "PitchModel", "rmse", variants
        ),
        "Temperatures: test RMSE (°C)": variant_means(
            frame, "ThermalModel", "rmse", variants
        ),
    }
    save(
        comparison_figure(*tables.values(), report["colors"]["variants"]),
        out / "model_comparison.png",
        report["dpi"],
    )
    return tables


def cross_site(config: dict[str, Any], out: Path) -> dict[str, pd.DataFrame]:
    """Cross-site figure; returns its table."""
    report = config["report"]
    colors = report["colors"]
    records = pd.DataFrame(
        read_json(Path(config["paths"]["models"]) / "cross_site.json")
    )
    summary = summarise(records)
    save(
        cross_site_figure(
            cross_site_panels(summary), colors["variants"], colors["muted"]
        ),
        out / "cross_site.png",
        report["dpi"],
    )
    title = "Cross-site: test RMSE trained on the test farm → on the other farm"
    return {title: cross_site_table(summary, list(colors["variants"]))}


def detection(config: dict[str, Any], out: Path) -> dict[str, pd.DataFrame]:
    """Detection-vs-chance figure (component scope, fleet-relative); returns
    the mean lift per scope and variant."""
    report = config["report"]
    curves = pd.DataFrame(read_json(Path(config["paths"]["models"]) / "curves.json"))
    points = detection_points(curves, "component", True, report["min_episodes"])
    colors = report["colors"]
    save(
        detection_figure(
            points, colors["variants"], report["variant_markers"], colors["muted"]
        ),
        out / "detection.png",
        report["dpi"],
    )
    lifts = {
        (scope, variant): mean_lift(rows)
        for scope in ("component", "signal")
        for variant, rows in detection_points(
            curves, scope, True, report["min_episodes"]
        ).groupby("variant")
    }
    table = pd.Series(lifts).unstack().rename_axis("scope")
    return {"Mean lift (detection − chance), fleet-relative": table}


def case_studies(g: Graph, config: dict[str, Any], out: Path) -> None:
    """One figure per configured case."""
    report = config["report"]
    start = pd.Timestamp(report["case_start"], tz="UTC")
    train_end, calibration_end = split_times(config["train_test_split"])
    calibration = (train_end.tz_convert(None), calibration_end.tz_convert(None))
    for case in report["cases"]:
        code = config["farms"][case["farm"]]["code"]
        paths = turbine_paths(g, code)
        events = pd.read_parquet(
            farm_processed_dir(config, case["farm"]) / "events.parquet"
        )
        scores = {
            v: s[s.index >= start]
            for v, s in case_scores(config, g, code, paths, case).items()
        }
        title = (
            f"{code} {case['turbine_id']} {signal_label(case['target'])}: "
            f"residual vs the rest of the farm, and {case['component']} events"
        )
        figure = case_figure(
            scores,
            case_events(events, case["turbine_id"], case["component"], start),
            calibration,
            report,
            title,
        )
        name = f"case_{code}_{case['turbine_id']}_{case['target']}.png".lower()
        save(figure, out / name, report["dpi"])
        logger.info("case %s done", name)


def wake_alerts(g: Graph, config: dict[str, Any], out: Path) -> dict[str, pd.DataFrame]:
    """Waked share of each power variant's test alert time (one turbine
    read at a time); returns its table."""
    report = config["report"]
    _, calibration_end = split_times(config["train_test_split"])
    columns = [
        "timestamp",
        "is_normal",
        "expected_power_wake",
        "expected_power_freestream",
    ]
    records = []
    for farm in farm_names(config):
        code = config["farms"][farm]["code"]
        alerts = get_alerts(g, code)
        power = alerts[
            (alerts["model_class"] == "PowerModel")
            & (alerts["start"] > calibration_end)
        ]
        for tid, path in turbine_paths(g, code).items():
            frame = pd.read_parquet(path, columns=columns)
            test = frame[
                frame["is_normal"]
                & (frame["timestamp"] > calibration_end)
                & comparable_rows(frame)
            ]
            waked = waked_rows(test, report["wake_power_ratio"])
            records += [
                {"farm_code": code} | r
                for r in alert_time(test, power[power["turbine_id"] == tid], waked)
            ]
    variants = [v for v in report["colors"]["variants"] if v not in ("pinn",)]
    table = waked_share_table(pd.DataFrame(records), variants)
    colors = report["colors"]
    save(
        wake_alert_figure(table, colors["variants"], colors["muted"]),
        out / "wake_alerts.png",
        report["dpi"],
    )
    return {"Power-alert time while waked (test period)": table}


def main() -> None:
    """Write every report figure and print the report tables."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    report = config["report"]
    apply_style(report["colors"])
    out = Path(report["figures"])
    g = load_graph(config["paths"]["ontology"])
    data_overview(g, config, out)
    logger.info("data overview done")
    tables = model_comparison(config, out) | cross_site(config, out)
    tables |= detection(config, out) | wake_alerts(g, config, out)
    case_studies(g, config, out)
    for title, table in tables.items():
        print(f"\n{title}\n\n{markdown_table(table)}")


if __name__ == "__main__":
    main()
