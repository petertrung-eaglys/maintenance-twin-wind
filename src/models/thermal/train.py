"""Train and compare the thermal models (spec §7.9) for both farms.

Per temperature target: physics (all turbines fitted together, own
parameters each), PINN (one network for all turbines), and per turbine
LightGBM-only (`baseline`) and hybrid (physics + LightGBM residual).
All four are scored on the same test rows: normal, after calibration_end, target
measured and every prediction defined (the physics warm-up is excluded).

Run for selected targets only (others keep their saved models):
    uv run python -m src.models.thermal.train --targets front_bearing_temp
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdflib import Graph

from src.config import farm_names, load_config
from src.graph.graph_utils import get_turbines, load_graph, save_graph
from src.models.modelling import (
    evaluate,
    merge_metrics,
    model_id_of,
    prediction_column,
    register_model,
    split_times,
    turbine_paths,
    with_turbine_specs,
)
from src.models.residual import ResidualModel
from src.models.thermal.baseline import LightGBMThermalModel
from src.models.thermal.features import (
    add_history_features,
    feature_columns,
    normal_target_rows,
    target_spec,
    turbine_key,
)
from src.models.thermal.physics import ThermalPhysicsModel
from src.models.thermal.pinn import ThermalPINN
from src.storage import write_parquet

logger = logging.getLogger(__name__)

VARIANTS = ("physics", "baseline", "hybrid", "pinn")
DEGREE_SCALE = 1.0  # nRMSE of a temperature is its RMSE in degC
TYPE_FEATURES = ["rotor_diameter_m"]


@dataclass(frozen=True)
class Turbine:
    """Where a turbine key comes from."""

    farm: str
    farm_code: str
    turbine_id: str
    path: Path


def thermal_columns(thermal: dict[str, Any]) -> list[str]:
    """The Parquet columns the thermal models read (all others stay on disk)."""
    columns = [
        "timestamp",
        "is_normal",
        *thermal["history_inputs"],
        thermal["reference"],
        *thermal["targets"],
    ]
    return list(dict.fromkeys(columns))


def prepare(
    frame: pd.DataFrame, key: str, spec: pd.Series, thermal: dict[str, Any]
) -> pd.DataFrame:
    """Sorted frame with turbine key (categorical: one value per frame),
    turbine specs and history features."""
    ordered = frame.sort_values("timestamp", ignore_index=True)
    keyed = with_turbine_specs(ordered, spec).assign(
        turbine_key=pd.Categorical([key] * len(ordered))
    )
    return add_history_features(
        keyed, thermal["history_inputs"], thermal["history_windows_min"]
    )


def load_turbines(
    g: Graph, config: dict[str, Any]
) -> tuple[dict[str, Turbine], dict[str, pd.DataFrame]]:
    """Every turbine of every farm, keyed by turbine key, found via the
    graph; only the thermal columns are read."""
    thermal = config["thermal"]
    turbines, frames = {}, {}
    for farm in farm_names(config):
        code = config["farms"][farm]["code"]
        specs = get_turbines(g, code).set_index("turbine_id")
        for tid, path in turbine_paths(g, code).items():
            key = turbine_key(code, tid)
            turbines[key] = Turbine(farm, code, tid, path)
            frame = pd.read_parquet(path, columns=thermal_columns(thermal))
            frames[key] = prepare(frame, key, specs.loc[tid], thermal)
    return turbines, frames


def training_period(frame: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """Rows up to train_end of a time-sorted frame, as a slice (no copy of
    the data; a boolean mask would copy it)."""
    end = frame["timestamp"].searchsorted(train_end, side="right")
    return frame.iloc[:end]


def scored_rows(
    frame: pd.DataFrame, target: str, test_after: pd.Timestamp, outputs: dict
) -> pd.Series:
    """Normal rows after `test_after` (calibration_end) with the target and
    every prediction."""
    defined = np.isfinite(np.column_stack(list(outputs.values()))).all(1)
    after = frame["timestamp"] > test_after
    return normal_target_rows(target)(frame) & after & defined


def train_target(
    g: Graph,
    config: dict[str, Any],
    target: str,
    component: str,
    turbines: dict[str, Turbine],
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """All four variants for one target; returns metrics records and the
    prediction columns per turbine key."""
    thermal, seed = config["thermal"], config["seed"]
    interval = config["sampling"]["interval_minutes"]
    train_end, calibration_end = split_times(config["train_test_split"])
    train = {k: training_period(f, train_end) for k, f in frames.items()}
    features = feature_columns(thermal)
    specs = {
        v: target_spec(config["models"][f"thermal_{v}"], target, component)
        for v in VARIANTS
    }
    shared_dir = Path(config["paths"]["models"]) / "thermal"
    shared_dir.mkdir(parents=True, exist_ok=True)
    physics = ThermalPhysicsModel(target, thermal, interval, seed).fit(train)
    physics_file = shared_dir / f"physics_{target}.json"
    physics_file.write_text(json.dumps(physics.to_dict(), indent=2))
    pinn = ThermalPINN(target, thermal, interval, [*features, *TYPE_FEATURES], seed)
    pinn.fit(train)
    pinn_file = shared_dir / f"pinn_{target}.pt"
    pinn.save(pinn_file)
    entries, columns = {}, {}
    for key, frame in frames.items():
        turbine = turbines[key]
        baseline = LightGBMThermalModel(specs["baseline"], seed, target, features)
        baseline.fit({key: train[key]})
        hybrid = ResidualModel(
            physics,
            specs["hybrid"] | {"features": features},
            seed,
            target,
            rows=normal_target_rows(target),
            base_as_feature=specs["hybrid"]["base_as_feature"],
        ).fit({key: train[key]})
        outputs = {
            "physics": physics.predict(frame),
            "baseline": baseline.predict(frame),
            "hybrid": hybrid.predict(frame),
            "pinn": pinn.predict(frame),
        }
        files = {"physics": physics_file, "pinn": pinn_file}
        for variant, model in (("baseline", baseline), ("hybrid", hybrid)):
            name = model_id_of(specs[variant], turbine.farm_code, turbine.turbine_id)
            files[variant] = (
                Path(config["paths"]["models"]) / turbine.farm / f"{name}.txt"
            )
            model.regressor.booster_.save_model(files[variant])
        scored = scored_rows(frame, target, calibration_end, outputs)
        n_train = int(normal_target_rows(target)(train[key]).sum())
        rmse = {}
        for variant in VARIANTS:
            metrics = evaluate(
                frame.loc[scored, target], outputs[variant][scored], DEGREE_SCALE
            ) | {"n_train": n_train, "n_test": int(scored.sum())}
            model_id, entries[model_id] = register_model(
                g,
                specs[variant],
                turbine.farm_code,
                turbine.turbine_id,
                metrics,
                files[variant],
            )
            rmse[variant] = metrics["rmse"]
        print(
            f"{key} {target}: "
            + ", ".join(f"{v} {r:.2f}" for v, r in rmse.items())
            + " degC"
        )
        columns[key] = {prediction_column(target, v): o for v, o in outputs.items()}
    return entries, columns


def write_predictions(
    turbines: dict[str, Turbine], columns: dict[str, dict[str, np.ndarray]]
) -> None:
    """Add the expected-temperature columns to each turbine Parquet (rows in
    timestamp order, as the frames were prepared)."""
    for key, turbine in turbines.items():
        frame = pd.read_parquet(turbine.path).sort_values(
            "timestamp", ignore_index=True
        )
        write_parquet(frame.assign(**columns[key]), turbine.path)


def parse_targets(argv: list[str] | None = None) -> list[str] | None:
    """The --targets option (None: all configured targets)."""
    parser = argparse.ArgumentParser(description="Train the thermal models.")
    parser.add_argument(
        "--targets", nargs="+", help="temperature targets (default: all)"
    )
    return parser.parse_args(argv).targets


def selected_targets(
    targets: dict[str, str], requested: list[str] | None
) -> dict[str, str]:
    """The configured target -> component map, limited to `requested`."""
    if requested is None:
        return targets
    unknown = [t for t in requested if t not in targets]
    if unknown:
        raise ValueError(f"unknown thermal targets {unknown}; known: {list(targets)}")
    return {t: c for t, c in targets.items() if t in requested}


def main(argv: list[str] | None = None) -> None:
    """Train the four thermal variants for the selected targets (default:
    all) and every turbine."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    targets = selected_targets(config["thermal"]["targets"], parse_targets(argv))
    torch.set_num_threads(config["compute"]["threads"])
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    turbines, frames = load_turbines(g, config)
    for turbine in turbines.values():
        (Path(config["paths"]["models"]) / turbine.farm).mkdir(
            parents=True, exist_ok=True
        )
    metrics_file = Path(config["paths"]["models"]) / "metrics.json"
    for target, component in targets.items():
        entries, columns = train_target(g, config, target, component, turbines, frames)
        # Save after every target: bounded memory, and an interrupted run
        # keeps the targets it finished.
        write_predictions(turbines, columns)
        save_graph(g, ontology)
        merge_metrics(metrics_file, entries)
        logger.info("%s: registered %d models in %s", target, len(entries), ontology)


if __name__ == "__main__":
    main()
