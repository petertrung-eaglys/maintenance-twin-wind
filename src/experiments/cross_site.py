"""Cross-site experiment (spec §7.11).

Hypothesis: physics and hybrid models transfer across rotor sizes (MM92
Kelmarsh, 92 m; MM82 Penmanshiel, 82 m) better than ML-only models.

Every power and thermal variant is fitted once on all turbines of one farm
(the training years of the in-site models, up to train_end) and applied to
every turbine of the other farm; both directions. Its test RMSE is compared
with the in-site model's (the prediction columns in the turbine Parquet)
on the same rows: test period, normal, target measured, every in-site and
cross-site prediction of that signal defined.

Cross-site models need one model for a whole farm: power physics is one
Cp curve per type anyway and scales with each turbine's rotor area;
thermal physics is pooled (one parameter set); the PINN uses its mean
turbine embedding for the other farm's turbines and no turbine-type
feature (it is constant within one farm). Nothing is registered in the
graph; results go to models/cross_site.json. Each signal's records are
also saved to models/cross_site/<signal>.json as soon as it is done, and
a rerun skips the signals saved there (the workstation may reset during
the run); delete that folder to recompute everything.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import farm_names, load_config
from src.graph.graph_utils import get_turbines, load_graph
from src.models.modelling import (
    prediction_column,
    split_masks,
    split_times,
    turbine_paths,
)
from src.models.physics_power import HybridPowerModel, PhysicsPowerModel, prepare
from src.models.power_baseline import LightGBMPowerModel
from src.models.residual import ResidualModel
from src.models.thermal.baseline import LightGBMThermalModel
from src.models.thermal.features import (
    feature_columns,
    normal_target_rows,
    target_spec,
)
from src.models.thermal.physics import ThermalPhysicsModel
from src.models.thermal.pinn import ThermalPINN
from src.models.thermal.train import VARIANTS as THERMAL_VARIANTS
from src.models.thermal.train import load_turbines, training_period

logger = logging.getLogger(__name__)

POWER_VARIANTS = ("baseline", "physics", "hybrid")
POWER_COLUMNS = [
    "timestamp",
    "is_normal",
    "wind_speed",
    "wind_direction",
    "ambient_temp",
    "air_density",
    "power",
]


def rmse(error: np.ndarray) -> float:
    """Root mean square of an error array."""
    return float(np.sqrt(np.mean(np.square(error))))


def compare(
    measured: pd.Series,
    rows: pd.Series,
    in_site: Mapping[str, np.ndarray],
    cross: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """In-site and cross-site RMSE per variant on the same rows: `rows`
    where every in-site and cross-site prediction is defined."""
    defined = np.isfinite(np.column_stack([*in_site.values(), *cross.values()]))
    common = rows.to_numpy() & defined.all(axis=1)
    truth = measured.to_numpy()[common]
    return [
        {
            "variant": variant,
            "rmse_in": rmse(in_site[variant][common] - truth),
            "rmse_cross": rmse(cross[variant][common] - truth),
            "n": int(common.sum()),
        }
        for variant in in_site
    ]


def summarise(records: pd.DataFrame) -> pd.DataFrame:
    """Mean in-site and cross-site RMSE over turbines and the relative
    increase, per direction, signal and variant."""
    summary = (
        records.groupby(["train", "test", "signal", "variant"])[
            ["rmse_in", "rmse_cross"]
        ]
        .mean()
        .reset_index()
    )
    return summary.assign(increase=summary["rmse_cross"] / summary["rmse_in"] - 1)


def power_models(
    config: dict[str, Any], train_frames: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """The three power variants fitted on one farm's training rows."""
    models = config["models"]
    physics = PhysicsPowerModel(config["physics"]).fit(train_frames)
    return {
        "baseline": LightGBMPowerModel(models["power_baseline"], config["seed"]).fit(
            train_frames
        ),
        "physics": physics,
        "hybrid": HybridPowerModel(physics, models["power_hybrid"], config["seed"]).fit(
            train_frames
        ),
    }


def power_frames(g: Any, config: dict[str, Any]) -> dict[str, dict[str, pd.DataFrame]]:
    """Power-model frames (with the in-site predictions) per farm code and
    turbine id."""
    in_site = [prediction_column("power", v) for v in POWER_VARIANTS]
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for farm in farm_names(config):
        code = config["farms"][farm]["code"]
        specs = get_turbines(g, code).set_index("turbine_id")
        frames[code] = {
            tid: prepare(
                pd.read_parquet(path, columns=[*POWER_COLUMNS, *in_site]),
                specs.loc[tid],
            )
            for tid, path in turbine_paths(g, code).items()
        }
    return frames


def power_experiment(g: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Power records for both directions."""
    split = config["train_test_split"]
    required = [*config["models"]["power_hybrid"]["features"], "power"]
    frames = power_frames(g, config)
    records = []
    for source, target in directions(frames):
        train = {
            tid: f[split_masks(f, split, required)[0]]
            for tid, f in frames[source].items()
        }
        models = power_models(config, train)
        for tid, frame in frames[target].items():
            _, test = split_masks(frame, split, required)
            in_site = {
                v: frame[prediction_column("power", v)].to_numpy()
                for v in POWER_VARIANTS
            }
            cross = {v: np.asarray(m.predict(frame), float) for v, m in models.items()}
            records += [
                {"train": source, "test": target, "turbine_id": tid, "signal": "power"}
                | r
                for r in compare(frame["power"], test, in_site, cross)
            ]
        logger.info("power %s -> %s done", source, target)
    return records


def directions(frames: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Both (train, test) farm pairs."""
    codes = list(frames)
    return [(a, b) for a in codes for b in codes if a != b]


def thermal_models(
    config: dict[str, Any],
    target: str,
    component: str,
    train_frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """The four thermal variants fitted on one farm, usable on any turbine."""
    thermal, seed = config["thermal"], config["seed"]
    interval = config["sampling"]["interval_minutes"]
    features = feature_columns(thermal)
    spec = {
        v: target_spec(config["models"][f"thermal_{v}"], target, component)
        for v in THERMAL_VARIANTS
    }
    physics = ThermalPhysicsModel(target, thermal, interval, seed, pooled=True).fit(
        train_frames
    )
    return {
        "physics": physics,
        "baseline": LightGBMThermalModel(spec["baseline"], seed, target, features).fit(
            train_frames
        ),
        "hybrid": ResidualModel(
            physics,
            spec["hybrid"] | {"features": features},
            seed,
            target,
            rows=normal_target_rows(target),
            base_as_feature=spec["hybrid"]["base_as_feature"],
        ).fit(train_frames),
        "pinn": ThermalPINN(
            target, thermal, interval, features, seed, unseen_turbines="mean"
        ).fit(train_frames),
    }


def in_site_predictions(
    path: Path, target: str, timestamps: pd.Series
) -> dict[str, np.ndarray]:
    """The in-site thermal predictions of a turbine, aligned to timestamps."""
    columns = {prediction_column(target, v): v for v in THERMAL_VARIANTS}
    stored = pd.read_parquet(path, columns=["timestamp", *columns])
    aligned = stored.set_index("timestamp").reindex(timestamps)
    return {variant: aligned[c].to_numpy() for c, variant in columns.items()}


def thermal_experiment(
    config: dict[str, Any],
    target: str,
    component: str,
    turbines: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Thermal records of one temperature for both directions."""
    train_end, calibration_end = split_times(config["train_test_split"])
    by_farm: dict[str, list[str]] = {}
    for key, turbine in turbines.items():
        by_farm.setdefault(turbine.farm_code, []).append(key)
    records = []
    for source, test_farm in directions(by_farm):
        train = {k: training_period(frames[k], train_end) for k in by_farm[source]}
        models = thermal_models(config, target, component, train)
        for key in by_farm[test_farm]:
            frame = frames[key]
            rows = normal_target_rows(target)(frame) & (
                frame["timestamp"] > calibration_end
            )
            in_site = in_site_predictions(
                turbines[key].path, target, frame["timestamp"]
            )
            cross = {v: m.predict(frame) for v, m in models.items()}
            records += [
                {
                    "train": source,
                    "test": test_farm,
                    "turbine_id": turbines[key].turbine_id,
                    "signal": target,
                }
                | r
                for r in compare(frame[target], rows, in_site, cross)
            ]
        logger.info("%s %s -> %s done", target, source, test_farm)
    return records


def save_part(folder: Path, signal: str, records: list[dict[str, Any]]) -> None:
    """One signal's records as <folder>/<signal>.json, written atomically
    (temporary file, then rename) so a reset leaves no partial part."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{signal}.json.tmp"
    temporary.write_text(json.dumps(records, indent=2))
    temporary.replace(folder / f"{signal}.json")


def load_part(folder: Path, signal: str) -> list[dict[str, Any]] | None:
    """A saved signal's records, or None if not saved yet."""
    path = folder / f"{signal}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    """Run both directions for power and every thermal target, skipping
    signals already saved; print and save."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    torch.set_num_threads(config["compute"]["threads"])
    g = load_graph(config["paths"]["ontology"])
    models_dir = Path(config["paths"]["models"])
    parts = models_dir / "cross_site"
    records = load_part(parts, "power")
    if records is None:
        records = power_experiment(g, config)
        save_part(parts, "power", records)
    thermal_data = None
    for target, component in config["thermal"]["targets"].items():
        saved = load_part(parts, target)
        if saved is None:
            thermal_data = thermal_data or load_turbines(g, config)
            saved = thermal_experiment(config, target, component, *thermal_data)
            save_part(parts, target, saved)
        else:
            logger.info("%s: loaded from %s", target, parts)
        records = records + saved
    records = pd.DataFrame(records)
    out = models_dir / "cross_site.json"
    out.write_text(json.dumps(json.loads(records.to_json(orient="records")), indent=2))
    summary = summarise(records)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("Mean test RMSE in-site vs cross-site (kW for power, degC else):")
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
