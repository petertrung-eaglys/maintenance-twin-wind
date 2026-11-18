"""Physics and hybrid pitch models (spec §7.8).

Above rated wind speed the controller pitches the blades to hold rated
power, so the pitch angle is a function of the density-normalised wind
speed v_n. Physics: binned pitch(v_n) per turbine type. Hybrid: physics +
a LightGBM model of its residual, per turbine. Below rated the pitch sits
near fine pitch and carries no signal: predictions there are NaN.

Metrics are in degrees (nRMSE = RMSE in degrees, as for temperatures).
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from rdflib import Graph

from src.binned import binned_means, interpolate
from src.config import farm_names, load_config
from src.graph_utils import get_turbines, load_graph, save_graph
from src.modelling import (
    add_direction_features,
    evaluate,
    merge_metrics,
    model_id_of,
    prediction_column,
    register_model,
    split_masks,
    turbine_paths,
)
from src.physics_power import normalised_wind_speed
from src.residual import ResidualModel

logger = logging.getLogger(__name__)

DEGREE_SCALE = 1.0  # nRMSE of an angle is its RMSE in degrees


def above_rated(frame: pd.DataFrame, rated_wind_speed: float) -> pd.Series:
    """Rows at or above rated wind speed (measured, as in clean.py)."""
    return frame["wind_speed"] >= rated_wind_speed


class PhysicsPitchModel:
    """Expected pitch angle from a binned pitch(v_n) curve above rated."""

    def __init__(self, physics: dict[str, Any], rated_wind_speed: float) -> None:
        self.physics = physics
        self.rated_wind_speed = rated_wind_speed
        self.curve = pd.DataFrame(columns=["x", "y", "count"])

    def _v_n(self, frame: pd.DataFrame) -> pd.Series:
        rho0 = self.physics["reference_air_density"]
        return normalised_wind_speed(frame["wind_speed"], frame["air_density"], rho0)

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Mean pitch per v_n bin over above-rated rows with a pitch value."""
        rows = pd.concat(list(train_frames.values()), ignore_index=True)
        used = rows[above_rated(rows, self.rated_wind_speed) & rows["pitch"].notna()]
        self.curve = binned_means(
            self._v_n(used),
            used["pitch"],
            self.physics["bin_width"],
            self.physics["min_bin_count"],
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected pitch in degrees; NaN below rated wind speed."""
        pitch = interpolate(self.curve, self._v_n(frame))
        return np.where(above_rated(frame, self.rated_wind_speed), pitch, np.nan)

    def to_dict(self) -> dict[str, Any]:
        """The fitted curve, for saving."""
        return {
            "rated_wind_speed": self.rated_wind_speed,
            "wind_speed_normalised": self.curve["x"].tolist(),
            "pitch_deg": self.curve["y"].tolist(),
            "count": self.curve["count"].astype(int).tolist(),
        }


class HybridPitchModel(ResidualModel):
    """Physics pitch plus a learned residual; NaN below rated wind speed."""

    def __init__(
        self, physics: PhysicsPitchModel, spec: dict[str, Any], seed: int
    ) -> None:
        super().__init__(physics, spec, seed, target="pitch")


def fit_physics(
    config: dict[str, Any],
    turbine_model: str,
    train_frames: Mapping[str, pd.DataFrame],
    curve_file: Path,
) -> PhysicsPitchModel:
    """Fit one turbine type's pitch curve and save it."""
    rated = config["cleaning"]["rated_wind_speed"][turbine_model]
    physics = PhysicsPitchModel(config["physics"], rated).fit(train_frames)
    curve_file.write_text(json.dumps(physics.to_dict(), indent=2))
    logger.info("%s: %d pitch bins", curve_file.stem, len(physics.curve))
    return physics


def masks(
    frame: pd.DataFrame, config: dict[str, Any], rated: float
) -> tuple[pd.Series, pd.Series]:
    """Train and test rows above rated wind speed with all inputs."""
    required = [*config["models"]["pitch_hybrid"]["features"], "pitch"]
    train, test = split_masks(frame, config["train_test_split"]["train_end"], required)
    above = above_rated(frame, rated)
    return train & above, test & above


def train_turbine(
    g: Graph,
    config: dict[str, Any],
    code: str,
    turbine_id: str,
    physics: tuple[PhysicsPitchModel, Path],
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Fit the hybrid, score both variants, write predictions and nodes."""
    physics_model, curve_file = physics
    hybrid_spec = config["models"]["pitch_hybrid"]
    frame = add_direction_features(pd.read_parquet(path))
    train, test = masks(frame, config, physics_model.rated_wind_speed)
    hybrid = HybridPitchModel(physics_model, hybrid_spec, config["seed"])
    hybrid.fit({turbine_id: frame[train]})
    hybrid_file = (
        curve_file.parent / f"{model_id_of(hybrid_spec, code, turbine_id)}.txt"
    )
    hybrid.regressor.booster_.save_model(hybrid_file)
    outputs = {"physics": physics_model.predict(frame), "hybrid": hybrid.predict(frame)}
    counts = {"n_train": int(train.sum()), "n_test": int(test.sum())}
    entries = {}
    for spec, model_file in (
        (config["models"]["pitch_physics"], curve_file),
        (hybrid_spec, hybrid_file),
    ):
        expected = outputs[spec["variant"]][test]
        metrics = evaluate(frame.loc[test, "pitch"], expected, DEGREE_SCALE) | counts
        model_id, entries[model_id] = register_model(
            g, spec, code, turbine_id, metrics, model_file
        )
        rmse = metrics["rmse"]
        print(f"{code} {turbine_id} pitch {spec['variant']}: RMSE {rmse:.2f} deg")
    columns = {prediction_column("pitch", v): o for v, o in outputs.items()}
    pd.read_parquet(path).assign(**columns).to_parquet(path, index=False)
    return entries


def train_farm(
    g: Graph, config: dict[str, Any], farm: str
) -> dict[str, dict[str, Any]]:
    """Fit physics per turbine type, then hybrid per turbine, for one farm."""
    code = config["farms"][farm]["code"]
    specs = get_turbines(g, code).set_index("turbine_id")
    paths = turbine_paths(g, code)
    models_dir = Path(config["paths"]["models"]) / farm
    models_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for turbine_model, group in specs.groupby("turbine_model"):
        rated = config["cleaning"]["rated_wind_speed"][turbine_model]
        train_frames = {}
        for tid in group.index:
            frame = add_direction_features(pd.read_parquet(paths[tid]))
            train_frames[tid] = frame[masks(frame, config, rated)[0]]
        curve_file = models_dir / f"pitch_physics_{turbine_model}.json"
        physics = fit_physics(config, turbine_model, train_frames, curve_file)
        for tid in group.index:
            entries |= train_turbine(
                g, config, code, tid, (physics, curve_file), paths[tid]
            )
    return entries


def main() -> None:
    """Train physics and hybrid pitch models for every farm."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    entries = {}
    for farm in farm_names(config):
        entries |= train_farm(g, config, farm)
    save_graph(g, ontology)
    merge_metrics(Path(config["paths"]["models"]) / "metrics.json", entries)
    logger.info("Registered %d pitch models in %s", len(entries), ontology)


if __name__ == "__main__":
    main()
