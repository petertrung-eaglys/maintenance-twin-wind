"""Physics and hybrid power models (spec §7.8).

Physics: density-normalised wind speed v_n = v (rho / rho0)^(1/3) and a
power coefficient curve Cp(v_n) from the method of bins, fitted per turbine
type; expected power P = 1/2 rho0 A Cp(v_n) v_n^3 with each turbine's own
rotor area A, clipped to [0, P_rated] and 0 outside cut-in/cut-out. The
rotor area makes the curve transferable between rotor sizes.

Hybrid: physics prediction + a LightGBM model of its residual, per turbine.

Turbine specs (rotor_diameter_m, rated_power_kw) are read from the frame;
add them with modelling.with_turbine_specs.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from rdflib import Graph

from src.config import farm_names, load_config
from src.graph.graph_utils import get_turbines, load_graph, save_graph
from src.models.binned import binned_means, interpolate
from src.models.modelling import (
    add_direction_features,
    evaluate,
    merge_metrics,
    model_id_of,
    prediction_column,
    register_model,
    split_masks,
    turbine_paths,
    with_turbine_specs,
)
from src.models.residual import ResidualModel

logger = logging.getLogger(__name__)

BETZ_LIMIT = 16 / 27
WATTS_PER_KW = 1000.0


def normalised_wind_speed(
    wind_speed: pd.Series, air_density: pd.Series, reference_density: float
) -> pd.Series:
    """IEC 61400-12-1 density-normalised wind speed."""
    return wind_speed * (air_density / reference_density) ** (1 / 3)


def rotor_area(diameter_m: pd.Series | float) -> pd.Series | float:
    """Swept area in m2."""
    return np.pi * diameter_m**2 / 4


class PhysicsPowerModel:
    """Expected power from a binned power-coefficient curve Cp(v_n)."""

    def __init__(self, physics: dict[str, Any]) -> None:
        self.physics = physics
        self.cp_curve = pd.DataFrame(columns=["x", "y", "count"])

    def _v_n(self, frame: pd.DataFrame) -> pd.Series:
        rho0 = self.physics["reference_air_density"]
        return normalised_wind_speed(frame["wind_speed"], frame["air_density"], rho0)

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Cp per v_n bin from the mean power per unit rotor area, on the
        bins below the first one whose mean power reaches
        `max_power_fraction` of rated (the aerodynamic region; storm
        control can lower power again at high wind)."""
        rows = pd.concat(list(train_frames.values()), ignore_index=True)
        v_n = self._v_n(rows)
        width, min_count = self.physics["bin_width"], self.physics["min_bin_count"]
        per_area = rows["power"] * WATTS_PER_KW / rotor_area(rows["rotor_diameter_m"])
        curve = binned_means(v_n, per_area, width, min_count)
        loading = binned_means(
            v_n, rows["power"] / rows["rated_power_kw"], width, min_count
        )
        limited = loading["y"] >= self.physics["max_power_fraction"]
        first_limited = curve.loc[limited, "x"].min() if limited.any() else np.inf
        aerodynamic = curve[curve["x"] < first_limited]
        rho0 = self.physics["reference_air_density"]
        self.cp_curve = aerodynamic.assign(
            y=aerodynamic["y"] / (0.5 * rho0 * aerodynamic["x"] ** 3)
        ).reset_index(drop=True)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected power in kW for every row."""
        v_n = self._v_n(frame)
        cp = interpolate(self.cp_curve, v_n)
        area = rotor_area(frame["rotor_diameter_m"])
        rho0 = self.physics["reference_air_density"]
        power = 0.5 * rho0 * area * cp * v_n**3 / WATTS_PER_KW
        running = v_n.between(
            self.physics["cut_in_wind_speed"], self.physics["cut_out_wind_speed"]
        )
        clipped = power.clip(lower=0, upper=frame["rated_power_kw"])
        return np.where(running, clipped, 0.0)

    def max_cp(self) -> float:
        """Highest fitted power coefficient (must stay below Betz)."""
        return float(self.cp_curve["y"].max())

    def to_dict(self) -> dict[str, list[float]]:
        """The fitted curve, for saving."""
        return {
            "wind_speed_normalised": self.cp_curve["x"].tolist(),
            "cp": self.cp_curve["y"].tolist(),
            "count": self.cp_curve["count"].astype(int).tolist(),
        }


class HybridPowerModel(ResidualModel):
    """Physics prediction plus a learned residual, clipped to [0, rated]."""

    def __init__(
        self, physics: PhysicsPowerModel, spec: dict[str, Any], seed: int
    ) -> None:
        super().__init__(physics, spec, seed, target="power")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected power in kW for every row."""
        total = super().predict(frame)
        return np.clip(total, 0, frame["rated_power_kw"].to_numpy())


def prepare(frame: pd.DataFrame, spec: pd.Series) -> pd.DataFrame:
    """Add direction features and turbine specs to a turbine frame."""
    return with_turbine_specs(add_direction_features(frame), spec)


def score(
    frame: pd.DataFrame, expected: np.ndarray, test: pd.Series, rated: float
) -> dict[str, float]:
    """Test metrics of a power prediction."""
    return evaluate(frame.loc[test, "power"], expected[test], rated)


def fit_physics(
    physics_config: dict[str, Any],
    train_frames: Mapping[str, pd.DataFrame],
    curve_file: Path,
) -> PhysicsPowerModel:
    """Fit one turbine type's Cp curve, save it and print the Betz check."""
    physics = PhysicsPowerModel(physics_config).fit(train_frames)
    curve_file.write_text(json.dumps(physics.to_dict(), indent=2))
    betz = "below" if physics.max_cp() < BETZ_LIMIT else "ABOVE"
    print(
        f"{curve_file.stem}: max Cp {physics.max_cp():.3f} "
        f"({betz} Betz {BETZ_LIMIT:.3f}), {len(physics.cp_curve)} bins"
    )
    return physics


def train_turbine(
    g: Graph,
    config: dict[str, Any],
    farm: str,
    turbine_id: str,
    physics: tuple[PhysicsPowerModel, Path],
    path: Path,
    spec_row: pd.Series,
) -> dict[str, dict[str, Any]]:
    """Fit the hybrid, score both variants, write predictions and nodes."""
    code, (physics_model, curve_file) = config["farms"][farm]["code"], physics
    hybrid_spec = config["models"]["power_hybrid"]
    frame = prepare(pd.read_parquet(path), spec_row)
    required = [*hybrid_spec["features"], "power"]
    train, test = split_masks(frame, config["train_test_split"]["train_end"], required)
    hybrid = HybridPowerModel(physics_model, hybrid_spec, config["seed"])
    hybrid.fit({turbine_id: frame[train]})
    hybrid_file = (
        curve_file.parent / f"{model_id_of(hybrid_spec, code, turbine_id)}.txt"
    )
    hybrid.regressor.booster_.save_model(hybrid_file)
    outputs = {"physics": physics_model.predict(frame), "hybrid": hybrid.predict(frame)}
    counts = {"n_train": int(train.sum()), "n_test": int(test.sum())}
    entries = {}
    for spec, model_file in (
        (config["models"]["power_physics"], curve_file),
        (hybrid_spec, hybrid_file),
    ):
        rated = spec_row["rated_power_kw"]
        metrics = score(frame, outputs[spec["variant"]], test, rated) | counts
        model_id, entries[model_id] = register_model(
            g, spec, code, turbine_id, metrics, model_file
        )
        print(f"{code} {turbine_id} {spec['variant']}: nRMSE {metrics['nrmse']:.2%}")
    columns = {prediction_column("power", v): o for v, o in outputs.items()}
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
    required = [*config["models"]["power_hybrid"]["features"], "power"]
    entries = {}
    for turbine_model, group in specs.groupby("turbine_model"):
        train_frames = {}
        for tid in group.index:
            frame = prepare(pd.read_parquet(paths[tid]), specs.loc[tid])
            train, _ = split_masks(
                frame, config["train_test_split"]["train_end"], required
            )
            train_frames[tid] = frame[train]
        curve_file = models_dir / f"power_physics_{turbine_model}.json"
        physics = fit_physics(config["physics"], train_frames, curve_file)
        for tid in group.index:
            entries |= train_turbine(
                g,
                config,
                farm,
                tid,
                (physics, curve_file),
                paths[tid],
                specs.loc[tid],
            )
    return entries


def main() -> None:
    """Train physics and hybrid power models for every farm."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    entries = {}
    for farm in farm_names(config):
        entries |= train_farm(g, config, farm)
    save_graph(g, ontology)
    merge_metrics(Path(config["paths"]["models"]) / "metrics.json", entries)
    logger.info("Registered %d models in %s", len(entries), ontology)


if __name__ == "__main__":
    main()
