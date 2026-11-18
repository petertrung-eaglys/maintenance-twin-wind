"""LightGBM baseline power model per turbine (spec §7.7).

The reference every other variant is compared against. Writes the expected
power to each turbine's Parquet, the model node to the graph and the
metrics to models/metrics.json.
"""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdflib import Graph

from src.config import farm_names, load_config
from src.graph_utils import (
    add_model,
    get_turbines,
    load_graph,
    local_name,
    model_iri,
    save_graph,
)
from src.modelling import (
    add_direction_features,
    evaluate,
    merge_metrics,
    prediction_column,
    split_masks,
    turbine_paths,
)

logger = logging.getLogger(__name__)

MODEL_KEY = "power_baseline"


class LightGBMPowerModel:
    """Gradient-boosted expected power, monotone in the configured features."""

    def __init__(self, spec: dict[str, Any], seed: int) -> None:
        self.features: list[str] = spec["features"]
        self.target: str = spec["target"]
        constraints = [int(f in spec["monotone_increasing"]) for f in self.features]
        self.regressor = LGBMRegressor(
            **spec["lightgbm"],
            monotone_constraints=constraints,
            random_state=seed,
            verbose=-1,
        )

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Fit on the rows of all given turbines together."""
        rows = pd.concat(list(train_frames.values()), ignore_index=True)
        self.regressor.fit(rows[self.features], rows[self.target])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected power for every row."""
        return self.regressor.predict(frame[self.features])


def train_turbine(
    frame: pd.DataFrame,
    turbine_id: str,
    spec: dict[str, Any],
    train_end: str,
    rated_power_kw: float,
    seed: int,
) -> tuple[LightGBMPowerModel, dict[str, float], np.ndarray]:
    """Fit on one turbine's train rows; return model, test metrics and the
    prediction for every row."""
    data = add_direction_features(frame)
    train, test = split_masks(data, train_end, [*spec["features"], spec["target"]])
    model = LightGBMPowerModel(spec, seed).fit({turbine_id: data[train]})
    expected = model.predict(data)
    metrics = evaluate(data.loc[test, spec["target"]], expected[test], rated_power_kw)
    return (
        model,
        metrics | {"n_train": int(train.sum()), "n_test": int(test.sum())},
        expected,
    )


def metrics_entry(
    spec: dict[str, Any], farm_code: str, turbine_id: str, metrics: dict
) -> dict[str, Any]:
    """One models/metrics.json record."""
    keys = ("component", "variant", "target")
    return (
        {"farm_code": farm_code, "turbine_id": turbine_id}
        | {k: spec[k] for k in keys}
        | metrics
    )


def run_turbine(
    g: Graph,
    config: dict[str, Any],
    farm: str,
    turbine_id: str,
    path: Path,
    rated_power_kw: float,
) -> tuple[str, dict[str, Any]]:
    """Train one turbine, write its predictions, model file and graph node."""
    spec, code = config["models"][MODEL_KEY], config["farms"][farm]["code"]
    frame = pd.read_parquet(path)
    model, metrics, expected = train_turbine(
        frame,
        turbine_id,
        spec,
        config["train_test_split"]["train_end"],
        rated_power_kw,
        config["seed"],
    )
    column = prediction_column(spec["target"], spec["variant"])
    frame.assign(**{column: expected}).to_parquet(path, index=False)
    model_id = local_name(
        model_iri(spec["model_class"], code, turbine_id, spec["variant"])
    )
    model_file = Path(config["paths"]["models"]) / farm / f"{model_id}.txt"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.regressor.booster_.save_model(model_file)
    add_model(
        g,
        spec["model_class"],
        code,
        turbine_id,
        spec["component"],
        spec["variant"],
        spec["inputs"],
        spec["target"],
        metrics,
        datetime.now(UTC),
        str(model_file),
    )
    return model_id, metrics_entry(spec, code, turbine_id, metrics)


def main() -> None:
    """Train, save and register the baseline for every turbine in the graph."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    entries = {}
    for farm in farm_names(config):
        code = config["farms"][farm]["code"]
        rated = get_turbines(g, code).set_index("turbine_id")["rated_power_kw"]
        for tid, path in turbine_paths(g, code).items():
            model_id, entry = run_turbine(g, config, farm, tid, path, rated[tid])
            entries[model_id] = entry
            print(
                f"{code} {tid}: RMSE {entry['rmse']:.1f} kW, "
                f"nRMSE {entry['nrmse']:.2%}, MAE {entry['mae']:.1f} kW, "
                f"train {entry['n_train']}, test {entry['n_test']}"
            )
    save_graph(g, ontology)
    merge_metrics(Path(config["paths"]["models"]) / "metrics.json", entries)
    logger.info("Registered %d models in %s", len(entries), ontology)


if __name__ == "__main__":
    main()
