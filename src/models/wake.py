"""Wake-aware expected power (spec §7.13).

The physics power model predicts from each turbine's own nacelle wind
speed, which the wakes of upstream turbines have already lowered, so a
wake loss cannot simply be multiplied into it. Instead:

1. PyWake (Bastankhah-Porte-Agel Gaussian wake model with rotor
   averaging, farm layout from the static file) gives each turbine's speed
   ratio, waked / free-stream wind speed, per wind direction and
   free-stream speed (`WakeTable`).
2. Per timestamp, the farm wind direction is the circular mean of the
   turbines' wind directions, and the free-stream wind speed is the mean
   nacelle wind speed of the turbines that are free (speed ratio at least
   `free_ratio`) in that direction, on normal rows.
3. Two variants of the physics curve: `freestream` (physics at the
   free-stream speed) and `wake` (at free-stream speed x the turbine's
   speed ratio). Both are registered as PowerModels, so alerts and
   evaluation include them.

The wake expansion k is fitted per farm on the training rows: the value
at which waked and free rows have the same mean residual (below rated).
The test period then shows whether the wake variant removes the direction
bias of waked sectors. Results: models/wake.json.
"""

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.rotor_avg_models import CGIRotorAvg
from py_wake.site import UniformSite
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from rdflib import Graph
from scipy.optimize import brentq

from src.config import farm_names, load_config
from src.graph.graph_utils import get_turbines, load_graph, save_graph
from src.models.modelling import (
    evaluate,
    merge_metrics,
    prediction_column,
    register_model,
    split_masks,
    turbine_paths,
    with_turbine_specs,
)
from src.models.physics_power import PhysicsPowerModel
from src.storage import write_parquet

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0
FULL_CIRCLE_DEG = 360.0
PERCENT = 100.0
COLUMNS = [
    "timestamp",
    "is_normal",
    "wind_speed",
    "wind_direction",
    "air_density",
    "power",
    prediction_column("power", "physics"),
]


@dataclass(frozen=True)
class WakeTable:
    """Speed ratio (waked / free-stream) per turbine, wind direction and
    free-stream wind speed, on regular grids."""

    turbine_ids: list[str]
    wd: np.ndarray
    ws: np.ndarray
    ratio: np.ndarray  # turbine x wd x ws

    def lookup(self, turbine: np.ndarray, wd: np.ndarray, ws: np.ndarray) -> np.ndarray:
        """Ratio at the nearest grid point (directions wrap around, speeds
        are clipped to the grid); NaN where wd or ws is missing."""
        wd, ws = np.asarray(wd, float), np.asarray(ws, float)
        known = np.isfinite(wd) & np.isfinite(ws)
        wd_step, ws_step = self.wd[1] - self.wd[0], self.ws[1] - self.ws[0]
        wd_index = np.rint(np.where(known, wd, 0) / wd_step).astype(int) % len(self.wd)
        ws_index = np.clip(
            np.rint((np.where(known, ws, self.ws[0]) - self.ws[0]) / ws_step).astype(
                int
            ),
            0,
            len(self.ws) - 1,
        )
        return np.where(known, self.ratio[turbine, wd_index, ws_index], np.nan)

    def to_dict(self) -> dict[str, Any]:
        """The table, for saving."""
        return {
            "turbine_ids": self.turbine_ids,
            "wd": self.wd.tolist(),
            "ws": self.ws.tolist(),
            "ratio": np.round(self.ratio, 4).tolist(),
        }


@dataclass(frozen=True)
class FarmWind:
    """Wind speed, direction and normal flag of every turbine of a farm
    (columns in the wake table's order), rows aligned on timestamp."""

    timestamps: pd.Index
    speeds: np.ndarray
    directions: np.ndarray
    normal: np.ndarray


def local_xy(lat: pd.Series, lon: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """East and north positions in metres around the mean position
    (equirectangular; exact enough within a farm)."""
    scale = np.cos(np.radians(lat.mean()))
    east = np.radians(lon - lon.mean()) * EARTH_RADIUS_M * scale
    north = np.radians(lat - lat.mean()) * EARTH_RADIUS_M
    return east.to_numpy(), north.to_numpy()


def wake_table(turbines: pd.DataFrame, wake: dict[str, Any], k: float) -> WakeTable:
    """PyWake speed ratios for one farm (rows: turbine_id, lat, long,
    rotor_diameter_m, hub_height_m, rated_power_kw) at wake expansion k;
    one generic turbine type per (rotor, hub height, rated power)."""
    keys = list(
        zip(
            turbines["rotor_diameter_m"],
            turbines["hub_height_m"],
            turbines["rated_power_kw"],
            strict=True,
        )
    )
    kinds = list(dict.fromkeys(keys))
    machines = WindTurbines.from_WindTurbine_lst(
        [
            GenericWindTurbine(f"type{i}", d, h, p, wake["turbulence_intensity"])
            for i, (d, h, p) in enumerate(kinds)
        ]
    )
    model = Bastankhah_PorteAgel_2014(
        UniformSite(ti=wake["turbulence_intensity"]),
        machines,
        k=k,
        rotorAvgModel=CGIRotorAvg(wake["rotor_average_points"]),
    )
    wd = np.arange(0.0, FULL_CIRCLE_DEG, wake["wd_step_deg"])
    low, high = wake["ws_range"]
    ws = np.arange(low, high + 1.0, 1.0)
    x, y = local_xy(turbines["lat"], turbines["long"])
    result = model(x, y, type=[kinds.index(k) for k in keys], wd=wd, ws=ws)
    ratio = (result.WS_eff / result.ws).transpose("wt", "wd", "ws").to_numpy()
    return WakeTable(list(turbines["turbine_id"]), wd, ws, ratio)


def farm_direction(directions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Circular mean over the turbines (columns) of each row's valid wind
    directions, in degrees [0, 360); NaN without a valid one."""
    radians = np.radians(np.where(valid, directions, 0.0))
    count = valid.sum(axis=1)
    sin = np.where(valid, np.sin(radians), 0.0).sum(axis=1)
    cos = np.where(valid, np.cos(radians), 0.0).sum(axis=1)
    mean = np.degrees(np.arctan2(sin, cos)) % FULL_CIRCLE_DEG
    return np.where(count > 0, mean, np.nan)


def free_stream(
    speeds: np.ndarray,
    directions: np.ndarray,
    normal: np.ndarray,
    table: WakeTable,
    free_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Farm wind direction and free-stream wind speed per row (rows x
    turbines in the table's order): the mean speed of the turbines whose
    speed ratio in that direction is at least `free_ratio`."""
    valid = normal & np.isfinite(speeds) & np.isfinite(directions)
    wd = farm_direction(directions, valid)
    turbine = np.arange(speeds.shape[1])[None, :]
    ratio = table.lookup(turbine, wd[:, None], speeds)
    free = valid & (ratio >= free_ratio)
    count = free.sum(axis=1)
    total = np.where(free, speeds, 0.0).sum(axis=1)
    return wd, np.where(count > 0, total / np.maximum(count, 1), np.nan)


def wide(frames: Mapping[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """One column of every turbine frame, rows aligned on timestamp."""
    return pd.DataFrame(
        {t: f.set_index("timestamp")[column] for t, f in frames.items()}
    )


def farm_wind(frames: Mapping[str, pd.DataFrame]) -> FarmWind:
    """The farm's wind inputs from its turbine frames."""
    speeds = wide(frames, "wind_speed")
    return FarmWind(
        speeds.index,
        speeds.to_numpy(),
        wide(frames, "wind_direction").to_numpy(),
        wide(frames, "is_normal").fillna(False).to_numpy(bool),
    )


def turbine_views(
    frames: Mapping[str, pd.DataFrame],
    wind: FarmWind,
    table: WakeTable,
    free_ratio: float,
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Per turbine: its frame with `free_wind_speed` added, and its speed
    ratio per row (for the farm direction and free-stream speed)."""
    wd, free_ws = free_stream(
        wind.speeds, wind.directions, wind.normal, table, free_ratio
    )
    stream = pd.DataFrame({"wd": wd, "free_ws": free_ws}, index=wind.timestamps)
    views = {}
    for index, tid in enumerate(table.turbine_ids):
        aligned = stream.reindex(frames[tid]["timestamp"])
        view = frames[tid].assign(free_wind_speed=aligned["free_ws"].to_numpy())
        ratio = table.lookup(
            np.full(len(view), index), aligned["wd"], view["free_wind_speed"]
        )
        views[tid] = (view, ratio)
    return views


def expected_power(
    physics: PhysicsPowerModel, frame: pd.DataFrame, wind_speed: np.ndarray
) -> np.ndarray:
    """Physics power at the given wind speed; NaN where it is missing."""
    predicted = physics.predict(frame.assign(wind_speed=wind_speed))
    return np.where(np.isfinite(wind_speed), predicted, np.nan)


def variant_outputs(
    physics: PhysicsPowerModel, view: pd.DataFrame, ratio: np.ndarray
) -> dict[str, np.ndarray]:
    """Expected power of the freestream and wake variants."""
    free_speed = view["free_wind_speed"].to_numpy()
    return {
        "freestream": expected_power(physics, view, free_speed),
        "wake": expected_power(physics, view, free_speed * ratio),
    }


def direction_bias(power: pd.Series, expected: np.ndarray, rows: np.ndarray) -> float:
    """Relative bias in % over `rows`: sum(measured - expected) / sum(expected)."""
    expected_sum = np.sum(expected[rows])
    return float(
        PERCENT * np.sum(power.to_numpy()[rows] - expected[rows]) / expected_sum
    )


def bias_rows(
    view: pd.DataFrame,
    period: pd.Series,
    outputs: Mapping[str, np.ndarray],
    speed_range: list[float],
) -> np.ndarray:
    """Rows of a period, below rated (free-stream speed in `speed_range`),
    where every variant predicts."""
    low, high = speed_range
    defined = np.isfinite(np.column_stack(list(outputs.values()))).all(axis=1)
    return (
        period.to_numpy()
        & defined
        & view["free_wind_speed"].between(low, high).to_numpy()
    )


def pooled_gap(
    parts: list[tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]],
) -> float:
    """Bias on waked rows minus bias on free rows (%), pooled over turbines;
    each part is (measured, expected, rows, waked)."""

    def bias(waked_side: bool) -> float:
        masks = [rows & (waked == waked_side) for _, _, rows, waked in parts]
        pairs = list(zip(parts, masks, strict=True))
        residual = sum(np.sum(p.to_numpy()[m] - e[m]) for (p, e, _, _), m in pairs)
        expected = sum(np.sum(e[m]) for (_, e, _, _), m in pairs)
        return PERCENT * residual / expected

    return float(bias(True) - bias(False))


def training_gap(
    k: float,
    turbines: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    physics: Mapping[str, PhysicsPowerModel],
    wind: FarmWind,
    config: dict[str, Any],
) -> float:
    """Waked minus free bias of the wake variant on training rows at k."""
    wake = config["wake"]
    table = wake_table(turbines, wake, k)
    parts = []
    for tid, (view, ratio) in turbine_views(
        frames, wind, table, wake["free_ratio"]
    ).items():
        expected = variant_outputs(physics[tid], view, ratio)["wake"]
        train, _ = split_masks(view, config["train_test_split"], ["free_wind_speed"])
        rows = bias_rows(view, train, {"wake": expected}, wake["bias_wind_speed"])
        parts.append((view["power"], expected, rows, ratio < wake["free_ratio"]))
    gap = pooled_gap(parts)
    logger.info("k = %.4f: training gap %+.2f %%", k, gap)
    return gap


def calibrate_k(
    gap_of: Callable[[float], float], k_range: list[float], tolerance: float
) -> float:
    """The k where the gap crosses zero (it falls as k grows), or the end
    of `k_range` with the smaller gap if it does not cross."""
    low, high = k_range
    gap_low, gap_high = gap_of(low), gap_of(high)
    if gap_low * gap_high > 0:
        return low if abs(gap_low) < abs(gap_high) else high
    return float(brentq(gap_of, low, high, xtol=tolerance))


def bias_records(
    view: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
    test: pd.Series,
    ratio: np.ndarray,
    wake: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bias per variant on test rows below rated, waked (speed ratio below
    free_ratio) vs free."""
    rows = bias_rows(view, test, outputs, wake["bias_wind_speed"])
    waked = ratio < wake["free_ratio"]
    return [
        {
            "variant": variant,
            "bias_waked": direction_bias(view["power"], expected, rows & waked),
            "bias_free": direction_bias(view["power"], expected, rows & ~waked),
            "waked_share": float((rows & waked).sum() / max(rows.sum(), 1)),
        }
        for variant, expected in outputs.items()
    ]


def load_farm(
    g: Graph, config: dict[str, Any], farm: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, PhysicsPowerModel]]:
    """Turbine specs, frames (wake columns only) and fitted physics curves
    of one farm."""
    code = config["farms"][farm]["code"]
    specs = get_turbines(g, code).set_index("turbine_id")
    paths = turbine_paths(g, code)
    models_dir = Path(config["paths"]["models"]) / farm
    frames = {
        tid: with_turbine_specs(
            pd.read_parquet(paths[tid], columns=COLUMNS), specs.loc[tid]
        )
        for tid in specs.index
    }
    physics = {
        tid: PhysicsPowerModel.from_dict(
            config["physics"],
            json.loads(
                (
                    models_dir / f"power_physics_{specs.loc[tid, 'turbine_model']}.json"
                ).read_text()
            ),
        )
        for tid in specs.index
    }
    return specs, frames, physics


def train_farm(
    g: Graph, config: dict[str, Any], farm: str
) -> tuple[float, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Fit k, then predict, register and score both variants for every
    turbine of a farm; returns k, metrics and test bias records."""
    code, wake = config["farms"][farm]["code"], config["wake"]
    specs, frames, physics = load_farm(g, config, farm)
    turbines = specs.reset_index()
    wind = farm_wind(frames)
    k = calibrate_k(
        partial(
            training_gap,
            turbines=turbines,
            frames=frames,
            physics=physics,
            wind=wind,
            config=config,
        ),
        wake["k_range"],
        wake["k_tolerance"],
    )
    table = wake_table(turbines, wake, k)
    table_file = Path(config["paths"]["models"]) / farm / "wake_ratios.json"
    table_file.write_text(json.dumps({"k": k} | table.to_dict()))
    paths = turbine_paths(g, code)
    entries, biases = {}, []
    for tid, (view, ratio) in turbine_views(
        frames, wind, table, wake["free_ratio"]
    ).items():
        outputs = variant_outputs(physics[tid], view, ratio)
        _, test = split_masks(view, config["train_test_split"], ["free_wind_speed"])
        rated = specs.loc[tid, "rated_power_kw"]
        for variant, expected in outputs.items():
            scored = test & np.isfinite(expected)
            metrics = evaluate(view.loc[scored, "power"], expected[scored], rated) | {
                "n_train": 0,
                "n_test": int(scored.sum()),
            }
            spec = config["models"][f"power_{variant}"]
            model_id, entries[model_id] = register_model(
                g, spec, code, tid, metrics, table_file
            )
        compared = {"physics": view[prediction_column("power", "physics")].to_numpy()}
        biases += [
            {"farm_code": code, "turbine_id": tid} | record
            for record in bias_records(view, compared | outputs, test, ratio, wake)
        ]
        columns = {prediction_column("power", v): o for v, o in outputs.items()}
        write_parquet(pd.read_parquet(paths[tid]).assign(**columns), paths[tid])
    logger.info("%s: k = %.4f, %d turbines done", code, k, len(frames))
    return k, entries, biases


def print_bias(biases: pd.DataFrame) -> None:
    """Mean bias per farm and variant on waked and free test rows
    (acceptance output)."""
    summary = (
        biases.groupby(["farm_code", "variant"], sort=False)[
            ["bias_waked", "bias_free"]
        ]
        .mean()
        .assign(gap=lambda f: f["bias_waked"] - f["bias_free"])
    )
    print("Mean power residual (%) on waked vs free test rows below rated:")
    print(summary.round(2).to_string())


def main() -> None:
    """Free-stream and wake power models for every farm."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    fitted, entries, biases = {}, {}, []
    for farm in farm_names(config):
        k, farm_entries, farm_biases = train_farm(g, config, farm)
        fitted[config["farms"][farm]["code"]] = k
        entries |= farm_entries
        biases += farm_biases
    save_graph(g, ontology)
    models_dir = Path(config["paths"]["models"])
    merge_metrics(models_dir / "metrics.json", entries)
    (models_dir / "wake.json").write_text(
        json.dumps({"k": fitted, "bias": biases}, indent=2)
    )
    print("Fitted wake expansion k:", {c: round(k, 4) for c, k in fitted.items()})
    print_bias(pd.DataFrame(biases))


if __name__ == "__main__":
    main()
