"""Static yaw misalignment model (spec §7.8).

Below rated wind speed a turbine misaligned by gamma to the wind loses
power as cos^k(gamma). The measured misalignment is the vane's relative
wind angle (clean.py), so a vane offset shifts the curve: the power ratio
to the turbine's own training power curve, binned by measured
misalignment, peaks at the static misalignment `offset_deg`:

    ratio = gain * cos^k(gamma - offset)

Fitted per turbine on healthy training rows, then over rolling windows of
all rows to track drift (alerts in §7.10). The offset counts as identified
only if the binned data fall `min_peak_drop` below the fitted peak at both
ends; otherwise the data show no peak (README, "Yaw model").
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from rdflib import Graph
from scipy.optimize import curve_fit

from src.binned import binned_means, interpolate
from src.config import farm_names, farm_processed_dir, load_config
from src.graph_utils import get_turbines, load_graph, save_graph
from src.modelling import (
    evaluate,
    merge_metrics,
    model_id_of,
    register_model,
    split_masks,
    turbine_paths,
)
from src.physics_power import normalised_wind_speed

logger = logging.getLogger(__name__)

REQUIRED = ["wind_speed", "air_density", "power", "yaw_misalignment"]
# (gain, offset_deg, exponent): start values; the offset bound comes from
# max_misalignment_deg
INITIAL = (1.0, 0.0, 2.0)
GAIN_BOUNDS = (0.5, 1.5)
EXPONENT_BOUNDS = (0.1, 20.0)
WINDOW_COLUMNS = [
    "window_start",
    "window_end",
    "offset_deg",
    "exponent",
    "peak_drop",
    "identified",
    "n",
]


def cosine_power_ratio(
    gamma: np.ndarray, gain: float, offset: float, exponent: float
) -> np.ndarray:
    """gain * cos^k(gamma - offset), with gamma in degrees (0 past 90)."""
    cosine = np.clip(np.cos(np.deg2rad(gamma - offset)), 0.0, None)
    return gain * cosine**exponent


def fit_offset(
    ratio: pd.Series, gamma: pd.Series, yaw: dict[str, Any]
) -> dict[str, Any] | None:
    """Fit gain * cos^k(gamma - offset) to the binned power ratio.

    Returns gain, offset_deg, exponent, peak_drop (`peak_drop`),
    identified and n, or None when fewer than `min_bins` bins have enough
    rows with a ratio.
    """
    limit = yaw["max_misalignment_deg"]
    used = (gamma.abs() <= limit) & ratio.notna()
    bins = binned_means(
        gamma[used], ratio[used], yaw["bin_width_deg"], yaw["min_bin_count"]
    )
    if len(bins) < yaw["min_bins"]:
        return None
    (gain, offset, exponent), _ = curve_fit(
        cosine_power_ratio,
        bins["x"],
        bins["y"],
        p0=INITIAL,
        sigma=1 / np.sqrt(bins["count"]),
        bounds=(
            (GAIN_BOUNDS[0], -limit, EXPONENT_BOUNDS[0]),
            (GAIN_BOUNDS[1], limit, EXPONENT_BOUNDS[1]),
        ),
    )
    return {
        "gain": float(gain),
        "offset_deg": float(offset),
        "exponent": float(exponent),
        "peak_drop": (drop := peak_drop(bins, gain, yaw["edge_bins"])),
        "identified": drop >= yaw["min_peak_drop"],
        "n": int(used.sum()),
    }


def peak_drop(bins: pd.DataFrame, gain: float, edge_bins: int) -> float:
    """How far the data fall below the fitted peak at the ends: 1 minus the
    mean ratio of the `edge_bins` outermost bins over the gain, taking the
    smaller of the two sides (negative when the data rise past the peak)."""
    sides = (bins["y"].iloc[:edge_bins], bins["y"].iloc[-edge_bins:])
    return float(min(1 - side.mean() / gain for side in sides))


class YawModel:
    """Expected power below rated given the measured misalignment."""

    def __init__(self, physics: dict[str, Any], yaw: dict[str, Any]) -> None:
        self.physics = physics
        self.yaw = yaw
        self.power_curve = pd.DataFrame(columns=["x", "y", "count"])
        self.offset: dict[str, Any] = {}

    def _v_n(self, frame: pd.DataFrame) -> pd.Series:
        rho0 = self.physics["reference_air_density"]
        return normalised_wind_speed(frame["wind_speed"], frame["air_density"], rho0)

    def in_range(self, frame: pd.DataFrame) -> pd.Series:
        """Rows inside the fitted wind speed range."""
        low, high = self.yaw["wind_speed_range"]
        return self._v_n(frame).between(low, high)

    def ratio(self, frame: pd.DataFrame) -> pd.Series:
        """Power over the training power curve; NaN outside the range."""
        expected = interpolate(self.power_curve, self._v_n(frame))
        return (frame["power"] / expected).where(self.in_range(frame))

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Power curve P(v_n), then the cosine fit of the power ratio."""
        rows = pd.concat(list(train_frames.values()), ignore_index=True)
        used = rows[self.in_range(rows)]
        self.power_curve = binned_means(
            self._v_n(used),
            used["power"],
            self.physics["bin_width"],
            self.physics["min_bin_count"],
        )
        offset = fit_offset(self.ratio(used), used["yaw_misalignment"], self.yaw)
        if offset is None:
            raise ValueError("too few misalignment bins to fit the yaw model")
        self.offset = offset
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected power in kW; NaN outside the wind speed range."""
        curve = interpolate(self.power_curve, self._v_n(frame))
        loss = cosine_power_ratio(
            frame["yaw_misalignment"].to_numpy(float),
            self.offset["gain"],
            self.offset["offset_deg"],
            self.offset["exponent"],
        )
        return np.where(self.in_range(frame), curve * loss, np.nan)

    def to_dict(self) -> dict[str, Any]:
        """The fitted power curve and cosine fit, for saving."""
        return self.offset | {
            "wind_speed_normalised": self.power_curve["x"].tolist(),
            "power_kw": self.power_curve["y"].tolist(),
        }


def offset_windows(
    frame: pd.DataFrame, model: YawModel, yaw: dict[str, Any]
) -> pd.DataFrame:
    """Offset fitted over rolling windows of `window_days`, every
    `step_days`; only full windows, and only those with enough bins."""
    window = pd.Timedelta(days=yaw["window_days"])
    first, last = frame["timestamp"].min().floor("D"), frame["timestamp"].max()
    starts = pd.date_range(first, last - window, freq=f"{yaw['step_days']}D")
    ratio = model.ratio(frame)
    rows = []
    for start in starts:
        inside = frame["timestamp"].between(start, start + window, inclusive="left")
        fit = fit_offset(ratio[inside], frame.loc[inside, "yaw_misalignment"], yaw)
        if fit is not None:
            rows.append({"window_start": start, "window_end": start + window} | fit)
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def train_turbine(
    g: Graph,
    config: dict[str, Any],
    code: str,
    turbine_id: str,
    path: Path,
    rated_power_kw: float,
    models_dir: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Fit, score and register one turbine's yaw model; return its metrics
    record and offset windows."""
    spec, yaw = config["models"]["yaw"], config["yaw"]
    frame = pd.read_parquet(path)
    train, test = split_masks(frame, config["train_test_split"]["train_end"], REQUIRED)
    model = YawModel(config["physics"], yaw).fit({turbine_id: frame[train]})
    model_file = models_dir / f"{model_id_of(spec, code, turbine_id)}.json"
    model_file.write_text(json.dumps(model.to_dict(), indent=2))
    scored = test & model.in_range(frame)
    fitted = {k: model.offset[k] for k in ("offset_deg", "exponent", "peak_drop")}
    metrics = (
        evaluate(
            frame.loc[scored, "power"],
            model.predict(frame)[scored],
            rated_power_kw,
        )
        | {"n_train": int(train.sum()), "n_test": int(scored.sum())}
        | fitted
        | {"identified": model.offset["identified"]}
    )
    model_id, entry = register_model(g, spec, code, turbine_id, metrics, model_file)
    status = "identified" if metrics["identified"] else "NOT identified"
    print(
        f"{code} {turbine_id} yaw: offset {metrics['offset_deg']:+.1f} deg, "
        f"k {metrics['exponent']:.2f}, peak drop {metrics['peak_drop']:.1%} "
        f"({status}); nRMSE {metrics['nrmse']:.2%}"
    )
    usable = frame[REQUIRED].notna().all(axis=1) & frame["is_normal"]
    windows = offset_windows(frame[usable], model, yaw)
    return {model_id: entry}, windows.assign(turbine_id=turbine_id)


def train_farm(
    g: Graph, config: dict[str, Any], farm: str
) -> dict[str, dict[str, Any]]:
    """Yaw models for every turbine of a farm; writes yaw_offsets.parquet."""
    code = config["farms"][farm]["code"]
    specs = get_turbines(g, code).set_index("turbine_id")
    models_dir = Path(config["paths"]["models"]) / farm
    models_dir.mkdir(parents=True, exist_ok=True)
    entries, windows = {}, []
    for tid, path in turbine_paths(g, code).items():
        rated = specs.loc[tid, "rated_power_kw"]
        entry, turbine_windows = train_turbine(
            g, config, code, tid, path, rated, models_dir
        )
        entries |= entry
        windows.append(turbine_windows)
    out = farm_processed_dir(config, farm) / "yaw_offsets.parquet"
    pd.concat(windows, ignore_index=True).to_parquet(out, index=False)
    logger.info("Wrote %s", out)
    return entries


def main() -> None:
    """Train yaw models for every farm."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    ontology = config["paths"]["ontology"]
    g = load_graph(ontology)
    entries = {}
    for farm in farm_names(config):
        entries |= train_farm(g, config, farm)
    save_graph(g, ontology)
    merge_metrics(Path(config["paths"]["models"]) / "metrics.json", entries)
    logger.info("Registered %d yaw models in %s", len(entries), ontology)


if __name__ == "__main__":
    main()
