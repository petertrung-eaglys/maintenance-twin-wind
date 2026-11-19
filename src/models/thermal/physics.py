"""Grey-box thermal model (spec §7.9, extended; README "Thermal models").

A first-order lumped heat balance per turbine and temperature, per
10-minute step (P in MW, w = rotor speed / scale):

    T[k+1] = T[k] + c + a P + b P^2 - (h0 + h1 w) (T[k] - T_nac)
                  - h2 max(T[k] - T_set, 0)

c is heat at standstill, a P + b P^2 the losses, h0 + h1 w the cooling
(the forced-convection part grows with rotor speed) and h2 a cooler that
works above the set point T_set (oil cooler, fans); h0, h1, h2 > 0. The
parameters are fitted on the error of day-long free-run simulations, all
turbines at once on the GPU (each turbine has its own parameters). The
prediction is a free run from the inputs: it starts from the measured
temperature after each gap and is hidden for `warmup_steps`.
"""

import logging
from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softplus

logger = logging.getLogger(__name__)

PARAMETERS = ("c", "a", "b", "h0", "h1", "h2", "set_point")
# Raw start values of the heat balance (also the PINN's ODE parameters);
# h = softplus(-3) ~ 0.05 per step (time constant ~3 h).
INITIAL_RAW = (0.0, 0.0, 0.0, -3.0, -3.0)
# The cooler starts nearly off (softplus(-5) ~ 0.007) with its set point at
# the turbine's median training temperature.
INITIAL_COOLER_RAW = -5.0
NO_COOLER = (0.0, float("inf"))
POOLED = "pooled"  # parameter key of a pooled model


def device_of(thermal: dict[str, Any]) -> torch.device:
    """`thermal.device`: auto picks cuda when available."""
    name = thermal["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def scaled_inputs(
    frame: pd.DataFrame, thermal: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power in MW (negative clipped), scaled rotor speed, reference temp."""
    power = frame["power"].clip(lower=0) / thermal["power_scale_kw"]
    speed = frame["rotor_speed"] / thermal["rotor_speed_scale_rpm"]
    return (
        power.to_numpy(float),
        speed.to_numpy(float),
        frame[thermal["reference"]].to_numpy(float),
    )


def coefficients(
    frame: pd.DataFrame, params: Mapping[str, float], thermal: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """The heat balance as T[k+1] = alpha[k] T[k] + beta[k]."""
    power, speed, reference = scaled_inputs(frame, thermal)
    cooling = params["h0"] + params["h1"] * speed
    heat = params["c"] + params["a"] * power + params["b"] * power**2
    return 1 - cooling, heat + cooling * reference


def cooler(params: Mapping[str, float]) -> tuple[float, float]:
    """(h2, set point) of the cooler; no cooler when the parameters have
    none (models saved before it was added)."""
    return params.get("h2", NO_COOLER[0]), params.get("set_point", NO_COOLER[1])


def steady_state(
    params: Mapping[str, float], power_mw: float, speed: float, reference: float
) -> float:
    """Temperature where heating and cooling balance."""
    heat = params["c"] + params["a"] * power_mw + params["b"] * power_mw**2
    cooling = params["h0"] + params["h1"] * speed
    uncooled = reference + heat / cooling
    gain, set_point = cooler(params)
    if uncooled <= set_point:
        return uncooled
    return (heat + cooling * reference + gain * set_point) / (cooling + gain)


def simulate(
    alpha: np.ndarray,
    beta: np.ndarray,
    measured: np.ndarray,
    restart: np.ndarray,
    warmup: int,
    cooler: tuple[float, float] = NO_COOLER,
) -> np.ndarray:
    """Free run of T[k+1] = alpha[k] T[k] + beta[k] - h2 max(T[k] - T_set, 0)
    with cooler = (h2, T_set). The state restarts from the measurement where
    `restart` is set or after missing inputs (NaN state); the first `warmup`
    steps after a restart are NaN."""
    gain, set_point = cooler
    out = []
    temp, steps = np.nan, 0
    rows = zip(
        alpha.tolist(), beta.tolist(), measured.tolist(), restart.tolist(), strict=True
    )
    for a, b, m, r in rows:
        if r or temp != temp:  # temp != temp: NaN
            temp, steps = m, 0
        out.append(temp if steps >= warmup else np.nan)
        steps += 1
        excess = temp - set_point
        temp = a * temp + b - (gain * excess if excess > 0 else 0.0)
    return np.array(out)


def restarts(timestamps: pd.Series, interval: pd.Timedelta) -> np.ndarray:
    """True where the previous row is not exactly one interval earlier."""
    return (timestamps.diff() != interval).to_numpy()


def sequences(
    frame: pd.DataFrame,
    target: str,
    thermal: dict[str, Any],
    interval: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """Consecutive day-long training sequences of one turbine: arrays of
    shape (n, steps) for power, speed, reference, temp and the loss mask
    (normal rows with a measured temperature). Kept: contiguous sequences
    with all inputs and a measured start temperature."""
    steps = thermal["physics"]["sequence_steps"]
    n = len(frame) // steps * steps
    rows = frame.iloc[:n]
    power, speed, reference = (
        a.reshape(-1, steps) for a in scaled_inputs(rows, thermal)
    )
    temp = rows[target].to_numpy(float).reshape(-1, steps)
    gaps = restarts(rows["timestamp"], interval).reshape(-1, steps)[:, 1:]
    valid = (
        np.isfinite(power).all(1)
        & np.isfinite(speed).all(1)
        & np.isfinite(reference).all(1)
        & np.isfinite(temp[:, 0])
        & ~gaps.any(1)
    )
    mask = rows["is_normal"].to_numpy(bool).reshape(-1, steps) & np.isfinite(temp)
    return {
        "power": power[valid],
        "speed": speed[valid],
        "reference": reference[valid],
        "temp": np.nan_to_num(temp[valid]),
        "mask": mask[valid],
    }


def rollout(
    theta: torch.Tensor,
    power: torch.Tensor,
    speed: torch.Tensor,
    reference: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    """Free-run simulation of every sequence; theta holds the raw
    parameters of each sequence's turbine, shape (n, 7): c, a, b, raw h0,
    raw h1, raw h2, set point."""
    c, a, b = theta[:, 0], theta[:, 1], theta[:, 2]
    h0, h1, h2 = (softplus(theta[:, i]) for i in (3, 4, 5))
    set_point = theta[:, 6]
    temp, out = start, [start]
    for k in range(power.shape[1] - 1):
        p = power[:, k]
        cooling = h0 + h1 * speed[:, k]
        cooled = h2 * torch.relu(temp - set_point)
        change = c + a * p + b * p**2 - cooling * (temp - reference[:, k]) - cooled
        temp = temp + change
        out.append(temp)
    return torch.stack(out, dim=1)


def physical(raw: torch.Tensor) -> dict[str, float]:
    """Raw parameters to c, a, b, h0, h1, h2, set point (h through
    softplus)."""
    values = [*raw[:3].tolist(), *softplus(raw[3:6]).tolist(), raw[6].item()]
    return dict(zip(PARAMETERS, values, strict=True))


class ThermalPhysicsModel:
    """Heat-balance parameters per turbine (keyed by turbine key) for one
    temperature target. With `pooled`, one parameter set is fitted on all
    turbines and used for any turbine, also unseen ones (cross-site)."""

    def __init__(
        self,
        target: str,
        thermal: dict[str, Any],
        interval_minutes: int,
        seed: int = 0,
        pooled: bool = False,
    ) -> None:
        self.target = target
        self.thermal = thermal
        self.interval = pd.Timedelta(minutes=interval_minutes)
        self.seed = seed
        self.pooled = pooled
        self.params: dict[str, dict[str, float]] = {}

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Fit every turbine's parameters on its training-period frame
        (contiguous rows; the loss uses normal rows only)."""
        torch.manual_seed(self.seed)
        device = device_of(self.thermal)
        keys = [POOLED] if self.pooled else list(train_frames)
        parts = [
            sequences(f, self.target, self.thermal, self.interval)
            for f in train_frames.values()
        ]
        owner = np.concatenate(
            [
                np.full(len(p["temp"]), 0 if self.pooled else i)
                for i, p in enumerate(parts)
            ]
        )
        data = {
            name: torch.tensor(
                np.concatenate([p[name] for p in parts]),
                device=device,
                dtype=torch.bool if name == "mask" else torch.float32,
            )
            for name in parts[0]
        }
        index = torch.tensor(owner, device=device)
        temps = [f[self.target] for f in train_frames.values()]
        if self.pooled:
            temps = [pd.concat(temps)]
        set_points = [float(np.nanmedian(s)) if s.notna().any() else 0.0 for s in temps]
        theta = torch.tensor(
            [[*INITIAL_RAW, INITIAL_COOLER_RAW, s] for s in set_points],
            device=device,
            requires_grad=True,
        )
        physics = self.thermal["physics"]
        optimiser = torch.optim.Adam([theta], lr=physics["learning_rate"])
        mask = data["mask"].float()
        generator = torch.Generator(device=device).manual_seed(self.seed)
        batch = min(physics["batch_sequences"], len(owner))
        for _ in range(physics["iterations"]):
            optimiser.zero_grad()
            rows = torch.randint(
                len(owner), (batch,), device=device, generator=generator
            )
            predicted = rollout(
                theta[index[rows]],
                data["power"][rows],
                data["speed"][rows],
                data["reference"][rows],
                data["temp"][rows, 0],
            )
            weight = mask[rows]
            loss = ((predicted - data["temp"][rows]) ** 2 * weight).sum() / weight.sum()
            loss.backward()
            optimiser.step()
        logger.info(
            "%s physics: %d sequences, train RMSE %.2f",
            self.target,
            len(owner),
            loss.item() ** 0.5,
        )
        self.params = {k: physical(theta[i].detach().cpu()) for i, k in enumerate(keys)}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Free-run temperature for one turbine's frame (sorted by time)."""
        keys = frame["turbine_key"].unique()
        if len(keys) != 1:
            raise ValueError(f"expected one turbine per frame, got {keys}")
        params = self.params[POOLED if self.pooled else keys[0]]
        alpha, beta = coefficients(frame, params, self.thermal)
        return simulate(
            alpha,
            beta,
            frame[self.target].to_numpy(float),
            restarts(frame["timestamp"], self.interval),
            self.thermal["physics"]["warmup_steps"],
            cooler(params),
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Parameters per turbine key, for saving."""
        return self.params
