"""Physics-informed neural network for one temperature target (spec §7.9).

One network for all turbines (both farms): an MLP over the standardised
input features (current inputs, their rolling means, rotor diameter as the
turbine-type feature) and a learned turbine embedding predicts T. Loss:

    MSE(T_hat, T) on normal rows
    + lambda * MSE(ODE residual of T_hat, in degC per hour)

where the ODE residual compares T_hat[k+1] - T_hat[k] with the heat
balance of physics.py, with learnable parameters per turbine (h0, h1 > 0
through softplus). The ODE term uses every contiguous row pair with
inputs, stops included: the heat balance holds there too. Each step draws
the same number of rows from every turbine (per-turbine loaders, §7.6).
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import softplus

from src.models.thermal.features import normal_target_rows
from src.models.thermal.physics import (
    INITIAL_RAW,
    device_of,
    restarts,
    scaled_inputs,
)

logger = logging.getLogger(__name__)

PREDICT_CHUNK = 200_000
MINUTES_PER_HOUR = 60


def ode_residual(
    raw: torch.Tensor,
    temp: torch.Tensor,
    following: torch.Tensor,
    power: torch.Tensor,
    speed: torch.Tensor,
    reference: torch.Tensor,
    steps_per_hour: float,
) -> torch.Tensor:
    """Heat-balance residual of a temperature pair, in degC per hour; raw
    holds each row's turbine parameters (c, a, b, raw h0, raw h1)."""
    c, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
    cooling = softplus(raw[:, 3]) + softplus(raw[:, 4]) * speed
    change = c + a * power + b * power**2 - cooling * (temp - reference)
    return (following - temp - change) * steps_per_hour


class Network(nn.Module):
    """MLP over features and a turbine embedding, plus per-turbine raw
    heat-balance parameters."""

    def __init__(self, n_features: int, n_turbines: int, pinn: dict[str, Any]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_turbines, pinn["embedding_dim"])
        width = n_features + pinn["embedding_dim"]
        layers: list[nn.Module] = []
        for _ in range(pinn["layers"]):
            layers += [nn.Linear(width, pinn["hidden"]), nn.SiLU()]
            width = pinn["hidden"]
        self.mlp = nn.Sequential(*layers, nn.Linear(width, 1))
        self.ode = nn.Parameter(torch.tensor([INITIAL_RAW] * n_turbines))

    def forward(self, features: torch.Tensor, turbine: torch.Tensor) -> torch.Tensor:
        """Standardised temperature per row."""
        return self.head(features, self.embedding(turbine))

    def head(self, features: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
        """Standardised temperature from features and turbine embeddings."""
        return self.mlp(torch.cat([features, embedded], dim=1)).squeeze(-1)


class ThermalPINN:
    """Shared temperature network for every turbine key it was fitted on."""

    def __init__(
        self,
        target: str,
        thermal: dict[str, Any],
        interval_minutes: int,
        features: list[str],
        seed: int = 0,
        unseen_turbines: str = "error",
    ) -> None:
        """`unseen_turbines`: "error" (KeyError for a turbine not fitted on)
        or "mean" (use the mean turbine embedding, for cross-site use)."""
        self.target = target
        self.unseen_turbines = unseen_turbines
        self.thermal = thermal
        self.interval = pd.Timedelta(minutes=interval_minutes)
        self.steps_per_hour = MINUTES_PER_HOUR / interval_minutes
        self.features = features
        self.seed = seed
        self.device = device_of(thermal)
        self.keys: list[str] = []
        self.stats: dict[str, Any] = {}
        self.network: Network | None = None

    def _standardised(self, frame: pd.DataFrame) -> np.ndarray:
        """Standardised features as float32 (what the network uses)."""
        values = frame[self.features].to_numpy(np.float32)
        return ((values - self.stats["mean"]) / self.stats["std"]).astype(np.float32)

    def _turbine_tensors(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Features, physics inputs and the data and pair row indices."""
        x = self._standardised(frame)
        has_x = np.isfinite(x).all(1)
        temp = frame[self.target].to_numpy(float)
        power, speed, reference = scaled_inputs(frame, self.thermal)
        has_inputs = has_x & np.isfinite(power + speed + reference)
        follows = ~restarts(frame["timestamp"], self.interval)
        pair = has_inputs[:-1] & has_x[1:] & follows[1:]
        data = normal_target_rows(self.target)(frame).to_numpy() & has_x
        return {
            "x": np.nan_to_num(x),
            "temp": np.nan_to_num(
                (temp - self.stats["temp_mean"]) / self.stats["temp_std"]
            ),
            "power": np.nan_to_num(power),
            "speed": np.nan_to_num(speed),
            "reference": np.nan_to_num(reference),
            "data": np.flatnonzero(data),
            "pair": np.flatnonzero(pair),
        }

    def _fit_stats(self, frames: list[pd.DataFrame]) -> None:
        """Feature means and standard deviations over all rows, target mean
        and deviation over normal rows; from per-frame sums, so no frame is
        copied whole."""
        count = sum(f[self.features].count() for f in frames)
        mean = sum(f[self.features].sum() for f in frames) / count
        square = sum((f[self.features] ** 2).sum() for f in frames) / count
        std = np.sqrt((square - mean**2).clip(lower=0)).to_numpy(float)
        normal = pd.concat(
            [f.loc[normal_target_rows(self.target)(f), self.target] for f in frames]
        )
        self.stats = {
            "mean": mean.fillna(0).to_numpy(float),
            "std": np.where(std > 0, std, 1.0),
            "temp_mean": float(normal.mean()),
            "temp_std": float(normal.std()),
        }

    def _sampler(self, parts: list[dict[str, np.ndarray]], name: str):
        """Per-turbine sampling of global row indices: `batch` rows from
        every turbine that has such rows."""
        offsets = np.cumsum([0, *[len(p["x"]) for p in parts[:-1]]])
        rows = [p[name] + o for p, o in zip(parts, offsets, strict=True)]
        rows = [r for r in rows if len(r)]
        pool = torch.tensor(np.concatenate(rows), device=self.device)
        counts = torch.tensor([len(r) for r in rows], device=self.device)
        starts = torch.cumsum(counts, 0) - counts
        batch = self.thermal["pinn"]["batch_per_turbine"]

        def sample(generator: torch.Generator) -> torch.Tensor:
            u = torch.rand(len(rows), batch, device=self.device, generator=generator)
            return pool[(u * counts[:, None]).long() + starts[:, None]].flatten()

        return sample

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Fit the shared network on every turbine's training-period frame."""
        torch.manual_seed(self.seed)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.keys = list(train_frames)
        frames = list(train_frames.values())
        self._fit_stats(frames)
        parts = [self._turbine_tensors(f) for f in frames]
        owner = np.concatenate([np.full(len(p["x"]), i) for i, p in enumerate(parts)])
        data = {
            name: torch.tensor(
                np.concatenate([p[name] for p in parts]),
                device=self.device,
                dtype=torch.float32,
            )
            for name in ("x", "temp", "power", "speed", "reference")
        }
        turbine = torch.tensor(owner, device=self.device)
        sample_data = self._sampler(parts, "data")
        sample_pair = self._sampler(parts, "pair")
        pinn = self.thermal["pinn"]
        network = Network(len(self.features), len(self.keys), pinn).to(self.device)
        optimiser = torch.optim.Adam(network.parameters(), lr=pinn["learning_rate"])
        scale = self.stats["temp_std"]
        for _ in range(pinn["steps"]):
            optimiser.zero_grad()
            rows = sample_data(generator)
            fit_loss = (
                (network(data["x"][rows], turbine[rows]) - data["temp"][rows]) ** 2
            ).mean() * scale**2
            k = sample_pair(generator)
            now = self._degrees(network(data["x"][k], turbine[k]))
            following = self._degrees(network(data["x"][k + 1], turbine[k]))
            residual = ode_residual(
                network.ode[turbine[k]],
                now,
                following,
                data["power"][k],
                data["speed"][k],
                data["reference"][k],
                self.steps_per_hour,
            )
            loss = fit_loss + pinn["ode_weight"] * (residual**2).mean()
            loss.backward()
            optimiser.step()
        logger.info(
            "%s PINN: fit RMSE %.2f, ODE residual RMS %.2f degC/h",
            self.target,
            fit_loss.item() ** 0.5,
            (residual**2).mean().item() ** 0.5,
        )
        self.network = network.eval()
        return self

    def _degrees(self, standardised: torch.Tensor) -> torch.Tensor:
        return standardised * self.stats["temp_std"] + self.stats["temp_mean"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected temperature per row; NaN where a feature is missing."""
        keys = frame["turbine_key"].unique()
        if len(keys) != 1:
            raise ValueError(f"expected one turbine per frame, got {keys}")
        embedded = self._embedding(keys[0])
        x = self._standardised(frame)
        rows = np.flatnonzero(np.isfinite(x).all(1))
        out = np.full(len(frame), np.nan)
        with torch.no_grad():
            for start in range(0, len(rows), PREDICT_CHUNK):
                chunk = rows[start : start + PREDICT_CHUNK]
                features = torch.tensor(
                    x[chunk], device=self.device, dtype=torch.float32
                )
                standardised = self.network.head(
                    features, embedded.expand(len(chunk), -1)
                )
                out[chunk] = self._degrees(standardised).cpu().numpy()
        return out

    def _embedding(self, key: str) -> torch.Tensor:
        """The turbine's learned embedding (1 x dim); for an unseen turbine
        the mean embedding if allowed, else KeyError."""
        weights = self.network.embedding.weight
        if key in self.keys:
            return weights[self.keys.index(key)].unsqueeze(0)
        if self.unseen_turbines == "mean":
            return weights.mean(dim=0, keepdim=True)
        raise KeyError(key)

    def save(self, path: Path) -> None:
        """Network weights, turbine keys, features and standardisation."""
        torch.save(
            {
                "state": self.network.state_dict(),
                "keys": self.keys,
                "features": self.features,
                "stats": self.stats,
            },
            path,
        )
