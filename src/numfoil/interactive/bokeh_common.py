from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

import torch


@dataclass
class OverlayState:
    camber: bool = False
    thickness: bool = False
    max_thickness: bool = False
    max_camber: bool = False
    le_radius: bool = False


def cosine_x(n: int, *, dtype=torch.float32, device: torch.device | str = "cpu") -> torch.Tensor:
    # Avoid depending on numfoil.util here; this is enough for plotting.
    beta = torch.linspace(0.0, torch.pi, n, dtype=dtype, device=device)
    return 0.5 * (1.0 - torch.cos(beta))


def selig_full_coordinates(x: torch.Tensor, y_u: torch.Tensor, y_l: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return x,y in standard TE->LE->TE ordering for plotting."""
    # x: [N], y_u/y_l: [N]
    x_full = torch.cat([x.flip(0), x[1:]])
    y_full = torch.cat([y_u.flip(0), y_l[1:]])
    return x_full.detach().cpu().numpy(), y_full.detach().cpu().numpy()


def camber_thickness(x: torch.Tensor, y_u: torch.Tensor, y_l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    camber = 0.5 * (y_u + y_l)
    thickness = y_u - y_l
    return camber, thickness


def max_of_curve(x: torch.Tensor, y: torch.Tensor, *, use_abs: bool = False) -> Tuple[float, float]:
    yy = torch.abs(y) if use_abs else y
    idx = int(torch.argmax(yy).item())
    return float(x[idx].item()), float(y[idx].item())


def estimate_le_radius_from_second_derivative(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    x_eval: float = 0.01,
) -> float:
    """Cheap-ish LE radius estimate via curvature at a small x.

    Uses finite differences on a dense sampled y(x) curve.
    This is intentionally approximate (fast for interactive use).
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # pick nearest index to x_eval
    i = int(np.argmin(np.abs(x_np - x_eval)))
    i = int(np.clip(i, 2, len(x_np) - 3))

    # 5-point stencil for derivatives (uniform-ish spacing assumption)
    # Note: cosine spacing is non-uniform, but locally it works well enough.
    dx = x_np[i + 1] - x_np[i]
    if dx == 0:
        return float("nan")

    # first derivative
    dy = (y_np[i + 1] - y_np[i - 1]) / (x_np[i + 1] - x_np[i - 1])
    # second derivative
    d2y = (y_np[i + 1] - 2 * y_np[i] + y_np[i - 1]) / (dx * dx)

    kappa = d2y / ((1.0 + dy * dy) ** 1.5)
    if kappa == 0:
        return float("inf")
    return float(1.0 / abs(kappa))


@torch.no_grad()
def to_float_tensor(v: float, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(float(v), device=device, dtype=dtype)


@torch.no_grad()
def to_1d_tensor(values: Iterable[float], *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(list(values), device=device, dtype=dtype)
