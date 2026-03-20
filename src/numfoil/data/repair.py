from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional torch dependency
    torch = None

if TYPE_CHECKING:
    import torch as _torch


def repair_negative_thickness_points(
    upper_points: np.ndarray,
    lower_points: np.ndarray,
    min_thickness: float = 0.0,
    max_iterations: int = 10,
    locality_sigma: float = 0.02,
    safety_margin: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy version of local negative-thickness point-cloud repair.

    Applies local Gaussian separation bumps at the most negative-thickness
    locations until all sampled points satisfy the requested minimum thickness
    or the iteration budget is exhausted.

    Args:
        upper_points: Upper surface points, shape [B, N, 2].
        lower_points: Lower surface points, shape [B, N, 2].
        min_thickness: Minimum required thickness.
        max_iterations: Maximum local repair iterations.
        locality_sigma: Width of local Gaussian separation bump.
        safety_margin: Extra margin above min_thickness.

    Returns:
        tuple[np.ndarray, np.ndarray]: Repaired upper and lower points.
    """
    if min_thickness < 0:
        raise ValueError("min_thickness must be >= 0.")

    upper_points = np.asarray(upper_points)
    lower_points = np.asarray(lower_points)

    if upper_points.ndim != 3 or lower_points.ndim != 3:
        raise ValueError(
            "upper_points and lower_points must both have shape [B, N, 2]."
        )

    dtype = np.result_type(upper_points.dtype, lower_points.dtype, np.float32)
    upper_points = upper_points.astype(dtype, copy=False)
    lower_points = lower_points.astype(dtype, copy=False)

    x_upper = upper_points[..., 0]
    x_lower = lower_points[..., 0]
    y_u_adj = upper_points[..., 1].copy()
    y_l_adj = lower_points[..., 1].copy()

    if not np.allclose(x_upper, x_lower, atol=1e-7, rtol=1e-5):
        raise ValueError(
            "upper_points and lower_points must share the same x-grid for repair."
        )

    eps = np.finfo(dtype).eps
    envelope = np.clip(x_upper * (1.0 - x_upper), a_min=0.0, a_max=None)

    for _ in range(max_iterations):
        t = y_u_adj - y_l_adj
        idx_min = np.argmin(t, axis=-1)
        t_min = t[np.arange(t.shape[0]), idx_min]
        needs_fix = t_min < min_thickness

        if not np.any(needs_fix):
            break

        bad_rows = np.where(needs_fix)[0]
        for b in bad_rows.tolist():
            x0 = x_upper[b, idx_min[b]]
            deficit = max(min_thickness - t_min[b] + safety_margin, 0.0)

            bump = np.exp(-0.5 * ((x_upper[b] - x0) / locality_sigma) ** 2)
            bump = bump * envelope[b]
            bump = bump / (bump.max() + eps)

            delta = 0.5 * deficit * bump
            y_u_adj[b] = y_u_adj[b] + delta
            y_l_adj[b] = y_l_adj[b] - delta
    else:
        raise ValueError(
            "Unable to enforce positive thickness within max_iterations."
        )

    repaired_upper = upper_points.copy()
    repaired_lower = lower_points.copy()
    repaired_upper[..., 1] = y_u_adj
    repaired_lower[..., 1] = y_l_adj
    return repaired_upper, repaired_lower


def repair_negative_thickness_points_torch(
    upper_points: "_torch.Tensor",
    lower_points: "_torch.Tensor",
    min_thickness: float = 0.0,
    max_iterations: int = 10,
    locality_sigma: float = 0.02,
    safety_margin: float = 1e-6,
) -> tuple["_torch.Tensor", "_torch.Tensor"]:
    """Repair local surface intersections directly in point-cloud space.

    Applies local Gaussian separation bumps at the most negative-thickness
    locations until all sampled points satisfy the requested minimum thickness
    or the iteration budget is exhausted.

    Args:
        upper_points: Upper surface points, shape [B, N, 2].
        lower_points: Lower surface points, shape [B, N, 2].
        min_thickness: Minimum required thickness.
        max_iterations: Maximum local repair iterations.
        locality_sigma: Width of local Gaussian separation bump.
        safety_margin: Extra margin above min_thickness.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Repaired upper and lower points.
    """
    if torch is None:
        raise ImportError(
            "repair_negative_thickness_points requires torch. "
            "Use repair_negative_thickness_points_numpy for NumPy arrays."
        )

    if min_thickness < 0:
        raise ValueError("min_thickness must be >= 0.")

    if upper_points.ndim != 3 or lower_points.ndim != 3:
        raise ValueError(
            "upper_points and lower_points must both have shape [B, N, 2]."
        )

    x_upper = upper_points[..., 0]
    x_lower = lower_points[..., 0]
    y_u_adj = upper_points[..., 1].clone()
    y_l_adj = lower_points[..., 1].clone()

    if not torch.allclose(x_upper, x_lower, atol=1e-7, rtol=1e-5):
        raise ValueError(
            "upper_points and lower_points must share the same x-grid for repair."
        )

    dtype = y_u_adj.dtype
    eps = torch.finfo(dtype).eps
    envelope = (x_upper * (1.0 - x_upper)).clamp_min(0.0)

    for _ in range(max_iterations):
        t = y_u_adj - y_l_adj
        t_min, idx_min = torch.min(t, dim=-1)
        needs_fix = t_min < min_thickness

        if not torch.any(needs_fix):
            break

        bad_rows = torch.where(needs_fix)[0]
        for b in bad_rows.tolist():
            x0 = x_upper[b, idx_min[b]]
            deficit = (min_thickness - t_min[b] + safety_margin).clamp_min(0.0)

            bump = torch.exp(-0.5 * ((x_upper[b] - x0) / locality_sigma) ** 2)
            bump = bump * envelope[b]
            bump = bump / (bump.max() + eps)

            delta = 0.5 * deficit * bump
            y_u_adj[b] = y_u_adj[b] + delta
            y_l_adj[b] = y_l_adj[b] - delta
    else:
        raise ValueError(
            "Unable to enforce positive thickness within max_iterations."
        )

    repaired_upper = upper_points.clone()
    repaired_lower = lower_points.clone()
    repaired_upper[..., 1] = y_u_adj
    repaired_lower[..., 1] = y_l_adj
    return repaired_upper, repaired_lower



