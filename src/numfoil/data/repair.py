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
    """Repair negative local thickness for point clouds using vectorized NumPy updates.

    This routine accepts arbitrary leading dimensions and interprets the last two
    axes as `[N, 2]` point clouds (x,y). It iteratively applies a local Gaussian
    bump around the most critical x-location of each sample until all samples
    satisfy `thickness >= min_thickness` or `max_iterations` is reached.

    Args:
        upper_points (np.ndarray): Upper surface points, shape `[..., N, 2]`.
        lower_points (np.ndarray): Lower surface points, shape `[..., N, 2]`.
        min_thickness (float): Required minimum thickness.
        max_iterations (int): Maximum repair iterations.
        locality_sigma (float): Gaussian width for local correction.
        safety_margin (float): Extra thickness margin per correction step.

    Returns:
        tuple[np.ndarray, np.ndarray]: Repaired `(upper_points, lower_points)`.

    Notes:
        - Fully vectorized across all leading dims.
        - Keeps x-coordinates unchanged; only y-values are adjusted.
    """
    if min_thickness < 0:
        raise ValueError("min_thickness must be >= 0.")
    if locality_sigma <= 0:
        raise ValueError("locality_sigma must be > 0.")

    upper_points = np.asarray(upper_points)
    lower_points = np.asarray(lower_points)

    if upper_points.shape != lower_points.shape:
        raise ValueError("upper_points and lower_points must have identical shape.")
    if upper_points.ndim < 2 or upper_points.shape[-1] != 2:
        raise ValueError("upper_points and lower_points must have shape [..., N, 2].")

    dtype = np.result_type(upper_points.dtype, lower_points.dtype, np.float32)
    upper_points = upper_points.astype(dtype, copy=False)
    lower_points = lower_points.astype(dtype, copy=False)

    x_upper = upper_points[..., 0]
    x_lower = lower_points[..., 0]
    y_u_adj = upper_points[..., 1].copy()
    y_l_adj = lower_points[..., 1].copy()

    if not np.allclose(x_upper, x_lower, atol=1e-7, rtol=1e-5):
        raise ValueError("upper_points and lower_points must share the same x-grid.")

    lead_shape = x_upper.shape[:-1]
    n_points = x_upper.shape[-1]
    n_samples = int(np.prod(lead_shape)) if lead_shape else 1

    x_flat = x_upper.reshape(n_samples, n_points)
    y_u_flat = y_u_adj.reshape(n_samples, n_points)
    y_l_flat = y_l_adj.reshape(n_samples, n_points)

    eps = np.finfo(dtype).eps
    envelope = np.clip(x_flat * (1.0 - x_flat), a_min=0.0, a_max=None)

    for _ in range(max_iterations):
        t = y_u_flat - y_l_flat  # [S, N]
        idx_min = np.argmin(t, axis=-1)  # [S]
        t_min = t[np.arange(n_samples), idx_min]  # [S]
        needs_fix = t_min < min_thickness  # [S]

        if not np.any(needs_fix):
            break

        x0 = x_flat[np.arange(n_samples), idx_min]  # [S]
        deficit = np.clip(min_thickness - t_min + safety_margin, 0.0, None)  # [S]

        bump = np.exp(-0.5 * ((x_flat - x0[:, None]) / locality_sigma) ** 2)  # [S, N]
        bump = bump * envelope
        bump = bump / (np.max(bump, axis=-1, keepdims=True) + eps)

        delta = 0.5 * deficit[:, None] * bump
        mask = needs_fix[:, None].astype(delta.dtype)
        y_u_flat += delta * mask
        y_l_flat -= delta * mask
    else:
        raise ValueError("Unable to enforce positive thickness within max_iterations.")

    repaired_upper = upper_points.copy()
    repaired_lower = lower_points.copy()
    repaired_upper[..., 1] = y_u_flat.reshape(lead_shape + (n_points,))
    repaired_lower[..., 1] = y_l_flat.reshape(lead_shape + (n_points,))
    return repaired_upper, repaired_lower


def repair_negative_thickness_points_torch(
    upper_points: "_torch.Tensor",
    lower_points: "_torch.Tensor",
    min_thickness: float = 0.0,
    max_iterations: int = 10,
    locality_sigma: float = 0.02,
    safety_margin: float = 1e-6,
) -> tuple["_torch.Tensor", "_torch.Tensor"]:
    """Repair negative local thickness for point clouds using vectorized Torch updates.

    Accepts `[..., N, 2]` tensors and applies iterative Gaussian local separation in
    fully vectorized form across all leading dimensions.

    Args:
        upper_points (_torch.Tensor): Upper surface points, shape `[..., N, 2]`.
        lower_points (_torch.Tensor): Lower surface points, shape `[..., N, 2]`.
        min_thickness (float): Required minimum thickness.
        max_iterations (int): Maximum repair iterations.
        locality_sigma (float): Gaussian width for local correction.
        safety_margin (float): Extra thickness margin per correction step.

    Returns:
        tuple[_torch.Tensor, _torch.Tensor]: Repaired `(upper_points, lower_points)`.
    """
    if torch is None:
        raise ImportError(
            "repair_negative_thickness_points_torch requires torch. "
            "Use repair_negative_thickness_points for NumPy arrays."
        )
    if min_thickness < 0:
        raise ValueError("min_thickness must be >= 0.")
    if locality_sigma <= 0:
        raise ValueError("locality_sigma must be > 0.")

    if upper_points.shape != lower_points.shape:
        raise ValueError("upper_points and lower_points must have identical shape.")
    if upper_points.ndim < 2 or upper_points.shape[-1] != 2:
        raise ValueError("upper_points and lower_points must have shape [..., N, 2].")

    x_upper = upper_points[..., 0]
    x_lower = lower_points[..., 0]
    y_u_adj = upper_points[..., 1].clone()
    y_l_adj = lower_points[..., 1].clone()

    if not torch.allclose(x_upper, x_lower, atol=1e-7, rtol=1e-5):
        raise ValueError("upper_points and lower_points must share the same x-grid.")

    lead_shape = x_upper.shape[:-1]
    n_points = x_upper.shape[-1]
    n_samples = int(np.prod(lead_shape)) if len(lead_shape) > 0 else 1

    x_flat = x_upper.reshape(n_samples, n_points)
    y_u_flat = y_u_adj.reshape(n_samples, n_points)
    y_l_flat = y_l_adj.reshape(n_samples, n_points)

    eps = torch.finfo(y_u_flat.dtype).eps
    envelope = (x_flat * (1.0 - x_flat)).clamp_min(0.0)

    for _ in range(max_iterations):
        t = y_u_flat - y_l_flat  # [S, N]
        t_min, idx_min = torch.min(t, dim=-1)  # [S], [S]
        needs_fix = t_min < min_thickness  # [S]

        if not torch.any(needs_fix):
            break

        row_idx = torch.arange(n_samples, device=x_flat.device)
        x0 = x_flat[row_idx, idx_min]  # [S]
        deficit = (min_thickness - t_min + safety_margin).clamp_min(0.0)  # [S]

        bump = torch.exp(-0.5 * ((x_flat - x0.unsqueeze(-1)) / locality_sigma) ** 2)
        bump = bump * envelope
        bump = bump / (bump.max(dim=-1, keepdim=True).values + eps)

        delta = 0.5 * deficit.unsqueeze(-1) * bump
        mask = needs_fix.unsqueeze(-1).to(delta.dtype)
        y_u_flat = y_u_flat + delta * mask
        y_l_flat = y_l_flat - delta * mask
    # else:
    #     raise ValueError("Unable to enforce positive thickness within max_iterations.")

    repaired_upper = upper_points.clone()
    repaired_lower = lower_points.clone()
    repaired_upper[..., 1] = y_u_flat.reshape(*lead_shape, n_points)
    repaired_lower[..., 1] = y_l_flat.reshape(*lead_shape, n_points)
    return repaired_upper, repaired_lower



