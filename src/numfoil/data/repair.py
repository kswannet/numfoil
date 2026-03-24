from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import numpy as np
import warnings

try:
    import torch
except ImportError:  # pragma: no cover - optional torch dependency
    torch = None

if TYPE_CHECKING:
    import torch as _torch


WeightingMode = Literal["symmetric", "edge_anchored"]

# TODO symmetric one does not work !!!
def repair_negative_thickness_points(
    upper_points: np.ndarray,
    lower_points: np.ndarray,
    min_thickness: float = 0.0,
    max_iterations: int = 10,
    locality_sigma: float = 0.02,
    weighting: WeightingMode = "edge_anchored",
    edge_power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair negative thickness with one centered correction (NumPy).

    For each sample, this function finds the minimum-thickness point exactly
    once, computes the required opening there, and applies a single symmetric
    correction field across the full chord. This avoids stacked local updates.

    This routine is rank-agnostic over leading dimensions: any input with shape
    ``[..., N, 2]`` is accepted and processed by flattening leading dims into a
    sample axis and reshaping back after repair.

    Args:
        upper_points (np.ndarray): Upper surface coordinates with shape
            ``[..., N, 2]``.
        lower_points (np.ndarray): Lower surface coordinates with shape
            ``[..., N, 2]`` and identical x-grid as ``upper_points``.
        min_thickness (float): Target minimum thickness. Use ``0.0`` to only
            remove overlap.
        max_iterations (int): Kept for backward compatibility. The method uses
            a single correction pass by design.
        locality_sigma (float): Gaussian width used when ``weighting`` is
            ``"symmetric"``.
        weighting (WeightingMode):
            - ``"symmetric"``: mirrored Gaussian decay around the critical
              point (can alter LE/TE).
            - ``"edge_anchored"``: piecewise decay that enforces zero weight at
              LE and TE, preserving original LE/TE thickness.
        edge_power (float): Exponent controlling sharpness of
            ``"edge_anchored"`` decay.

    Returns:
        tuple[np.ndarray, np.ndarray]: Repaired ``(upper_points, lower_points)``
            with unchanged x-coordinates and repaired y-coordinates.

        Notes:
                - If the minimum-thickness index is at the first or last chord point
                    and ``min_thickness == 0``, no correction is applied. This preserves
                    valid LE/closed-TE contact.
                - ``weighting="edge_anchored"`` preserves LE and TE thickness exactly
                    by forcing weights to zero at both ends.
    """
    if min_thickness < 0.0:
        raise ValueError("min_thickness must be >= 0.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")
    if locality_sigma <= 0.0:
        raise ValueError("locality_sigma must be > 0.")
    if edge_power <= 0.0:
        raise ValueError("edge_power must be > 0.")

    # Normalize/validate input geometry.
    upper = np.asarray(upper_points)
    lower = np.asarray(lower_points)
    if upper.shape != lower.shape:
        raise ValueError("upper_points and lower_points must have identical shape.")
    if upper.ndim < 2 or upper.shape[-1] != 2:
        raise ValueError("Expected shape [..., N, 2] for both inputs.")

    dtype = np.result_type(upper.dtype, lower.dtype, np.float64)
    upper = upper.astype(dtype, copy=False)
    lower = lower.astype(dtype, copy=False)

    x_upper = upper[..., 0]
    x_lower = lower[..., 0]
    if not np.allclose(x_upper, x_lower, atol=1e-8, rtol=1e-6):
        raise ValueError("upper_points and lower_points must share x-coordinates.")

    lead_shape = x_upper.shape[:-1]
    n_points = x_upper.shape[-1]
    n_samples = int(np.prod(lead_shape)) if lead_shape else 1

    # Flatten any leading dimensions to one sample axis for vectorized math.
    x = x_upper.reshape(n_samples, n_points)
    y_u = upper[..., 1].reshape(n_samples, n_points).copy()
    y_l = lower[..., 1].reshape(n_samples, n_points).copy()

    row_idx = np.arange(n_samples)
    eps = np.finfo(dtype).eps

    # Single-pass correction: find worst thickness once, then apply one field.
    thickness = y_u - y_l
    idx_min = np.argmin(thickness, axis=-1)
    t_min = thickness[row_idx, idx_min]
    x0 = x[row_idx, idx_min]

    # Do not "repair" valid LE/TE contact when target minimum is zero.
    endpoint_contact = (idx_min == 0) | (idx_min == (n_points - 1))
    needs_fix = t_min < (min_thickness - eps)
    if min_thickness == 0.0:
        needs_fix = needs_fix & (~endpoint_contact)

    # Required symmetric correction at x0.
    delta = 0.5 * np.clip(min_thickness - t_min, a_min=0.0, a_max=None)
    delta = np.where(needs_fix, delta, 0.0)

    # Build chordwise weights with w(x0)=1.
    if weighting == "symmetric":
        weights = np.exp(-0.5 * ((x - x0[:, None]) / locality_sigma) ** 2)
        weights /= np.maximum(weights.max(axis=-1, keepdims=True), eps)
    elif weighting == "edge_anchored":
        left_span = np.maximum(x0, eps)
        right_span = np.maximum(1.0 - x0, eps)
        dx = x - x0[:, None]
        left = dx <= 0.0
        right = ~left

        weights = np.zeros_like(x)
        d_left = np.abs(dx) / left_span[:, None]
        d_right = np.abs(dx) / right_span[:, None]
        weights[left] = np.clip(1.0 - d_left[left], 0.0, 1.0) ** edge_power
        weights[right] = np.clip(1.0 - d_right[right], 0.0, 1.0) ** edge_power
    else:
        raise ValueError(f"Unknown weighting mode: {weighting}")

    # Apply symmetric opening once: upper up, lower down.
    y_u += delta[:, None] * weights
    y_l -= delta[:, None] * weights

    repaired_upper = upper.copy()
    repaired_lower = lower.copy()
    repaired_upper[..., 1] = y_u.reshape(lead_shape + (n_points,))
    repaired_lower[..., 1] = y_l.reshape(lead_shape + (n_points,))
    return repaired_upper, repaired_lower


def repair_negative_thickness_points_torch(
    upper_points: "_torch.Tensor",
    lower_points: "_torch.Tensor",
    min_thickness: float = 1e-3,
    weighting: str = "edge_anchored",
    locality_sigma: float = 0.1,
    max_iterations: int = 10,
    edge_power: float = 2.0,
) -> tuple["_torch.Tensor", "_torch.Tensor"]:
    """Repair negative thickness with one centered correction (PyTorch).

    Torch counterpart of :func:`repair_negative_thickness_points` with identical
    behavior and parameters, operating on tensors shaped ``[..., N, 2]``.

    Args:
        upper_points (_torch.Tensor): Upper surface coordinates with shape
            ``[..., N, 2]``.
        lower_points (_torch.Tensor): Lower surface coordinates with shape
            ``[..., N, 2]`` and identical x-grid as ``upper_points``.
        min_thickness (float): Target minimum thickness. Use ``0.0`` to only
            remove overlap.
        weighting (WeightingMode):
            - ``"symmetric"``: mirrored Gaussian decay around the critical
                point (can alter LE/TE).
            - ``"edge_anchored"``: piecewise decay that enforces zero weight at
                LE and TE, preserving original LE/TE thickness.
        locality_sigma (float): Standard deviation for the Gaussian weighting function.
        max_iterations (int): Kept for backward compatibility. The method uses
            a single correction pass by design.
        edge_power (float): Exponent controlling sharpness of
            ``"edge_anchored"`` decay.

    Returns:
        tuple[_torch.Tensor, _torch.Tensor]: Repaired
            ``(upper_points, lower_points)`` with unchanged x-coordinates.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is not installed. Use repair_negative_thickness_points "
            "for NumPy arrays."
        )

    if min_thickness < 0.0:
        raise ValueError("min_thickness must be >= 0.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")
    if edge_power <= 0.0:
        raise ValueError("edge_power must be > 0.")

    # Validate shape and align dtype for stable numerical operations.
    if upper_points.shape != lower_points.shape:
        raise ValueError("upper_points and lower_points must have identical shape.")
    if upper_points.ndim < 2 or upper_points.shape[-1] != 2:
        raise ValueError("Expected shape [..., N, 2] for both inputs.")
    if locality_sigma <= 0.0:
        raise ValueError("locality_sigma must be > 0.")
    if 0.15 > locality_sigma < 0.3:
        warnings.warn(
            "Low locality_sigma values (<0.15) may cause excessive correction localized corrections, "
            "leading to a 'bump' in the geometry. High values (>0.3) may excessively effect global geometry."
        )
    if weighting not in ("symmetric", "edge_anchored"):
        raise ValueError(f"Unknown weighting mode: {weighting}")

    dtype = torch.promote_types(upper_points.dtype, lower_points.dtype)
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32

    upper = upper_points.to(dtype=dtype)
    lower = lower_points.to(dtype=dtype)

    x_upper = upper[..., 0]
    x_lower = lower[..., 0]
    if not torch.allclose(x_upper, x_lower, atol=1e-8, rtol=1e-6):
        raise ValueError("upper_points and lower_points must share x-coordinates.")

    lead_shape = x_upper.shape[:-1]
    n_points = x_upper.shape[-1]
    n_samples = int(np.prod(lead_shape)) if lead_shape else 1

    # Flatten leading dims for vectorized sample-wise updates.
    x = x_upper.reshape(n_samples, n_points)
    y_u = upper[..., 1].reshape(n_samples, n_points).clone()
    y_l = lower[..., 1].reshape(n_samples, n_points).clone()

    row_idx = torch.arange(n_samples, device=x.device)
    eps = torch.finfo(dtype).eps

    # Single-pass correction: find worst thickness once, then apply one field.
    thickness = y_u - y_l
    t_min, idx_min = torch.min(thickness, dim=-1)
    x0 = x[row_idx, idx_min]

    endpoint_contact = (idx_min == 0) | (idx_min == (n_points - 1))
    needs_fix = t_min < (min_thickness - eps)
    if min_thickness == 0.0:
        needs_fix = needs_fix & (~endpoint_contact)

    delta = 0.5 * (min_thickness - t_min).clamp_min(0.0)
    delta = torch.where(needs_fix, delta, torch.zeros_like(delta))

    # Build chordwise weights with w(x0)=1.
    if weighting == "symmetric":
        weights = torch.exp(-0.5 * ((x - x0.unsqueeze(-1)) / locality_sigma) ** 2)
        weights = weights / weights.max(dim=-1, keepdim=True).values.clamp_min(eps)
    elif weighting == "edge_anchored":
        left_span = x0.clamp_min(eps)
        right_span = (1.0 - x0).clamp_min(eps)
        dx = x - x0.unsqueeze(-1)
        left = dx <= 0.0
        right = ~left

        weights = torch.zeros_like(x)
        d_left = dx.abs() / left_span.unsqueeze(-1)
        d_right = dx.abs() / right_span.unsqueeze(-1)
        weights[left] = (1.0 - d_left[left]).clamp_min(0.0).pow(edge_power)
        weights[right] = (1.0 - d_right[right]).clamp_min(0.0).pow(edge_power)
    else:
        raise ValueError(f"Unknown weighting mode: {weighting}")

    # Symmetric opening once: upper up, lower down.
    y_u = y_u + delta.unsqueeze(-1) * weights
    y_l = y_l - delta.unsqueeze(-1) * weights

    repaired_upper = upper.clone()
    repaired_lower = lower.clone()
    repaired_upper[..., 1] = y_u.reshape(*lead_shape, n_points)
    repaired_lower[..., 1] = y_l.reshape(*lead_shape, n_points)
    return repaired_upper, repaired_lower

