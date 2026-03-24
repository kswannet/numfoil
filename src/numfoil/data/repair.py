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

# TODO: probably just remove the "edge_anchored" option, likely inferior to smooth version


def repair_negative_thickness_points(
    upper_points: np.ndarray,
    lower_points: np.ndarray,
    min_thickness: float = 0.0,
    weighting: WeightingMode = "edge_anchored",
    locality_sigma: float = 0.3,
    edge_power: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair negative thickness with one centered correction (NumPy).

    This routine finds the minimum-thickness location per sample, computes the
    required opening there, and applies a single smooth correction field across
    chordwise points. Inputs are rank-agnostic over leading dimensions for
    shapes ``[..., N, 2]``.

    Args:
        upper_points (np.ndarray): Upper surface coordinates with shape
            ``[..., N, 2]``.
        lower_points (np.ndarray): Lower surface coordinates with shape
            ``[..., N, 2]`` and identical x-grid as ``upper_points``.
        min_thickness (float): Target minimum thickness. Use ``0.0`` to only
            remove overlap.
        weighting (WeightingMode):
            - ``"symmetric"``: Gaussian decay with mirrored smoothstep taper
                around the critical point.
            - ``"edge_anchored"``: Gaussian field multiplied by two smoothstep
                tapers to preserve both LE and TE.
        locality_sigma (float): Standard deviation of the Gaussian field.
        edge_power (float): Exponent controlling sharpness of the two-sided
            smoothstep taper in anchored modes.

    Returns:
        tuple[np.ndarray, np.ndarray]: Repaired ``(upper_points, lower_points)``
            with unchanged x-coordinates.

    Notes:
        - All modes use a smooth LE taper so correction influence decays to
            zero at the leading edge.
        - If the minimum-thickness index is at LE or TE and
          ``min_thickness == 0``, no correction is applied.
        - ``edge_anchored`` enforce zero influence at both LE and TE.
    """
    if min_thickness < 0.0:
        raise ValueError("min_thickness must be >= 0.")
    if locality_sigma <= 0.0:
        raise ValueError("locality_sigma must be > 0.")
    if edge_power <= 0.0:
        raise ValueError("edge_power must be > 0.")
    if locality_sigma < 0.15 or locality_sigma > 0.3:
        warnings.warn(
            "Low locality_sigma values (<0.15) may cause overly localized "
            "corrections. High values (>0.3) may affect global geometry."
        )

    # Normalize/validate input geometry.
    upper = np.asarray(upper_points)
    lower = np.asarray(lower_points)
    if upper.shape != lower.shape:
        raise ValueError("upper_points and lower_points must have identical shape.")
    if upper.ndim < 2 or upper.shape[-1] != 2:
        raise ValueError("Expected shape [..., N, 2] for both inputs.")

    dtype = np.result_type(upper.dtype, lower.dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32
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

    # Only repair when there is an actual intersection (negative thickness).
    endpoint_contact = (idx_min == 0) | (idx_min == (n_points - 1))
    needs_fix = t_min < (-eps)
    if min_thickness == 0.0:
        needs_fix = needs_fix & (~endpoint_contact)

    # Required symmetric correction at x0.
    delta = 0.5 * np.clip(min_thickness - t_min, a_min=0.0, a_max=None)
    delta = np.where(needs_fix, delta, 0.0)

    # Build chordwise weights with w(x0)=1.
    weights = np.exp(-0.5 * ((x - x0[:, None]) / locality_sigma) ** 2)
    weights /= np.maximum(weights.max(axis=-1, keepdims=True), eps)

    # Left-side smoothstep taper: 0 at LE, 1 at x0.
    left_span = np.maximum(x0, eps)
    r_le = np.clip(x / left_span[:, None], 0.0, 1.0)
    left_taper = r_le**3 * (10.0 + r_le * (-15.0 + 6.0 * r_le))

    right_span = np.maximum(1.0 - x0, eps)

    match weighting:
        case "symmetric":
            # Mirror left-side smoothstep around x0 for symmetric tapering.
            x_mirror = 2.0 * x0[:, None] - x
            r_mirror = np.clip(x_mirror / left_span[:, None], 0.0, 1.0)
            right_mirror_taper = r_mirror**3 * (10.0 + r_mirror * (-15.0 + 6.0 * r_mirror))
            symmetric_taper = np.where(x <= x0[:, None], left_taper, right_mirror_taper)
            weights *= symmetric_taper
        case "edge_anchored":
            # Two-sided smoothstep taper: preserve both LE and TE.
            r_te = np.clip((1.0 - x) / right_span[:, None], 0.0, 1.0)
            right_taper = r_te**3 * (10.0 + r_te * (-15.0 + 6.0 * r_te))
            weights *= (left_taper * right_taper) ** edge_power
            weights[:, -1] = 0.0
        case _:
            raise ValueError(f"Unknown weighting mode: {weighting}")

    weights[:, 0] = 0.0

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
    min_thickness: float = 0.0,
    weighting: str = "edge_anchored",
    locality_sigma: float = 0.3,
    edge_power: float = 0.9,
) -> tuple["_torch.Tensor", "_torch.Tensor"]:
    """Repair negative thickness with one centered correction (PyTorch).

    Torch counterpart of :func:`repair_negative_thickness_points`, operating on
    tensors shaped ``[..., N, 2]``.

    Args:
        upper_points (_torch.Tensor): Upper surface coordinates with shape
            ``[..., N, 2]``.
        lower_points (_torch.Tensor): Lower surface coordinates with shape
            ``[..., N, 2]`` and identical x-grid as ``upper_points``.
        min_thickness (float): Target minimum thickness. Use ``0.0`` to only
            remove overlap.
        weighting (str):
            - ``"symmetric"``: Gaussian decay with mirrored smoothstep taper
                around the critical point.
            - ``"edge_anchored"``: Gaussian field multiplied by two smoothstep
                tapers to preserve both LE and TE.
        locality_sigma (float): Standard deviation of the Gaussian field.
        edge_power (float): Exponent controlling sharpness of the two-sided
            smoothstep taper in anchored modes.

    Returns:
        tuple[_torch.Tensor, _torch.Tensor]: Repaired
            ``(upper_points, lower_points)`` with unchanged x-coordinates.

    Notes:
        - All modes use a smooth LE taper so correction influence decays to
          zero at the leading edge.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is not installed. Use repair_negative_thickness_points "
            "for NumPy arrays."
        )

    if min_thickness < 0.0:
        raise ValueError("min_thickness must be >= 0.")
    if edge_power <= 0.0:
        raise ValueError("edge_power must be > 0.")

    # Validate shape and align dtype for stable numerical operations.
    if upper_points.shape != lower_points.shape:
        raise ValueError("upper_points and lower_points must have identical shape.")
    if upper_points.ndim < 2 or upper_points.shape[-1] != 2:
        raise ValueError("Expected shape [..., N, 2] for both inputs.")
    if locality_sigma <= 0.0:
        raise ValueError("locality_sigma must be > 0.")
    if locality_sigma < 0.15 or locality_sigma > 0.3:
        warnings.warn(
            "Low locality_sigma values (<0.15) may cause overly localized "
            "corrections. High values (>0.3) may affect global geometry."
        )

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
    needs_fix = t_min < (-eps)
    if min_thickness == 0.0:
        needs_fix = needs_fix & (~endpoint_contact)

    delta = 0.5 * (min_thickness - t_min).clamp_min(0.0)
    delta = torch.where(needs_fix, delta, torch.zeros_like(delta))

    # Build chordwise weights with w(x0)=1.
    weights = torch.exp(-0.5 * ((x - x0.unsqueeze(-1)) / locality_sigma) ** 2)
    weights = weights / weights.max(dim=-1, keepdim=True).values.clamp_min(eps)

    # Left-side smoothstep taper: 0 at LE, 1 at x0.
    left_span = x0.clamp_min(eps)
    r_le = (x / left_span.unsqueeze(-1)).clamp(0.0, 1.0)
    left_taper = r_le.pow(3) * (10.0 + r_le * (-15.0 + 6.0 * r_le))

    right_span = (1.0 - x0).clamp_min(eps)

    match weighting:
        case "symmetric":
            # Mirror left-side smoothstep around x0 for symmetric tapering.
            x_mirror = 2.0 * x0.unsqueeze(-1) - x
            r_mirror = (x_mirror / left_span.unsqueeze(-1)).clamp(0.0, 1.0)
            right_mirror_taper = r_mirror.pow(3) * (10.0 + r_mirror * (-15.0 + 6.0 * r_mirror))
            symmetric_taper = torch.where(x <= x0.unsqueeze(-1), left_taper, right_mirror_taper)
            weights = weights * symmetric_taper
        case "edge_anchored":
            # Two-sided smoothstep taper: preserve both LE and TE.
            r_te = ((1.0 - x) / right_span.unsqueeze(-1)).clamp(0.0, 1.0)
            right_taper = r_te.pow(3) * (10.0 + r_te * (-15.0 + 6.0 * r_te))
            weights = weights * (left_taper * right_taper).pow(edge_power)
            weights[..., -1] = 0.0  # just in case
        case _:
            raise ValueError(f"Unknown weighting mode: {weighting}")

    weights[..., 0] = 0.0  # just in case

    # Symmetric opening once: upper up, lower down.
    y_u = y_u + delta.unsqueeze(-1) * weights
    y_l = y_l - delta.unsqueeze(-1) * weights

    repaired_upper = upper.clone()
    repaired_lower = lower.clone()
    repaired_upper[..., 1] = y_u.reshape(*lead_shape, n_points)
    repaired_lower[..., 1] = y_l.reshape(*lead_shape, n_points)
    return repaired_upper, repaired_lower

