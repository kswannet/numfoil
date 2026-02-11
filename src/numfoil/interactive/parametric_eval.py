from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch


def _binomial_weights(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # k=0..n
    k = torch.arange(n + 1, device=device, dtype=dtype)
    n_plus_1 = torch.full((n + 1,), float(n + 1), device=device, dtype=dtype)
    log_binom = torch.lgamma(n_plus_1) - torch.lgamma(k + 1.0) - torch.lgamma(n_plus_1 - k)
    return torch.exp(log_binom)  # [K]


def kulfan_modified_cst_y(
    *,
    x: torch.Tensor,
    coefficients: torch.Tensor,
    w_le: torch.Tensor,
    t_te: torch.Tensor,
    surface: Literal["upper", "lower"],
    n1: float = 0.5,
    n2: float = 1.0,
) -> torch.Tensor:
    """Fast Kulfan-modified CST ordinate evaluation.

    Shapes:
      - x: [N]
      - coefficients: [K]
      - w_le, t_te: scalar tensors

    Returns:
      - y: [N]
    """
    device = x.device
    dtype = x.dtype

    if coefficients.ndim != 1:
        raise ValueError("coefficients must be 1D")

    K = int(coefficients.shape[0])
    n = K - 1
    k = torch.arange(K, device=device, dtype=dtype)
    binom = _binomial_weights(n, device=device, dtype=dtype)  # [K]

    x_clamped = x.clamp(min=torch.finfo(dtype).eps, max=1.0 - torch.finfo(dtype).eps)
    Cx = torch.pow(x_clamped, n1) * torch.pow(1.0 - x_clamped, n2)  # [N]

    # Bernstein basis: x^k (1-x)^(n-k)
    Bx = torch.pow(x_clamped.unsqueeze(-1), k) * torch.pow(1.0 - x_clamped.unsqueeze(-1), (n - k))  # [N, K]

    # weighted basis includes class function and binomial
    Wx = Cx.unsqueeze(-1) * Bx * binom.unsqueeze(0)  # [N, K]
    y_base = Wx @ coefficients  # [N]

    # Kulfan modifications
    te_sign = 1.0 if surface == "upper" else -1.0
    y_le = w_le * x_clamped * torch.pow(1.0 - x_clamped, n + 0.5)
    y_te = te_sign * t_te * x_clamped / 2.0

    return y_base + y_le + y_te


def parsec_surface_coefficients(
    *,
    r_le: torch.Tensor,
    x_z: torch.Tensor,
    y_z: torch.Tensor,
    k_z: torch.Tensor,
    y_te: torch.Tensor,
    dy_te: torch.Tensor,
    surface: Literal["upper", "lower"],
) -> torch.Tensor:
    """Solve PARSEC polynomial coefficients a1..a6 for one surface.

    Returns coeffs [6] for:
        y(x) = a1 x^{1/2} + a2 x + a3 x^{3/2} + a4 x^2 + a5 x^{5/2} + a6 x^3
    """
    dtype = r_le.dtype
    device = r_le.device
    eps = torch.finfo(dtype).eps

    r_le = torch.abs(r_le).clamp_min(eps)
    x_z = x_z.clamp(min=eps, max=1.0 - eps)

    sign = 1.0 if surface == "upper" else -1.0
    a1 = sign * torch.sqrt(2.0 * r_le)

    sqrt_xc = torch.sqrt(x_z)
    xc = x_z
    xc_3_2 = xc * sqrt_xc
    xc2 = xc * xc
    xc_5_2 = xc2 * sqrt_xc
    xc3 = xc2 * xc
    inv_sqrt_xc = 1.0 / sqrt_xc
    inv_xc_3_2 = inv_sqrt_xc / xc

    ones = torch.ones((), dtype=dtype, device=device)
    zeros = torch.zeros((), dtype=dtype, device=device)

    # A is 5x5 for unknowns [a2..a6]
    A = torch.stack(
        [
            torch.stack([ones, ones, ones, ones, ones]),
            torch.stack([ones, 1.5 * ones, 2.0 * ones, 2.5 * ones, 3.0 * ones]),
            torch.stack([xc, xc_3_2, xc2, xc_5_2, xc3]),
            torch.stack([ones, 1.5 * sqrt_xc, 2.0 * xc, 2.5 * xc_3_2, 3.0 * xc2]),
            torch.stack([zeros, 0.75 * inv_sqrt_xc, 2.0 * ones, 3.75 * sqrt_xc, 6.0 * xc]),
        ],
        dim=0,
    )

    b1 = y_te - a1
    b2 = dy_te - 0.5 * a1
    b3 = y_z - a1 * sqrt_xc
    b4 = -0.5 * a1 * inv_sqrt_xc
    b5 = k_z + 0.25 * a1 * inv_xc_3_2
    b = torch.stack([b1, b2, b3, b4, b5], dim=0)

    sol = torch.linalg.solve(A, b)  # [5]
    a2, a3, a4, a5, a6 = sol.unbind(dim=0)
    return torch.stack([a1, a2, a3, a4, a5, a6], dim=0)


def parsec_airfoil_y(
    *,
    x: torch.Tensor,
    r_le: torch.Tensor,
    x_z_u: torch.Tensor,
    y_z_u: torch.Tensor,
    k_z_u: torch.Tensor,
    x_z_l: torch.Tensor,
    y_z_l: torch.Tensor,
    k_z_l: torch.Tensor,
    y_te: torch.Tensor,
    t_te: torch.Tensor,
    theta_te: torch.Tensor,
    gamma_te: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast PARSEC airfoil evaluation (upper + lower) without module re-instantiation."""
    y_te_u = y_te + t_te / 2.0
    y_te_l = y_te - t_te / 2.0

    dy_te_u = torch.tan(theta_te + gamma_te / 2.0)
    dy_te_l = torch.tan(theta_te - gamma_te / 2.0)

    cu = parsec_surface_coefficients(
        r_le=r_le,
        x_z=x_z_u,
        y_z=y_z_u,
        k_z=k_z_u,
        y_te=y_te_u,
        dy_te=dy_te_u,
        surface="upper",
    )
    cl = parsec_surface_coefficients(
        r_le=r_le,
        x_z=x_z_l,
        y_z=y_z_l,
        k_z=k_z_l,
        y_te=y_te_l,
        dy_te=dy_te_l,
        surface="lower",
    )

    dtype = x.dtype
    eps = torch.finfo(dtype).eps
    x_safe = x.clamp(min=eps)
    sqrtx = torch.sqrt(x_safe)
    basis = torch.stack(
        [
            sqrtx,
            x_safe,
            x_safe * sqrtx,
            x_safe * x_safe,
            (x_safe * x_safe) * sqrtx,
            (x_safe * x_safe) * x_safe,
        ],
        dim=-1,
    )  # [N, 6]

    y_u = basis @ cu
    y_l = basis @ cl
    return y_u, y_l
