import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from .curve import TorchCSTCurve, KulfanModifiedCST
from functools import cached_property

import numpy as np

from ..data import datafile, normalization

# TODO : make airfoil class with dependcy injection of curve types, a truly
# TODO |    generic airfoil class with CST, Bezier, etc curve support and
# TODO |    universal fit method.

class TorchKulfanAirfoil(nn.Module):
    """
    Batched Kulfan (CST) airfoil with comprehensive geometric analysis.

    Automatically handles both single airfoils and batches. All geometric
    properties are computed on the MODIFIED surfaces (including LE/TE adjustments).

    Parameter layout: [upper_coeffs | lower_coeffs | le_weight | te_thickness]
    """

    def __init__(
        self,
        upper_surface: KulfanModifiedCST,
        lower_surface: KulfanModifiedCST,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()

        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create base CST curves
        self.register_buffer("upper_surface", upper_surface)
        self.register_buffer("lower_surface", lower_surface)

        self._validate_curves()

    def _validate_curves(self):
        # if self.upper_surface.n_coefficients != self.lower_surface.n_coefficients:
        if self.upper_surface.parameters.shape != self.lower_surface.parameters.shape:
            raise ValueError(
                "Mismatched upper and lower surface parameter shapes: "
                f"Upper and lower surfaces must have the same number of CST coefficients, "
                f"got {self.upper_surface.parameters.shape} and {self.lower_surface.parameters.shape}, "
            )
        if self.upper_surface.leading_edge_weight.shape != self.lower_surface.leading_edge_weight.shape:
            raise ValueError(
                "Upper and lower surfaces must have the same leading edge weight shape, "
                f"got {self.upper_surface.leading_edge_weight.shape} and "
                f"{self.lower_surface.leading_edge_weight.shape}"
            )
        if self.upper_surface.trailing_edge_thickness.shape != self.lower_surface.trailing_edge_thickness.shape:
            raise ValueError(
                "Upper and lower surfaces must have the same trailing edge thickness shape, "
                f"got {self.upper_surface.trailing_edge_thickness.shape} and "
                f"{self.lower_surface.trailing_edge_thickness.shape}"
            )
        if self.upper_surface.batch_size != self.lower_surface.batch_size:
            raise ValueError(
                "Upper and lower surfaces must have the same batch size, "
                f"got {self.upper_surface.batch_size} and {self.lower_surface.batch_size}"
            )

    @classmethod
    def from_param_tensor(
        cls,
        parameters: torch.Tensor | np.ndarray,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
    ) -> "TorchKulfanAirfoil":
        """
        Create airfoil from parameter tensor.

        Args:
            parameters (torch.Tensor | np.ndarray, shape [batch, 2*n_coeffs + 2]):
                tensor of upper and lower CST coefficients concatenated
                with leading edge weight and trailing edge thickness.
            n1 (float): CST exponent n1, default 0.5
            n2 (float): CST exponent n2, default 1.0
            device (Optional[torch.device | str]): Torch device
        """
        if parameters.ndim > 2:
            raise ValueError(
                f"Parameters tensor must be 2D (batch, n_params), got shape {parameters.shape}"
            )
        n_params = parameters.shape[1]
        if n_params < 4:
            raise ValueError(
                f"Need at least 4 parameters (1 upper, 1 lower, w_le, t_te), got {n_params}"
            )
        if n_params % 2 != 0:
            raise ValueError(
                f"Number of CST parameters must be even (2*n_coeffs + 2), got {n_params}"
            )
        n_coeffs = (n_params - 2) // 2
        return cls.from_kulfan_params(
            upper_coeffs=parameters[..., :n_coeffs],
            lower_coeffs=parameters[..., n_coeffs:-2],
            w_le=parameters[..., -2],
            t_te=parameters[..., -1],
            n1=n1,
            n2=n2,
            device=device,
        )

    @classmethod
    def from_kulfan_params(
        cls,
        upper_coeffs: torch.Tensor | np.ndarray,
        lower_coeffs: torch.Tensor | np.ndarray,
        w_le: torch.Tensor | np.ndarray,
        t_te: torch.Tensor | np.ndarray,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
        ) -> "TorchKulfanAirfoil":
        return cls(
            KulfanModifiedCST(
                upper_coeffs,
                leading_edge_weight=w_le,
                trailing_edge_thickness=t_te,
                n1=n1,
                n2=n2,
                device=device,
            ),
            KulfanModifiedCST(
                lower_coeffs,
                leading_edge_weight=w_le,
                trailing_edge_thickness=t_te,
                n1=n1,
                n2=n2,
                device=device,
            ),
        )

    @property
    def is_batched(self) -> bool:
        return self.batch_size > 1

    @property
    def batch_size(self) -> int:
        return self.upper_surface.batch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate upper and lower surfaces.

        Args:
            x: Chordwise locations, shape (n_points,)

        Returns:
            (y_upper, y_lower), both shape (batch, n_points)
        """
        return self.upper(x), self.lower(x)

    def get_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get airfoil coordinates in standard counterclockwise format.

        Args:
            x: Chordwise locations, shape (n_points,)

        Returns:
            Coordinates, shape (batch, 2*n_points-1, 2)
        """
        y_upper, y_lower = self.forward(x)

        # Standard format: upper TE→LE (reversed), then lower LE→TE
        x_full = torch.cat([x.flip(0), x[1:]])
        y_full = torch.cat([y_upper.flip(-1), y_lower[:, 1:]], dim=-1)

        x_batched = x_full.unsqueeze(0).expand(self.batch_size, -1)
        return torch.stack([x_batched, y_full], dim=-1)

    def thickness_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Local thickness t(x) = y_upper(x) - y_lower(x)"""
        y_upper, y_lower = self.forward(x)
        return y_upper - y_lower

    def camber_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Local camber c(x) = (y_upper(x) + y_lower(x)) / 2"""
        y_upper, y_lower = self.forward(x)
        return (y_upper + y_lower) / 2

    def max_thickness(self, x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find maximum thickness and its location.

        Args:
            x: Optional evaluation points. If None, uses dense cosine spacing.

        Returns:
            (t_max, x_max), both shape (batch,)
        """
        if x is None:
            x = torch.cos(torch.linspace(0, torch.pi, 200, device=self.device)) * 0.5 + 0.5

        t = self.thickness_distribution(x)
        t_max, idx = torch.max(t, dim=-1)
        x_max = x[idx]

        return t_max, x_max

    def max_camber(self, x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find maximum camber and its location.

        Returns:
            (c_max, x_max), both shape (batch,)
        """
        if x is None:
            x = torch.linspace(0, 1, 200, device=self.device)

        c = self.camber_distribution(x)
        c_abs = torch.abs(c)
        c_max, idx = torch.max(c_abs, dim=-1)
        x_max = x[idx]

        # Preserve sign
        c_max = c[torch.arange(self.batch_size), idx]

        return c_max, x_max

    def leading_edge_radius(self) -> torch.Tensor:
        """
        Estimate leading edge radius using curvature at x=0.

        r_LE ≈ 1 / |κ(0)|

        Returns:
            LE radius, shape (batch,)
        """
        x_le = torch.tensor([0.01], device=self.device)  # Very close to LE

        # Average curvature of upper and lower surfaces
        kappa_upper = self.upper.curvature(x_le, mode="autograd").squeeze(-1)
        kappa_lower = self.lower.curvature(x_le, mode="autograd").squeeze(-1)

        kappa_avg = (torch.abs(kappa_upper) + torch.abs(kappa_lower)) / 2

        return 1.0 / (kappa_avg + 1e-8)  # Avoid division by zero

    def trailing_edge_angle(self) -> torch.Tensor:
        """
        Compute trailing edge wedge angle in degrees.

        Returns:
            TE angle, shape (batch,)
        """
        x_te = torch.tensor([0.99], device=self.device)

        dy_upper = self.upper.first_derivative(x_te, mode="autograd").squeeze(-1)
        dy_lower = self.lower.first_derivative(x_te, mode="autograd").squeeze(-1)

        # Angle between surfaces
        angle_rad = torch.atan(dy_upper) - torch.atan(dy_lower)
        return torch.abs(angle_rad) * 180 / torch.pi

    def trailing_edge_thickness(self) -> torch.Tensor:
        """
        Actual trailing edge thickness at x=1.

        Returns:
            TE thickness, shape (batch,)
        """
        x_te = torch.tensor([1.0], device=self.device)
        return self.thickness_distribution(x_te).squeeze(-1)


# Example usage
if __name__ == "__main__":
    batch_size = 32
    n_coeffs = 8
    nn_output = torch.randn(batch_size, 2 * n_coeffs + 2)

    airfoils = TorchKulfanAirfoil(nn_output)

    # Evaluate geometry
    x = torch.linspace(0, 1, 100)
    y_upper, y_lower = airfoils(x)
    print(f"Surfaces: {y_upper.shape}")

    # Geometric properties
    t_max, x_t_max = airfoils.max_thickness()
    print(f"Max thickness: {t_max.shape}, at x={x_t_max.shape}")

    c_max, x_c_max = airfoils.max_camber()
    print(f"Max camber: {c_max.shape}, at x={x_c_max.shape}")

    r_le = airfoils.leading_edge_radius()
    print(f"LE radius: {r_le.shape}")

    te_angle = airfoils.trailing_edge_angle()
    print(f"TE angle: {te_angle.shape}")

    # Curvature on modified surfaces
    kappa_upper = airfoils.upper.curvature(x)
    print(f"Upper curvature: {kappa_upper.shape}")
