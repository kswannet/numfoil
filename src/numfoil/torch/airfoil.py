import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from .curve import TorchCSTCurve, KulfanModifiedCST
from functools import cached_property

class TorchKulfanAirfoil(nn.Module):
    """
    Batched Kulfan (CST) airfoil with comprehensive geometric analysis.

    Automatically handles both single airfoils and batches. All geometric
    properties are computed on the MODIFIED surfaces (including LE/TE adjustments).

    Parameter layout: [upper_coeffs | lower_coeffs | le_weight | te_thickness]
    """

    def __init__(
        self,
        parameters: torch.Tensor,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()

        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.parameters = torch.as_tensor(parameters, dtype=torch.float32, device=self.device)

        if self.parameters.ndim == 1:
            self.parameters = self.parameters.unsqueeze(0)
        elif self.parameters.ndim != 2:
            raise ValueError("parameters must be 1D or 2D")

        n_params = parameters.shape[1]
        if n_params < 4:
            raise ValueError(f"Need at least 4 parameters, got {n_params}")

        n_cst_params = n_params - 2
        if n_cst_params % 2 != 0:
            raise ValueError(f"Number of CST parameters must be even (got {n_cst_params})")

        self.n_coeffs_per_surface = n_cst_params // 2
        self.batch_size = self.parameters.shape[0]
        self.n1 = n1
        self.n2 = n2

        # Extract parameters
        # upper_coeffs = parameters[:, :self.n_coeffs_per_surface]
        lower_coeffs = self.parameters[:, self.n_coeffs_per_surface:-2]
        le_weight = self.parameters[:, -2]
        te_thickness = self.parameters[:, -1]

        # Create base CST curves
        # base_upper = TorchCSTCurve(upper_coeffs, n1=n1, n2=n2, device=self.device)
        # base_lower = TorchCSTCurve(lower_coeffs, n1=n1, n2=n2, device=self.device)

        # Wrap in modification layers
        # self.upper = KulfanModifiedCST(base_upper, le_weight, te_thickness, "upper")
        # self.lower = KulfanModifiedCST(base_lower, le_weight, te_thickness, "lower")

    @cached_property
    def upper_surface(self) -> TorchCSTCurve:
        """Base (unmodified) upper surface curve."""
        baseCSTcurve = TorchCSTCurve(
            self.parameters[:, : self.n_coeffs_per_surface],
            n1=self.n1,
            n2=self.n2,
            device=self.device,
        )
        return KulfanModifiedCST(
            baseCSTcurve,
            leading_edge_weight=self.parameters[:, -2],
            trailing_edge_thickness=self.parameters[:, -1],
            surface_type="upper"
        )

    @cached_property
    def lower_surface(self) -> TorchCSTCurve:
        """Base (unmodified) lower surface curve."""
        baseCSTcurve = TorchCSTCurve(
            self.parameters[:, self.n_coeffs_per_surface:-2],
            n1=self.n1,
            n2=self.n2,
            device=self.device,
        )
        return KulfanModifiedCST(
            baseCSTcurve,
            leading_edge_weight=self.parameters[:, -2],
            trailing_edge_thickness=self.parameters[:, -1],
            surface_type="lower"
        )

    @property
    def is_batched(self) -> bool:
        return self.batch_size > 1

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
