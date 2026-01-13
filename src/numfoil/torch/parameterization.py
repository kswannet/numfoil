import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass





@dataclass(frozen=True)
class Kulfan:
    """Container for Kulfan modified airfoil parameters.
    Number of parameters is fixed at 18 (16 shape + 2 TE).

    This dataclass exists to:
      1) centralize parsing of 16- vs 18-parameter vectors,
      2) provide utilities to move parameters to device/dtype and to convert back to tensors.

    Nomenclature (requested symbols and subscripts):
        a_u_1 ... a_u_6 : upper surface shape coefficients
        a_l_1 ... a_l_6 : lower surface shape coefficients
        w_le            : leading edge weight, default=0
        t_te            : trailing edge thickness, default=0

    Notes:
        - `frozen=True` makes instances immutable. This is useful because these
          objects represent “input parameters”; immutability prevents accidental
          in-place modification. Methods like `.to(...)` return a new instance.
    """

    a_u_1: torch.Tensor
    a_u_2: torch.Tensor
    a_u_3: torch.Tensor
    a_u_4: torch.Tensor
    a_u_5: torch.Tensor
    a_u_6: torch.Tensor
    a_l_1: torch.Tensor
    a_l_2: torch.Tensor
    a_l_3: torch.Tensor
    a_l_4: torch.Tensor
    a_l_5: torch.Tensor
    a_l_6: torch.Tensor
    w_le: torch.Tensor = torch.tensor(0.0)
    t_te: torch.Tensor = torch.tensor(0.0)


    @classmethod
    def from_tensor(cls, parameters: torch.Tensor | np.ndarray) -> "Kulfan":
        """Parse a 16- or 18-parameter tensor into a named `Kulfan`.

        Supported layouts:
            Preferred 10-parameter layout (omits y_te and assumes y_te=0):
                [ r_le,
                  x_z_u, y_z_u, k_z_u,
                  x_z_l, y_z_l, k_z_l,
                  t_te, theta_te, gamma_te ]

            Classic 11-parameter layout:
                [ r_le,
                  x_z_u, y_z_u, k_z_u,
                  x_z_l, y_z_l, k_z_l,
                  y_te, t_te, theta_te, gamma_te ]

        Shape:
            - [P] or [B, P], where P in {10, 11}

        Returns:
            PARSEC where each field has shape [B].
        """
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.ndim == 1:
            params = params.unsqueeze(0)
        if params.ndim != 2 or params.shape[-1] not in (16, 18):
            raise ValueError(
                "PARSEC tensor must have shape [10] or [B, 10] (preferred), "
                "or legacy [11]/[B, 11]; "
                f"got {tuple(params.shape)}."
            )



@dataclass(frozen=True)
class PARSEC:
    """Container for PARSEC airfoil parameters (with a safe default convention).

    This dataclass exists to:
      1) centralize parsing of 10- vs 11-parameter vectors,
      2) enforce the convention y_te = 0 by default (TE at (1,0) in chord-line coords),
      3) provide utilities to move parameters to device/dtype and to convert back to tensors.

    Nomenclature (requested symbols and subscripts):
        r_le       : leading edge radius
        x_z_u,y_z_u: upper crest (z) coordinates
        k_z_u      : upper crest curvature parameter (typically y''_u(x_z_u))
        x_z_l,y_z_l: lower crest (z) coordinates
        k_z_l      : lower crest curvature parameter (typically y''_l(x_z_l))
        y_te       : camber-line trailing-edge ordinate (a.k.a. z_te); default 0
        t_te       : trailing edge thickness (gap)
        theta_te   : trailing edge camber-line angle (theta; alpha reserved for AoA)
        gamma_te   : trailing edge wedge angle (gamma)

    Notes:
        - `frozen=True` makes instances immutable. This is useful because these
          objects represent “input parameters”; immutability prevents accidental
          in-place modification. Methods like `.to(...)` return a new instance.
    """

    r_le: torch.Tensor
    x_z_u: torch.Tensor
    y_z_u: torch.Tensor
    k_z_u: torch.Tensor
    x_z_l: torch.Tensor
    y_z_l: torch.Tensor
    k_z_l: torch.Tensor
    y_te: torch.Tensor = torch.tensor(0.0)
    t_te: torch.Tensor = torch.tensor(0.0)
    theta_te: torch.Tensor = torch.tensor(0.0)
    gamma_te: torch.Tensor = torch.tensor(0.0)

    @classmethod
    def validate_params(cls, parameters: torch.Tensor) -> None:
        """Validate that the given parameters tensor has a correct shape."""
        if parameters.ndim not in (1, 2) or parameters.shape[-1] not in (10, 11):
            raise ValueError(
                "PARSEC tensor must have shape [10] or [B, 10] (preferred), "
                "or legacy [11]/[B, 11]; "
                f"got {tuple(parameters.shape)}."
            )
        pass

    @classmethod
    def from_tensor(cls, parameters: torch.Tensor | np.ndarray) -> "PARSEC":
        """Parse a 10- or 11-parameter tensor into a named `PARSEC`.

        Supported layouts:
            Preferred 10-parameter layout (omits y_te and assumes y_te=0):
                [ r_le,
                  x_z_u, y_z_u, k_z_u,
                  x_z_l, y_z_l, k_z_l,
                  t_te, theta_te, gamma_te ]

            Classic 11-parameter layout:
                [ r_le,
                  x_z_u, y_z_u, k_z_u,
                  x_z_l, y_z_l, k_z_l,
                  y_te, t_te, theta_te, gamma_te ]

        Shape:
            - [P] or [B, P], where P in {10, 11}

        Returns:
            PARSEC where each field has shape [B].
        """
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.ndim == 1:
            params = params.unsqueeze(0)

        if params.shape[-1] == 10:
            (
                r_le,
                x_z_u, y_z_u, k_z_u,
                x_z_l, y_z_l, k_z_l,
                t_te, theta_te, gamma_te,
            ) = params.unbind(dim=-1)
            y_te = torch.zeros_like(r_le)
        else:
            (
                r_le,
                x_z_u, y_z_u, k_z_u,
                x_z_l, y_z_l, k_z_l,
                y_te, t_te, theta_te, gamma_te,
            ) = params.unbind(dim=-1)

        return cls(
            r_le=r_le,
            x_z_u=x_z_u, y_z_u=y_z_u, k_z_u=k_z_u,
            x_z_l=x_z_l, y_z_l=y_z_l, k_z_l=k_z_l,
            y_te=y_te, t_te=t_te, theta_te=theta_te, gamma_te=gamma_te,
        )

    @classmethod
    def from_values(
        cls,
        *,
        r_le: torch.Tensor | np.ndarray | float,
        x_z_u: torch.Tensor | np.ndarray | float,
        y_z_u: torch.Tensor | np.ndarray | float,
        k_z_u: torch.Tensor | np.ndarray | float,
        x_z_l: torch.Tensor | np.ndarray | float,
        y_z_l: torch.Tensor | np.ndarray | float,
        k_z_l: torch.Tensor | np.ndarray | float,
        y_te: torch.Tensor | np.ndarray | float = 0.0,
        t_te: torch.Tensor | np.ndarray | float = 0.0,
        theta_te: torch.Tensor | np.ndarray | float = 0.0,
        gamma_te: torch.Tensor | np.ndarray | float = 0.0,
        device: Optional[torch.device] = None,
    ) -> "PARSEC":
        """Build a `PARSEC` from explicit named values (keyword-only)."""
        def f32(x: torch.Tensor | np.ndarray | float) -> torch.Tensor:
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        return cls(
            r_le=f32(r_le),
            x_z_u=f32(x_z_u),
            y_z_u=f32(y_z_u),
            k_z_u=f32(k_z_u),
            x_z_l=f32(x_z_l),
            y_z_l=f32(y_z_l),
            k_z_l=f32(k_z_l),
            y_te=f32(y_te),
            t_te=f32(t_te),
            theta_te=f32(theta_te),
            gamma_te=f32(gamma_te),
        )

    def to(self, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> "PARSEC":
        """Return a new `PARSEC` moved/cast to (device, dtype)."""
        def _cast(t: torch.Tensor) -> torch.Tensor:
            return torch.as_tensor(t, dtype=dtype, device=device)

        return PARSEC(
            r_le=_cast(self.r_le),
            x_z_u=_cast(self.x_z_u),
            y_z_u=_cast(self.y_z_u),
            k_z_u=_cast(self.k_z_u),
            x_z_l=_cast(self.x_z_l),
            y_z_l=_cast(self.y_z_l),
            k_z_l=_cast(self.k_z_l),
            y_te=_cast(self.y_te),
            t_te=_cast(self.t_te),
            theta_te=_cast(self.theta_te),
            gamma_te=_cast(self.gamma_te),
        )

    def as_tensor11(self) -> torch.Tensor:
        """Return the classic 11-parameter PARSEC tensor (shape [B, 11])."""
        return torch.stack(
            [
                self.r_le,
                self.x_z_u, self.y_z_u, self.k_z_u,
                self.x_z_l, self.y_z_l, self.k_z_l,
                self.y_te, self.t_te,
                self.theta_te, self.gamma_te,
            ],
            dim=-1,
        )