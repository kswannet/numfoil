import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from .curve import TorchCSTCurve, KulfanModifiedCST
from functools import cached_property

from ..util import cosine_spacing

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
        # self.register_buffer("upper_surface", upper_surface)
        # self.register_buffer("lower_surface", lower_surface)
        self.upper_surface = upper_surface.to(device=self.device)
        self.lower_surface = lower_surface.to(device=self.device)

        self.eps = upper_surface.eps

        self._validate_curves()

    def _validate_curves(self):
        # if self.upper_surface.n_coefficients != self.lower_surface.n_coefficients:
        if self.upper_surface.parameters.shape != self.lower_surface.parameters.shape:
            # TODO : I don't think they really need to have the same number of
            # TODO | coefficients, but it's good to be consistent
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
    def from_kulfan_tensor(
        cls,
        parameters: torch.Tensor | np.ndarray,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
    ) -> "TorchKulfanAirfoil":
        """Create airfoil from Kulfan parameter tensor.
        This measns a tensor of shape [batch, 2*n_coeffs + 2], where the batch
        dimension is optional, and the number of CST coefficients should be the
        same for upper and lower surfaces.

        The parameters should be ordered as:
            - upper CST coefficients
            - lower CST coefficients
            - leading edge weight
            - trailing edge thickness

        Args:
            parameters (torch.Tensor | np.ndarray, shape [batch, 2*n_coeffs + 2]):
                tensor of upper and lower CST coefficients concatenated
                with leading edge weight and trailing edge thickness.
            n1 (float): CST exponent n1, default 0.5
            n2 (float): CST exponent n2, default 1.0
            device (Optional[torch.device | str]): Torch device

        Returns:
            TorchKulfanAirfoil instance

        Note:
            This method is just a convenience wrapper around
            `from_kulfan_params`. The difference being that `from_kulfan_params`
            requires separate tensors for upper and lower coefficients, leading
            edge weight, and trailing edge thickness, while this method accepts
            a single tensor of parameters which is split internally, before
            passing the components to `from_kulfan_params`.
        """
        # first some validation
        if parameters.ndim > 2:
            raise ValueError(
                f"Parameters tensor must be 2D (batch, n_params), got shape {parameters.shape}"
            )
        n_params = parameters.shape[-1]
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
        w_le: torch.Tensor | np.ndarray = 0.0,
        t_te: torch.Tensor | np.ndarray = 0.0,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
        ) -> "TorchKulfanAirfoil":
        t_te = abs(t_te)  # ensure non-negative TE thickness
        return cls(
            KulfanModifiedCST(
                upper_coeffs,
                leading_edge_weight=w_le,
                trailing_edge_thickness=t_te,
                surface_type="upper",
                n1=n1,
                n2=n2,
                device=device,
            ),
            KulfanModifiedCST(
                lower_coeffs,
                leading_edge_weight=w_le,
                trailing_edge_thickness=t_te,
                surface_type="lower",
                n1=n1,
                n2=n2,
                device=device,
            ),
        )

    @property
    def kulfan_params(self) -> torch.Tensor:
        """Get Kulfan parameters as a single tensor.

        Returns:
        torch.Tensor, shape [batch, 2*n_coeffs + 2]
        """
        return torch.cat([
            self.upper_surface.coefficients,            # [B, n_coeffs]
            self.lower_surface.coefficients,            # [B, n_coeffs]
            self.upper_surface.leading_edge_weight,     # [B, 1]
            self.upper_surface.trailing_edge_thickness, # [B, 1]
        ], dim=-1).squeeze()

    @property
    def batch_size(self) -> int:
        # sanity check, shouldn't trigger due to earlier validation, but just in case
        if self.upper_surface.batch_size != self.lower_surface.batch_size:
            raise ValueError(
                "Upper and lower surfaces have different batch sizes: "
                f"{self.upper_surface.batch_size} and {self.lower_surface.batch_size}"
            )
        return self.upper_surface.batch_size

    @property
    def is_batched(self) -> bool:
        return self.batch_size > 1

    @property
    def shape(self) -> Tuple[int, int]:
        return self.kulfan_params.shape

    @property
    def __len__(self) -> int:
        return self.batch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate upper and lower surfaces.

        Args:
            x: Chordwise locations, shape (n_points,)

        Returns:
            (y_upper, y_lower), both shape (batch, n_points)
        """
        return self.upper_surface_at(x), self.lower_surface_at(x)

    def coordinates_at(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get airfoil coordinates in standard counterclockwise format.

        Args:
            x: Chordwise locations, shape (n_points,)

        Returns:
            Coordinates, shape (batch, 2*n_points-1, 2)
        """
        y_upper, y_lower = self.forward(x)

        # Standard Selig format: upper TE to LE (reversed), then lower LE to TE
        x_full = torch.cat([x.flip(0), x[1:]])
        y_full = torch.cat([y_upper.flip(-1), y_lower[:, 1:]], dim=-1)

        x_batched = x_full.unsqueeze(0).expand(self.batch_size, -1)
        return torch.stack([x_batched, y_full], dim=-1)

    @property
    def points(self, n_points: int = 100, dtype=torch.float32) -> torch.Tensor:
        """
        Get airfoil points at cosine-spaced locations in Selig format.

        Args:
            n_points: Number of chordwise points per surface.

        Returns:
            Coordinates, shape [batch, 2*n_points-1, 2]
        """
        x = torch.as_tensor(
            cosine_spacing(0, 1, n_points),
            device=self.device,
            dtype=dtype,
        )

        upper = self.upper_surface(x).to(dtype=dtype)  # [B>1, 100]
        lower = self.lower_surface(x).to(dtype=dtype)  # [B>1, 100]

        if self.is_batched:
            x = x.unsqueeze(0).expand(self.batch_size, -1)  # [B, 100]

        return torch.cat([
            torch.stack([x, upper], dim=-1).flip(dims=[-2]),  # [B, 100, 2] reversed
            torch.stack([x, lower], dim=-1)[..., 1:, :],      # [B, 99, 2] skip LE
            ], dim=-2
        ).to(dtype=dtype).squeeze() # [B>1, 199, 2]

    def upper_surface_at(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate upper surface at given x locations.

        Args:
            x: Chordwise locations, shape (n_points,)

        Returns:
            y_upper: Upper surface y-coordinates, shape (batch, n_points)
        """
        return self.upper_surface(x)

    def lower_surface_at(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate lower surface at given x locations.

        Args:
            x: Chordwise locations, shape (n_points,)

        Returns:
            y_lower: Lower surface y-coordinates, shape (batch, n_points)
        """
        return self.lower_surface(x)

    def thickness_at(self, x: torch.Tensor) -> torch.Tensor:
        """Local thickness t(x) = y_upper(x) - y_lower(x)"""
        y_upper, y_lower = self.forward(x)
        return y_upper - y_lower

    def camber_at(self, x: torch.Tensor) -> torch.Tensor:
        """Local camber c(x) = (y_upper(x) + y_lower(x)) / 2

        Args:
            x: Chordwise locations, shape (n_points,)
        Returns:
            c: Camber values, shape (batch, n_points)
        """
        y_upper, y_lower = self.forward(x)
        return (y_upper + y_lower) / 2

    @property
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

        t = self.thickness_at(x)
        t_max, idx = torch.max(t, dim=-1)
        x_max = x[idx]

        return t_max, x_max

    @property
    def max_camber(self, x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find maximum camber and its location.

        Returns:
            (c_max, x_max), both shape (batch,)
        """
        if x is None:
            x = torch.linspace(0, 1, 2000, device=self.device)

        c = self.camber_at(x)
        c_abs = torch.abs(c)
        c_max, idx = torch.max(c_abs, dim=-1)
        x_max = x[idx]

        # Preserve sign
        c_max = c[torch.arange(self.batch_size), idx]

        return c_max, x_max

    @property
    def leading_edge_radius(self) -> torch.Tensor:
        """
        Estimate leading edge radius using curvature at x=0.

        r_LE ≈ 1 / |κ(0)|

        Returns:
            LE radius, shape (batch,)
        """
        x_le = torch.tensor([0.01], device=self.device)  # Very close to LE

        # Average curvature of upper and lower surfaces
        kappa_upper = self.upper.curvature(x_le, mode="analytic").squeeze(-1)
        kappa_lower = self.lower.curvature(x_le, mode="analytic").squeeze(-1)

        kappa_avg = (torch.abs(kappa_upper) + torch.abs(kappa_lower)) / 2

        return 1.0 / (kappa_avg + self.eps)  # Avoid division by zero

    @property
    def trailing_edge_angle(self) -> torch.Tensor:
        """
        Compute trailing edge wedge angle in degrees.

        Returns:
            TE angle, shape (batch,)
        """
        x_te = torch.tensor([0.99], device=self.device)

        dy_upper = self.upper.first_derivative(x_te, mode="analytic").squeeze(-1)
        dy_lower = self.lower.first_derivative(x_te, mode="analytic").squeeze(-1)

        # Angle between surfaces
        angle_rad = torch.atan(dy_upper) - torch.atan(dy_lower)
        return torch.abs(angle_rad) * 180 / torch.pi

    @property
    def trailing_edge_thickness(self) -> torch.Tensor:
        """
        Actual trailing edge thickness at x=1.

        Returns:
            TE thickness, shape (batch,)
        """
        x_te = torch.tensor([1.0], device=self.device)
        return self.thickness_at(x_te).squeeze(-1)

    @property
    def area(self, n_points: int = 1000) -> torch.Tensor:
        """
        Compute airfoil cross-sectional area using numerical integration.

        Args:
            n_points: Number of chordwise points for integration.

        Returns:
            Airfoil area, shape (batch,)
        """
        x = torch.linspace(0, 1, n_points, device=self.device)
        t = self.thickness_at(x)  # shape (batch, n_points)

        # Numerical integration using the trapezoidal rule
        area = torch.trapz(t, x, dim=-1)  # shape (batch,)

        return area

    def _validate_thickness(self) -> bool:
        """
        Check if thickness stays positive along the chord.

        Args:
            x (torch.Tensor, optional): Evaluation points. Defaults to cosine spacing.

        Returns:
            bool: True if all sampled thickness values exceed tolerance.
        """
        beta = torch.linspace(0.0, torch.pi, 512, device=self.device)
        x = 0.5 * (1.0 - torch.cos(beta))
        return torch.all(self.thickness(x) >= 0).item()

    def plot(
        self,
        idx: int = 0,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> "plt.Figure":
        """Plot airfoil using matplotlib.

        Args:
            idx (int): Index of airfoil to plot in batch.
            name (Optional[str]): Title for the plot.
            save_dir (Optional[str]): Directory to save the plot.
                Default None, does not save.

        Returns:
            Tuple[plt.Figure, plt.Axes]
        """
        import matplotlib.pyplot as plt

        x = torch.linspace(0, 1, 2000, device=self.device)
        y_upper, y_lower = self.forward(x)

        if not self.is_batched:
            y_upper = y_upper.unsqueeze(0)
            y_lower = y_lower.unsqueeze(0)

        x = x.detach().cpu().numpy()
        y_upper = y_upper.detach().cpu().numpy()
        y_lower = y_lower.detach().cpu().numpy()

        figsize = (10, 6) #if not self.is_batched else (10, 4)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y_upper[idx], 'b-', label='Upper Kulfan Surface')
        ax.plot(x, y_lower[idx], 'r-', label='Lower Kulfan Surface')
        ax.axis('equal')
        ax.set_title(name or 'Kulfan Airfoil')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.axis('equal')
        ax.grid(True)
        ax.legend()
        ax.set_facecolor("white")
        plt.tight_layout()

        # if there is only one airfoil, display parameters
        if not self.is_batched:
            # Get parameters for the selected airfoil (idx)
            upper_coeffs = self.upper_surface.coefficients[idx].detach().cpu().numpy().round(4)
            lower_coeffs = self.lower_surface.coefficients[idx].detach().cpu().numpy().round(4)
            w_le = self.upper_surface.leading_edge_weight[idx].item()
            t_te = self.upper_surface.trailing_edge_thickness[idx].item()

            # # Format the textbox text
            # textstr = (f"Upper coeffs: {upper_coeffs.round(4)}\n"
            #     f"Lower coeffs: {lower_coeffs.round(4)}\n"
            #     f"LE weight:       {w_le:.4f}\n"
            #     f"TE thickness:    {t_te:.4f}")

            # # Add textbox under the plot
            # plt.subplots_adjust(bottom=0.225, top=0.9)  # Adjust bottom margin to create space
            # plt.figtext(0.2, 0.03, textstr, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

            # Format as aligned columns (monospace helps alignment)
            row_format = "{:<14} " + " ".join(["{:>8.4f}"] * len(upper_coeffs))

            textstr = (
                row_format.format("Upper coeffs:", *upper_coeffs) + "\n" +
                row_format.format("Lower coeffs:", *lower_coeffs) + "\n" +
                f"{'LE weight:':<15} {w_le:>8.4f}\n"
                f"{'TE thickness:':<14} {t_te:>8.4f}"
            )
            plt.subplots_adjust(bottom=0.245, top=0.925)  # Adjust bottom margin to create space
            plt.figtext(
                0.1, 0.0275, textstr, fontsize=10,
                # fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"),
                linespacing=1.5,
            )



        if save_dir is not None:
            filepath = f"{save_dir}"
            fig.savefig(filepath, )
        return fig, ax

    @classmethod
    def fit(
        cls,
        upper_points: torch.Tensor | np.ndarray,
        lower_points: torch.Tensor | np.ndarray,
        n_coefficients: int = 8,
        *,
        use_kulfan_modifiers: bool = True,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
        rcond: Optional[float] = None,
    ) -> "TorchKulfanAirfoil":
        """Jointly fit upper and lower surfaces to Selig-format coordinates with
        shared Kulfan modifiers for leading and trailing edge.

        ! MAKE SURE THE INPUT DATA IS CLEANED AND NORMALIZED FFS!

        This method solves a single least squares problem for both surfaces,
        ensuring the leading-edge weight and trailing-edge thickness are shared.
        The design matrix stacks upper and lower bases vertically, with shared
        LE/TE columns.

        The least-squares system solves, for each batch element, the block-linear
        problem:

        :math:`\begin{bmatrix} M_{upper} & w_{le,upper} & t_{te,upper} \\
                  M_{lower} & w_{le,lower} & t_{te,lower} \end{bmatrix}
         \begin{bmatrix} a_u \\ a_l \\ w_{le} \\ t_{te} \end{bmatrix}
         = \begin{bmatrix} y_{upper} \\ y_{lower} \end{bmatrix}`,

         or simpler:
            M θ = y,
            θ = [a_u, a_l, w_le, t_te]^T,

         where:
            - a_u ∈ R^{K} are the upper CST coefficients,
            - a_l ∈ R^{K} are the lower CST coefficients,
            - w_le is the shared leading-edge weight,
            - t_te is the shared trailing-edge thickness magnitude (sign handled
              via the row construction: +x/2 for the upper rows, -x/2 for the lower rows).

        Args:
            upper_points (torch.Tensor | np.ndarray):
                Upper surface points [(B,) N, 2].
            lower_points (torch.Tensor | np.ndarray):
                Lower surface points [(B,) N, 2].
            n_coefficients (int):
                Number of CST coefficients per surface.
            use_kulfan_modifiers (bool):
                Fit KulfanModifiedCST when True.
            n1 (float):
                Class exponent near x=0.
            n2 (float):
                Class exponent near x=1.
            device (torch.device | str, optional):
                Target device.
            rcond (float, optional):
                Cutoff for least squares.
            prevent_overlap (bool):
                Enforce positive thickness.

        Returns:
            TorchKulfanAirfoil: Fitted model.

        Raises:
            ValueError: If insufficient points or surfaces overlap.
        """
        target_device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if upper_points.ndim > 3 or upper_points.shape[-1] != 2 or \
           lower_points.ndim > 3 or lower_points.shape[-1] != 2:
            raise ValueError(
                "coordinates must have shape [N, 2], but got:\n"
                f" upper_points.shape={upper_points.shape}, "
                f" lower_points.shape={lower_points.shape}"
            )
        # todo: this one needs work
        # if not (
        #         torch.all(  # all x in [0, 1]
        #             (0 <= pts[..., 0])     # x >= 0
        #             & (pts[..., 0] <= 1)   # x <= 1
        #         )
        #     or
        #         torch.all(  # all x1 in (0, eps)
        #             all(0 < pts[..., 0, 0].item() < eps)
        #         )
        #     or
        #         torch.all(  # all x1 in (0, eps)
        #             all(eps < pts[..., -1, 1].item() < 1)
        #         )
        #     ) for pts in (upper_points, lower_points):
        #     raise ValueError("x-coordinates must be in [0, 1] and endpoints should be at x = 0 and 1.")

        def _ensure_batched(points: torch.Tensor, name: str) -> torch.Tensor:
            pts = torch.as_tensor(points, dtype=torch.float32, device=target_device)
            if pts.ndim == 2:
                pts = pts.unsqueeze(0)
            if pts.ndim != 3 or pts.shape[-1] != 2:
                raise ValueError(
                    f"{name} must have shape [N, 2] or [B, N, 2]; received {tuple(points.shape)}"
                )
            return pts  # [B, N, 2]

        upper_points = _ensure_batched(upper_points, "upper_points")
        lower_points = _ensure_batched(lower_points, "lower_points")
        if upper_points.shape[0] != lower_points.shape[0]:
            raise ValueError(
                "Upper and lower surfaces must share the same batch size; "
                f"got {upper_points.shape[0]} and {lower_points.shape[0]}."
            )
        min_samples = n_coefficients + (2 if use_kulfan_modifiers else 0)
        if upper_points.shape[-2] < min_samples or lower_points.shape[-2] < min_samples:
            raise ValueError(
                "Insufficient points to fit requested order. "
                f"Need at least {min_samples} points per surface for "
                f"{n_coefficients} coefficients, got "
                f"{upper_points.shape[-2]} and {lower_points.shape[-2]}."
            )

        x_upper = upper_points[..., 0]  # [B, N_u]
        x_lower = lower_points[..., 0]  # [B, N_l]
        y_upper = upper_points[..., 1]  # [B, N_u]
        y_lower = lower_points[..., 1]  # [B, N_l]

        batch_size = upper_points.shape[0]
        dtype = upper_points.dtype
        eps = torch.finfo(dtype).eps

        # Basis construction parameters
        k = torch.arange(n_coefficients, device=target_device, dtype=dtype)  # [K]
        n = n_coefficients - 1
        n_plus_1 = torch.full((n_coefficients,), n + 1.0, dtype=dtype, device=target_device)
        binom = torch.exp(
            torch.lgamma(n_plus_1)
            - torch.lgamma(k + 1.0)
            - torch.lgamma(n_plus_1 - k)
        ).unsqueeze(0)  # [1, K]

        # Upper surface design matrix
        Cx_upper = torch.pow(x_upper, n1) * torch.pow(1.0 - x_upper, n2)  # [1, N_upper]
        Bx_upper = torch.pow(x_upper.unsqueeze(-1), k) * torch.pow(1.0 - x_upper.unsqueeze(-1), n - k)  # [1, N_upper, K]
        Mx_upper = Cx_upper.unsqueeze(-1) * Bx_upper * binom  # [1, N_upper, K]

        # Lower surface design matrix
        Cx_lower = torch.pow(x_lower, n1) * torch.pow(1.0 - x_lower, n2)  # [1, N_lower]
        Bx_lower = torch.pow(x_lower.unsqueeze(-1), k) * torch.pow(1.0 - x_lower.unsqueeze(-1), n - k)  # [1, N_lower, K]
        Mx_lower = Cx_lower.unsqueeze(-1) * Bx_lower * binom  # [1, N_lower, K]

        if use_kulfan_modifiers:
            # Shared LE modifier
            le_mod_upper = x_upper * torch.pow(1.0 - x_upper, n + 0.5)  # [N_upper]
            le_mod_lower = x_lower * torch.pow(1.0 - x_lower, n + 0.5)  # [N_lower]

            # TE modifier (with sign for surface)
            te_mod_upper = x_upper / 2.0  # [1, N_upper] (positive for upper)
            te_mod_lower = -x_lower / 2.0  # [1, N_lower] (negative for lower)

            # Zeros for the off-diagonal blocks
            # zeros_upper: [B, N_upper, K]
            zeros_upper = torch.zeros(batch_size, x_upper.size(-1), n_coefficients, device=target_device, dtype=dtype)
            # zeros_lower: [B, N_lower, K]
            zeros_lower = torch.zeros(batch_size, x_lower.size(-1), n_coefficients, device=target_device, dtype=dtype)

            # Construct the Upper Block: [Mx_upper | 0 | le_mod | te_mod]
            # Dimensions: [B, N_upper, K + K + 1 + 1] = [B, N_upper, 2K+2]
            M_upper = torch.cat(
                [
                    Mx_upper,                     # Upper coeffs basis
                    zeros_upper,                  # Lower coeffs basis (zeros)
                    le_mod_upper.unsqueeze(-1),   # Shared LE weight basis
                    te_mod_upper.unsqueeze(-1),   # Shared TE thickness basis
                ],
                dim=-1,
            )  # [B, N_upper, 2K+2]

            # Construct the Lower Block: [0 | Mx_lower | le_mod | te_mod]
            # Dimensions: [B, N_lower, 2K+2]
            M_lower = torch.cat(
                [
                    zeros_lower,                  # Upper coeffs basis (zeros)
                    Mx_lower,                     # Lower coeffs basis
                    le_mod_lower.unsqueeze(-1),   # Shared LE weight basis
                    te_mod_lower.unsqueeze(-1),   # Shared TE thickness basis
                ],
                dim=-1,
            )  # [B, N_lower, 2K+2]

            # Combined design matrix and targets
            M_combined = torch.cat([M_upper, M_lower], dim=1)  # [1, N_upper + N_lower, 2*K+2]
            y_combined = torch.cat([y_upper, y_lower], dim=1).unsqueeze(-1)  # [1, N_total, 1]

            # Solve joint least squares
            solution = torch.linalg.lstsq(M_combined, y_combined, rcond=rcond).solution.squeeze()  # [B>1, 2*K+2]

            # Extract parameters
            upper_coeffs = solution[..., :n_coefficients]
            lower_coeffs = solution[..., n_coefficients : 2 * n_coefficients]
            shared_le_weight = solution[..., -2]
            shared_te_thickness_signed = solution[..., -1]

            # first coefficient on either side must be clamped
            upper_coeffs[..., 0] = torch.clamp(upper_coeffs[..., 0], min=eps)
            lower_coeffs[..., 0] = torch.clamp(lower_coeffs[..., 0], max=-eps)

            # Ensure TE thickness is positive
            shared_te_thickness = torch.abs(shared_te_thickness_signed)

            # Infer surface types from signs
            # upper_sign = torch.sign(upper_coeffs[0])
            # lower_sign = torch.sign(lower_coeffs[0])
            # te_sign = torch.sign(shared_te_thickness_signed)

            # if upper_sign != te_sign or lower_sign != -te_sign:
            #     raise ValueError("Inconsistent surface signs in fitted solution.")

            upper_curve = KulfanModifiedCST(
                coefficients=upper_coeffs,
                leading_edge_weight=shared_le_weight,
                trailing_edge_thickness=shared_te_thickness,
                surface_type="upper",
                n1=n1,
                n2=n2,
                device=target_device,
            )
            lower_curve = KulfanModifiedCST(
                coefficients=lower_coeffs,
                leading_edge_weight=shared_le_weight,
                trailing_edge_thickness=shared_te_thickness,
                surface_type="lower",
                n1=n1,
                n2=n2,
                device=target_device,
            )
        else:
            # Plain CST: no shared modifiers
            M_combined = torch.cat([Mx_upper, Mx_lower], dim=0)
            y_combined = torch.cat([y_upper, y_lower], dim=0).unsqueeze(-1)
            solution = torch.linalg.lstsq(M_combined, y_combined, rcond=rcond).solution.squeeze(-1)
            upper_coeffs = solution[:n_coefficients]
            lower_coeffs = solution[n_coefficients:]

            upper_curve = TorchCSTCurve(upper_coeffs, n1=n1, n2=n2, device=target_device)
            lower_curve = TorchCSTCurve(lower_coeffs, n1=n1, n2=n2, device=target_device)

        airfoil = cls(
            upper_surface=upper_curve,
            lower_surface=lower_curve,
            device=target_device,
            # allow_overlap=not prevent_overlap,
        )
        # if prevent_overlap:
        #     airfoil._enforce_positive_thickness()
        return airfoil

    # !!! Untested !!!
    def _enforce_positive_thickness(self, max_iterations: int = 8) -> None:
        """Apply minimal vertical offsets to eliminate overlaps.
        Corrective method that iteratively adjusts upper and lower surfaces
        to ensure positive thickness along the chord.

        Note that this modifies the original airfoil!

        Args:
            max_iterations (int): Maximum adjustment passes.

        Raises:
            ValueError: If positive thickness cannot be achieved.
        """
        raise NotImplementedError(
            "This method is untested and may not work as intended."
        )
        if self.allow_overlap:
            return

        beta = torch.linspace(0.0, torch.pi, 512, device=self.device)
        x = 0.5 * (1.0 - torch.cos(beta))

        for _ in range(max_iterations):
            thickness = self.thickness(x)
            min_gap = thickness.min(dim=-1).values
            if torch.all(min_gap > self.overlap_tolerance):
                return
            correction = (self.overlap_tolerance - min_gap).clamp_min(0.0).unsqueeze(-1)
            self.upper_offset += 0.5 * correction
            self.lower_offset -= 0.5 * correction

        if torch.any(self.thickness(x) <= 0):
            raise ValueError("Unable to eliminate overlap while preserving curve shapes.")

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
