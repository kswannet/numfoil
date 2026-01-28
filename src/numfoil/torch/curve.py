import numpy as np

import torch
import torch.nn as nn
from torch.func import vmap

from typing import Union, Tuple, Optional, Literal
from functools import cached_property

import warnings


class TorchCSTCurve(nn.Module):
    """
    PyTorch implementation of a CST curve for differentiable airfoil design.

    The standard CST formulation: y(x) = C(x) * S(x)
    where:
        :math:`C(x) = x^{n_1} (1 - x)^{n_2}`  (class function)
        :math:`S(x) = \sum_{k=0}^n a_k B_{n,k}(x)`  (shape function)
        :math:`B_{n,k}(x) = \binom{n}{k} x^k (1-x)^{n-k}`  (Bernstein basis polynomials)

    where:
        - k is the index of the Bernstein polynomial
        - n is the degree of the polynomial (n = number of coefficients - 1)
        - n1, n2 are the class function exponents
        - a_k are the shape coefficients (the "CST coefficients")
        - B_{n,k}(x) are the Bernstein basis polynomials of degree n
        - x is the normalized chordwise position in [0, 1]

    Dimension Convention:
        - B: batch size (number of curves)
        - X: number of evaluation points
        - K: number of coefficients
        - coefficients: always stored as [B, K] (batch format)
        - x: always processed as [X] (1D evaluation points)
        - Outputs: [B, X] internally, squeezed to [X] if B=1

    Attributes:
        coefficients (torch.Tensor): [B, K]
            Learnable coefficients a_k for the shape function
        n1 (float):
            Leading edge class exponent C(x) (0.5 for normal airfoils)
        n2 (float):
            Trailing edge class exponent C(x) (1.0 for normal airfoils)
        device (torch.device):
            Device to store tensors on (CPU or GPU)
        k_vec (torch.Tensor):
            Vector of Bernstein indices
        binomial_weights (torch.Tensor):
            Precomputed binomial coefficients for Bernstein polynomials

    Methods:
        class_function(x): Compute the class function C(x)
        bernstein_basis(x): Compute the Bernstein basis functions B_{n,k}(x)
        shape_function(x): Compute the shape function S(x)
        forward(x): Evaluate the CST curve at points x
        evaluate_points(x): Return (x,y) coordinates of the curve
        first_derivative(x): Compute first derivative using autograd
        second_derivative(x): Compute second derivative using autograd
        curvature(x): Compute curvature using first and second derivatives
        from_numpy(coefficients, n1, n2): Create a TorchCSTCurve from numpy
            coefficients
    """

    def __init__(
        self,
        coefficients: Union[torch.Tensor, np.ndarray],
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
    ):
        """Initialize the TorchCSTCurve
        Args:
            coefficients (Union[torch.Tensor, np.ndarray]):
                Bernstein weight coefficients ``a_k`` for the shape function
                - Shape (n_coeffs,) for a single curve
                - Shape (batch_size, n_coeffs) for multiple curves
            n1 (float): Leading-edge exponent for the class function C(x)
                Default is 0.5 (typical for airfoils)
            n2 (float): Trailing-edge exponent for the class function C(x)
                Default is 1.0 (typical for airfoils)
            device (Optional[torch.device | str]): Device to store tensors on.
                If None, will use 'cuda' if available, else 'cpu'.

        Raises:
            ValueError: If coefficients is not 1D or 2D
        """
        super().__init__()
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        coefficients = torch.as_tensor(
            coefficients, dtype=torch.float32, device=self.device
        )
        # Ensure batch dimension [K] → [B, K]
        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)  # [K] → [B=1, K]
        elif coefficients.ndim != 2:
            raise ValueError("coefficients must be 1D or 2D")

        # Register coefficients as a parameter so they can be optimized
        self.register_buffer(
            "coefficients", coefficients  # [B, K]
        )

        # Class function exponents
        self.n1 = n1
        self.n2 = n2

        # Bernstein indices vector [K]
        self.register_buffer(
            "k_vec",
            torch.arange(self.n_coefficients, dtype=torch.float32, device=self.device)
        )

        # Precompute binomial weights [K]
        self.register_buffer(
            "binomial_weights", self._compute_binomial_weights()
        )

        self.eps = torch.finfo(coefficients.dtype).eps

    # @property
    # def coefficients(self) -> torch.Tensor:
    #     """Bernstein weight coefficients a_k for the shape function, shape [B, K]"""
    #     return self._coefficients

    @property
    def batch_size(self) -> int:
        """Number of curves in the batch (B)"""
        return self.coefficients.shape[0]

    @property
    def n_coefficients(self) -> int:
        """Number of coefficients in the shape function"""
        return self.coefficients.shape[-1]

    @property
    def degree(self) -> int:
        """Degree of the Bernstein polynomial"""
        return self.n_coefficients - 1

    # Aliases for clarity
    @property
    def k(self) -> torch.Tensor:
        """Alias for k vector for Bernstein polynomials"""
        return self.k_vec

    @property
    def n(self) -> int:
        """Alias for degree of the Bernstein polynomial"""
        return self.degree

    @property
    def fitted_points(self) -> torch.Tensor | None:
        """Original points used for fitting, if available."""
        return getattr(self, "_fitted_points", None)

    @staticmethod
    def _ensure_1d(x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is at least 1D.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: The input reshaped to a vector if necessary.
        """
        return x.unsqueeze(0) if x.dim() == 0 else x  # [X]

    def _prepare_input(self, x: torch.Tensor | np.ndarray | float) -> torch.Tensor:
        """Prepare input tensor by ensuring it's 1D and on the correct device

        Args:
            x: Evaluation points in [0, 1]

        Returns:
            torch.Tensor: input tensor, shape [X]

        Raises:
            ValueError: If input x is not 1D or scalar
        """
        x = self._ensure_1d(
            x.to(dtype=self.coefficients.dtype, device=self.coefficients.device) \
            if torch.is_tensor(x) else \
            torch.as_tensor(x, dtype=self.coefficients.dtype, device=self.coefficients.device)
        )

        if x.ndim > 1:
            raise ValueError("Input x must be 1D or scalar")

        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("x must be in the range [0, 1]")

        return x  # [X]

    def _compute_binomial_weights(self) -> torch.Tensor:
        """Compute binomial coefficients for Bernstein polynomials
        Use log-space to avoid numerical issues with large factorials

        :math:`\binom{n}{k} = \frac{n!}{k! (n-k)!}`

        where
            - n is the degree of the polynomial (n = number of coefficients - 1)
            - k is the index of the Bernstein polynomial

        returns:
            torch.Tensor: Binomial coefficients of shape [K]
        """
        # pre-create tensor of n+1 values
        n_plus_1 = torch.full((self.n_coefficients,), self.n + 1.0, device=self.device)
        # log-space computation of binomial coefficients (n choose k)
        log_binom = (
            torch.lgamma(n_plus_1)
            - torch.lgamma(self.k + 1.0)
            - torch.lgamma(n_plus_1 - self.k)
        )
        return torch.exp(log_binom)  # [K]

    def class_function(self, x: torch.Tensor) -> torch.Tensor:
        """Class function for the CST curve

        :math:`C(x) = x^{n_1} (1 - x)^{n_2}`

        args:
            x (torch.Tensor): Input tensor of shape [X] with values in [0, 1]

        returns:
            torch.Tensor: Output tensor of shape [B, X] representing C(x) for
                each curve in the batch.
        """
        x = self._prepare_input(x)  # [X]
        Cx = torch.pow(x, self.n1) * torch.pow(1.0 - x, self.n2)  # [X]
        return Cx.unsqueeze(0).expand(self.batch_size, -1)  # [X] → [1, X] → [B, X]

    def bernstein_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the Bernstein polynomial basis functions

        :math:`B_{n,k}(x) = x^k (1-x)^{n-k}`

        where
            - n is the degree of the polynomial (n = number of coefficients - 1)
            - k is the index of the Bernstein polynomial

        args:
            x (torch.Tensor): Input tensor of shape [X] with values in [0, 1]

        returns:
            torch.Tensor: Output tensor of shape [X, K] representing the
                basis functions. The last dimension corresponds to k=0,...,n. So
                column k contains B_{n,k}(x).
        """
        # Reshape x for broadcasting with k
        x = self._prepare_input(x).unsqueeze(-1)  # [X] → [X, 1]
        return torch.pow(x, self.k) * torch.pow(1.0 - x, self.n - self.k)  # [X, K]

    def weighted_basis_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the weighted Bernstein polynomial basis matrix.
        The last dimension corresponds to k=0,...,n.
        So column k contains W_{n,k}(x).

        :math:`W_{n,k}(x) = \binom{n}{k} B_{n,k}(x)`
        :math:`M(x) = C(x) * B(x) * binom`

        where
            - n is the degree of the polynomial (n = number of coefficients - 1)
            - k is the index of the Bernstein polynomial

        args:
            x (torch.Tensor): Input tensor of shape (...,) with values in [0, 1]

        returns:
            torch.Tensor: shape [B, X, K] weighted basis matrix
        """
        Cx = self.class_function(x).unsqueeze(-1)                   # [B, X, 1]
        basis = self.bernstein_basis(x).unsqueeze(0)                # [1, X, K]
        weights = self.binomial_weights.unsqueeze(0).unsqueeze(0)   # [1, 1, K]
        return Cx * basis * weights                                 # [B, X, K]

    def shape_function(self, x: torch.Tensor) -> torch.Tensor:
        """Shape function ``S(x)``

        :math:`S(x) = \sum_{k=0}^{n} a_k B_{n,k}(x)`

        Broadcasts [B, K] weights with [X, K] basis to get [B, X, K],
        then sum over K.

        Args:
            x: Chordwise coordinates in ``[0, 1]``.

        Returns:
            torch.Tensor: Shape function values with the same leading shape as ``x``.

        Notes:
            Multiplication by the pre-computed binomial coefficients keeps the expression
            numerically identical to the standard CST definition.

        **Dimension flow:**
            - basis: [X, K] from bernstein_basis
            - coefficients: [B, K]
            - binomial_weights: [K]
            - weights = coefficients * binomial_weights: [B, K]
            - weights.unsqueeze(1): [B, K] → [B, 1, K]
            - basis.unsqueeze(0): [X, K] → [1, X, K]
            - Broadcasting: [B, 1, K] * [1, X, K] → [B, X, K]
            - Sum over K: [B, X, K] → [B, X]

        """
        basis = self.bernstein_basis(x)  # [X, K]
        weights = self.coefficients * self.binomial_weights  # [B, K], [B, K] * [K] → [B, K]
        weighted_basis = weights.unsqueeze(1) * basis.unsqueeze(0)  # [B, X, K], [B, 1, K] * [1, X, K] → [B, X, K]

        return weighted_basis.sum(dim=-1)  # [B, X, K] → sum(-1) → [B, X]

    C = class_function
    S = shape_function
    B = bernstein_basis
    M = weighted_basis_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the full CST curve

        :math:`y(x) = C(x) * S(x)`

        An explicit range check protects against accidental extrapolation,
        which can lead to complex numbers in the class function and undefined
        gradients.

        Args:
            x: Chordwise coordinates in ``[0, 1]``.

        Returns:
            torch.Tensor:
                - Shape [X] if single curve (B=1)
                - Shape [B, X] if batch (B>1)

        Raises:
            ValueError: If any element of ``x`` falls outside ``[0, 1]``.

        Dimension flow:
            - class_function(x): [B, X]
            - shape_function(x): [B, X]
            - Element-wise multiply: [B, X]
            - Squeeze if B=1: [1, X] → [X] or [B, X] → [B, X]
        """
        x = self._prepare_input(x)  # [X]

        # validate input range
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("x must be in the range [0, 1]")

        # yx = self.C(x) * self.S(x)  # [B, X]
        yx = self.class_function(x) * self.shape_function(x)  # [B, X]

        return yx.squeeze(0)  # [X] if B=1 else [B, X]

    def evaluate_at(self, x: torch.Tensor) -> torch.Tensor:
        """Return stacked ``(x, y)`` coordinates for convenience.

        Args:
            x: Chordwise coordinates in ``[0, 1]``, shape [X].

        Returns:
            torch.Tensor: Coordinates of shape [X, 2], or [B, X, 2].
        """
        x = self._prepare_input(x)  # [X]
        y = self.forward(x)  # [X] or [B, X]
        if y.ndim == 1:
            # Single curve: y is shape [X]
            return torch.stack([x, y], dim=-1)  # [X, 2]
        else:
            # Batch: y is shape [B, X], so expand x to [B, X]
            return torch.stack([x.unsqueeze(0).expand_as(y) , y], dim=-1)  # [B, X, 2]

    def alt_evaluate_at(self, x: torch.Tensor) -> torch.Tensor:
        """Return stacked ``(x, y)`` coordinates for convenience.

        :math:`y(x) = M(x) * a`

        Args:
            x: Chordwise coordinates in ``[0, 1]``, shape [X].

        Returns:
            torch.Tensor: Coordinates of shape [X, 2], or [B, X, 2].

        dimension flow:
            - M(x): [B, X, K]
            - coefficients: [B, K] → unsqueeze(-1) → [B, K, 1]
            - matmul: [B, X, K] @ [B, K, 1] → [B, X, 1] → squeeze(-1) → [B, X]
        """
        y =  torch.matmul(
            self.M(x),                       # [B, X, K]
            self.coefficients.unsqueeze(-1)  # [B, K, 1]
        ).squeeze(-1)  # [B, X]
        if y.ndim == 1:
            # Single curve: y is shape [X]
            return torch.stack([x, y], dim=-1)  # [X, 2]
        else:
            # Batch: y is shape [B, X], so expand x to [B, X]
            return torch.stack([x.unsqueeze(0).expand_as(y) , y], dim=-1)  # [B, X, 2]

    #  ====== Derivatives ======
    def class_first_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute first derivative of class function analytically.

        :math:`C'(x) = n_1 x^{n_1-1}(1-x)^{n_2} - n_2 x^{n_1}(1-x)^{n_2-1}`

        Args:
            x: Chordwise locations in [0, 1], shape [X].

        Returns:
            torch.Tensor: C'(x), shape [B, X].

        Notes:
            Epsilon clamping prevents division by zero when computing gradients
            near endpoints.

        Dimension flow:
            - Input: x [X]
            - Computation: [X] element-wise operations
            - Expansion: [X] → [B, X]
        """
        x = self._prepare_input(x)              # [X]
        one_minus_x = (1.0 - x).clamp_min(self.eps)  # [X]
        x = x.clamp_min(self.eps)                    # [X]

        dCx = (
            self.n1 * torch.pow(x, self.n1 - 1) * torch.pow(one_minus_x, self.n2)
            - self.n2 * torch.pow(x, self.n1) * torch.pow(one_minus_x, self.n2 - 1)
        )  # [X]
        return dCx.unsqueeze(0).expand(self.batch_size, -1)  # [B, X]

    def class_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute second derivative of class function analytically.

        :math:`C''(x) = n_1 (n_1-1) x^{n_1-2} (1-x)^{n_2} - 2n_1 n_2
        x^{n_1-1} (1-x)^{n_2-1} + n_2 (n_2-1) x^{n_1} (1-x)^{n_2-2}`

        Args:
            x: Chordwise locations in [0, 1], shape [X].

        Returns:
            torch.Tensor: C''(x), shape [B, X].

        Notes:
            Epsilon clamping prevents division by zero when computing gradients
            near endpoints.

        Dimension flow:
            - Input: x [X]
            - Computation: [X]
            - Expansion: [X] → [B, X]
        """
        x = self._prepare_input(x)                   # [X]
        one_minus_x = (1.0 - x).clamp_min(self.eps)  # [X]
        x = x.clamp_min(self.eps)                    # [X]

        d2Cx = (
            self.n1 * (self.n1 - 1) * torch.pow(x, self.n1 - 2) * torch.pow(one_minus_x, self.n2)
            - 2.0 * self.n1 * self.n2 * torch.pow(x, self.n1 - 1) * torch.pow(one_minus_x, self.n2 - 1)
            + self.n2 * (self.n2 - 1) * torch.pow(x, self.n1) * torch.pow(one_minus_x, self.n2 - 2)
        )  # [X]
        return d2Cx.unsqueeze(0).expand(self.batch_size, -1)  # [B, X]

    def shape_first_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute first derivative of shape function analytically.

        :math:`S'(x) = \sum a_k \binom{n}{k} [k x^{k-1}(1-x)^{n-k} - (n-k)x^k(1-x)^{n-k-1}]`

        Args:
            x: Chordwise locations in [0, 1], shape [X].

        Returns:
            torch.Tensor: S'(x), shape [B, X].

        Notes:
            Epsilon clamping prevents division by zero when computing gradients
            near endpoints.

        Dimension flow:
            - Input: x [X]
            - Unsqueeze: [X] → [X, 1] for k-broadcasting
            - Basis derivative: [X, 1] * [K] → [X, K]
            - weights: [B, K]
            - Broadcasting: [B, 1, K] * [1, X, K] → [B, X, K]
            - Sum: [B, X, K] → [B, X]
        """
        x = self._prepare_input(x)          # [X]

        # Clamp and unsqueeze for k-broadcasting: [X] → [X, 1]
        one_minus_x = (1.0 - x).clamp_min(self.eps).unsqueeze(-1)  # [X, 1]
        x = x.clamp_min(self.eps).unsqueeze(-1)                    # [X, 1]
        weights = self.coefficients * self.binomial_weights   # [B, K]

        # Compute basis derivative analytically: [X, 1] * [K] → [X, K]
        basis_derivative = (
            self.k * torch.pow(x, self.k - 1) * torch.pow(one_minus_x, self.n - self.k)
            - (self.n - self.k) * torch.pow(x, self.k) * torch.pow(one_minus_x, self.n - self.k - 1)
        )  # [X, K]

        return torch.sum(
            weights.unsqueeze(1) * basis_derivative.unsqueeze(0),  # [B, 1, K] * [1, X, K]
            dim=-1
        )  # [B, X, K] → sum(-1) → [B, X]

    def shape_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute second derivative of shape function analytically.

        :math:`S''(x) = \sum a_k \binom{n}{k} [k(k-1)x^{k-2}(1-x)^{n-k} -
        2k(n-k)x^{k-1}(1-x)^{n-k-1} + (n-k)(n-k-1)x^k(1-x)^{n-k-2}]`

        Args:
            x: Chordwise locations in [0, 1], shape [X].

        Returns:
            torch.Tensor: S''(x), shape [B, X].

        Notes:
            Epsilon clamping prevents division by zero when computing gradients
            near endpoints.

        Dimension flow:
            - Input: x [X]
            - Basis 2nd derivative: [X, K]
            - Broadcasting with weights: [B, X, K]
            - Sum: [B, X]
        """
        x = self._prepare_input(x)  # [X]

        # Clamp and unsqueeze for k-broadcasting: [X] → [X, 1]
        one_minus_x = (1.0 - x).clamp_min(self.eps).unsqueeze(-1)  # [X, 1]
        x = x.clamp_min(self.eps).unsqueeze(-1)                    # [X, 1]

        weights = self.coefficients * self.binomial_weights   # [B, K]

        basis_second_derivative = (
            self.k * (self.k - 1) * torch.pow(x, self.k - 2) * torch.pow(one_minus_x, self.n - self.k)
            - 2.0 * self.k * (self.n - self.k) * torch.pow(x, self.k - 1) * torch.pow(one_minus_x, self.n - self.k - 1)
            + (self.n - self.k) * (self.n - self.k - 1) * torch.pow(x, self.k) * torch.pow(one_minus_x, self.n - self.k - 2)
        )  # [X, K]

        return torch.sum(
            weights.unsqueeze(1) * basis_second_derivative.unsqueeze(0),  # [B, 1, K] * [1, X, K]
            dim=-1
        )  # [B, X, K] → sum(-1) → [B, X]

    #  ==== Aliases ====
    dC = class_first_derivative
    d2C = class_second_derivative
    dS = shape_first_derivative
    d2S = shape_second_derivative

    def first_derivative_at(
        self,
        x: torch.Tensor,
        mode: str = "analytic",
    ) -> torch.Tensor:
        """Compute first derivative of the CST curve.

        :math:`dy/dx = C'(x) S(x) + C(x) S'(x)`.

        Args:
            x: Chordwise locations in [0, 1], shape [X].
            mode: ``"analytic"`` for closed-form or ``"autograd"`` for automatic differentiation.
                Default is ``"analytic"``.

        Returns:
            torch.Tensor:
                - Shape [X] if single curve (B=1)
                - Shape [B, X] if batch (B>1)

        Raises:
            ValueError: If mode is not recognized.
        """
        if mode == "analytic":
            # return self.dC(x) * self.S(x) + self.C(x) * self.dS(x)
            return (
                self.class_first_derivative(x) * self.shape_function(x)
                + self.class_function(x) * self.shape_first_derivative(x)
            ).squeeze()

        if mode == "autograd":
            x = (
                self._prepare_input(x)
                .clamp_min(self.eps)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            def grad_fn(x):
                y = self.forward(x)  # [X] or [B, X]
                return torch.autograd.grad(
                    y, x, torch.ones_like(y), create_graph=True
                )[0]
            if self.batch_size == 1:
                return grad_fn(x)  # [X]
            else:
                # map the grad_fn over the batch dimension
                return vmap(grad_fn)(
                    x.unsqueeze(0).expand(self.batch_size, -1)
                )  # [B, X]

        raise ValueError("mode must be 'analytic' or 'autograd'.")

    def second_derivative_at(
        self,
        x: torch.Tensor,
        mode: str = "analytic",
    ) -> torch.Tensor:
        """Compute second derivative of the CST curve.

        :math:`d^2y/dx^2 = C''(x) S(x) + 2 C'(x) S'(x) + C(x) S''(x)`.

        Args:
            x: Chordwise locations in [0, 1], shape [X].
            mode: ``"analytic"`` or ``"autograd"``. Default is ``"analytic"``.

        Returns:
            torch.Tensor:
                - Shape [X] if single curve (B=1)
                - Shape [B, X] if batch (B>1)

        Raises:
            ValueError: If mode is not recognized.
        """
        if mode == "analytic":
            # return (
            #     self.d2C(x) * self.S(x)
            #     + 2.0 * self.dC(x) * self.dS(x)
            #     + self.C(x) * self.d2S(x)
            # )
            return (
                self.class_second_derivative(x) * self.shape_function(x)
                + 2.0 * self.class_first_derivative(x) * self.shape_first_derivative(x)
                + self.class_function(x) * self.shape_second_derivative(x)
            )

        if mode == "autograd":
            x = (
                self._prepare_input(x)
                .clamp_min(self.eps)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            def grad_fn(x):
                y = self.forward(x)
                dy = torch.autograd.grad(
                    y, x, torch.ones_like(y), create_graph=True
                )[0]
                ddy = torch.autograd.grad(
                    dy, x, torch.ones_like(dy), create_graph=True
                )[0]
                return ddy

            if self.batch_size == 1:
                return grad_fn(x)  # [X]
            else:
                # map the grad_fn over the batch dimension
                return vmap(grad_fn)(
                    x.unsqueeze(0).expand(self.batch_size, -1)
                )  # [B, X]

        raise ValueError("mode must be 'analytic' or 'autograd'.")

    def tangent_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        """Compute tangent vector at each point x.

        Args:
            x: Chordwise locations in :math:`[0, 1]`.
            mode: Derivative mode (``"analytic"`` or ``"autograd"``).

        Returns:
            torch.Tensor: Tangent angle values in radians.
                - Shape [X, 2] if single curve
                - Shape [B, X, 2] if batch
        """
        dy_dx = self.first_derivative_at(x, mode=mode)
        dx = torch.ones_like(dy_dx)
        tangent = torch.stack([dx, dy_dx], dim=-1)
        magnitude = torch.sqrt(1.0 + dy_dx**2).unsqueeze(-1)
        return (tangent / magnitude).squeeze()

    def normal_at(
        self,
        x: torch.Tensor,
        mode: str = "analytic",
        direction: Literal["outward", "inward"] = "outward",
    ) -> torch.Tensor:
        """Compute normal vector at each point x.

        Args:
            x (torch.Tensor): Chordwise locations in :math:`[0, 1]`.
            mode (str): Derivative mode (``"analytic"`` or ``"autograd"``).
                Default is ``"analytic"``.
            direction (str): Normal direction ``"outward"`` or ``"inward"``.
                Default is ``"outward"``.

        Returns:
            torch.Tensor: Normal angle values in radians.
            - Shape [X, 2] if single curve
            - Shape [B, X, 2] if batch
        """
        dy_dx = self.first_derivative_at(x, mode=mode)
        dx = torch.ones_like(dy_dx)

        if direction == "outward":   # turn 90 degrees CCW
            normal = torch.stack([-dy_dx, dx], dim=-1)
        elif direction == "inward":  # turn 90 degrees CW
            normal = torch.stack([dy_dx, -dx], dim=-1)
        else:
            raise ValueError("direction must be 'outward' or 'inward'.")

        magnitude = torch.sqrt(1.0 + dy_dx**2).unsqueeze(-1)
        return (normal / magnitude).squeeze()

    def curvature_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        """
        Compute curvature :math:`\kappa(x) = y'' / (1 + (y')^2)^{3/2}`.

        Args:
            x: Chordwise locations in :math:`[0, 1]`.
            mode: Derivative mode (``"analytic"`` or ``"autograd"``).

        Returns:
            torch.Tensor: Curvature values.
        """
        dy_dx = self.first_derivative_at(x, mode=mode)
        d2y_dx2 = self.second_derivative_at(x, mode=mode)
        return (d2y_dx2 / torch.pow(1.0 + dy_dx**2, 1.5)).squeeze()

    def radius_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        """
        Compute radius of curvature :math:`R(x) = 1 / \kappa(x)`.

        Args:
            x: Chordwise locations in :math:`[0, 1]`.
            mode: Derivative mode (``"analytic"`` or ``"autograd"``).

        Returns:
            torch.Tensor: Radius of curvature values.
        """
        curvature = self.curvature_at(x, mode=mode)
        return (1.0 / curvature).squeeze()

    @classmethod
    def from_numpy(cls, coefficients: np.ndarray, n1: float = 0.5, n2: float = 1.0, device: str = "cpu") -> "TorchCSTCurve":
        """
        Instantiate from NumPy coefficients.

        Args:
            coefficients: Bernstein weights :math:`a_k`.
            n1: Leading-edge exponent.
            n2: Trailing-edge exponent.

        Returns:
            TorchCSTCurve: Differentiable CST curve.
        """
        return cls(torch.tensor(coefficients, dtype=torch.float32), n1, n2, device=device)

    def plot(
        self,
        idx: int = None,
        n_points: int = 1000,
        spacing: str = "cosine",
        ax=None,
        fig=None,
        **plot_kwargs,
    ):
        """
        Plot the CST curve using Matplotlib.

        Args:
            n_points: Number of points to evaluate along the chord.
            spacing: "uniform"|"linear" or "cosine" spacing of x points.
            ax: Matplotlib Axes object to plot on. If None, creates a new figure.
            **plot_kwargs: Additional keyword arguments for plt.plot().

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                Figure and Axes objects containing the plot.
        """
        import matplotlib.pyplot as plt

        x = torch.linspace(0.0, 1.0, n_points, device=self.coefficients.device)
        if spacing == "cosine":
            x = 0.5 * (1.0 - torch.cos(torch.pi * x))  # Cosine spacing

        y = self.forward(x).cpu().detach().numpy()

        if ax is None:
            fig, ax = plt.subplots()

        if self.batch_size == 1:
            ax.plot(x.cpu().numpy(), y, **plot_kwargs)
        elif idx is not None:
            ax.plot(x.cpu().numpy(), y[idx], label=f"Curve {idx+1}/{self.batch_size}", **plot_kwargs)
        else:
            for i in range(self.batch_size):
                ax.plot(x.cpu().numpy(), y[i], label=f"Curve {i+1}", **plot_kwargs)
            ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("CST Curve")
        ax.grid(True)
        return fig, ax

    @classmethod
    def fit(
        cls,
        points: torch.Tensor,
        n_coefficients: int,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
        rcond: Optional[float] = None,
    ) -> "TorchCSTCurve":
        """
        Fit CST coefficients to coordinates via batched least squares.

        The routine is fully differentiable: gradients flow from the fitted curve
        back to the input coordinates, making it suitable inside training loops.

        Args:
            points: Tensor of shape [N, 2] or [B, N, 2] with (x, y) samples.
            n_coefficients: Number of coefficients (degree + 1).
            n1: Leading-edge exponent.
            n2: Trailing-edge exponent.
            device: Optional device override.
            rcond: Optional cutoff passed to torch.linalg.lstsq.

        Returns:
            TorchCSTCurve with fitted coefficients.

        Raises:
            ValueError: For invalid shapes, insufficient points, or x outside
            [0, 1].

        Notes:
            In contrast to the other methods, here N is used to denote the
            number of input points, rather than X. This is to avoid confusion
            with the number of x-values used to evaluate the curve, which is the
            main purpose in the other methods.
            so, N=number of coordinate points (x,y),
            and X=number of evaluation locations x.
            K is used for the number of coefficients a_k.
        """
        device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        points = torch.as_tensor(points, dtype=torch.float32, device=device)

        dtype = points.dtype if points.dtype.is_floating_point else torch.float32
        points = points.to(device=device, dtype=dtype)

        # Validate input shape
        if points.ndim not in (2, 3) or points.shape[-1] != 2:
            raise ValueError(
                "points must have shape [N, 2] or [B, N, 2]; "
                f"got tensor with shape {tuple(points.shape)}"
            )

        # Add batch dimension for single curve
        # single_curve = points.ndim == 2
        if points.ndim == 2:
            points = points.unsqueeze(0)  # [N, 2] → [B=1, N, 2]

        # validate sufficient points
        B, N, _ = points.shape
        if N < n_coefficients:
            raise ValueError(
                f"Insufficient points to fit {n_coefficients} coefficients " \
                f"(received N={N}, need N>={n_coefficients} points)."
            )

        x = points[..., 0]  # [B, N]
        y = points[..., 1]  # [B, N]

        # validate x range
        if torch.any((x < 0) | (x > 1)):
            raise ValueError("x coordinates must lie within [0, 1].")

        # clamp x values to avoid numerical issues at endpoints
        eps = torch.finfo(dtype).eps
        x = x.clamp(min=eps, max=1.0 - eps)

        # Parameters needed for basis construction
        k = torch.arange(n_coefficients, device=device, dtype=dtype)  # [K]
        n = n_coefficients - 1 # degree
        x = x.unsqueeze(-1)  # [B, N, 1]
        y = y.unsqueeze(-1)  # [B, N, 1]

        # Binomial coefficients, log-space for numerical stability
        n_plus_1 = torch.full((n_coefficients,), n + 1.0, dtype=dtype, device=device)
        log_binom = (
            torch.lgamma(n_plus_1)
            - torch.lgamma(k + 1.0)
            - torch.lgamma(n_plus_1 - k)
        ) # [K]
        # Take exp() to leave log-space, and expand for broadcasting
        binom = torch.exp(log_binom).unsqueeze(0).unsqueeze(0)  # [1, 1, K]

        Cx = torch.pow(x, n1) * torch.pow(1.0 - x, n2)    # [B, N, 1]
        Bx = torch.pow(x, k) * torch.pow(1.0 - x, n - k)  # [B, N, K]

        # Weighted Basis Matrix | Design matrix
        Mx = Cx * Bx * binom  # [B, N, K]

        # Solve least squares
        lstsq_result = torch.linalg.lstsq(Mx, y, rcond=rcond)
        coeffs = lstsq_result.solution.squeeze(-1)  # [B, K]

        # if single_curve:
        #     coeffs = coeffs.squeeze(0)  # [K]

        curve = cls(coeffs.squeeze(0), n1=n1, n2=n2, device=device)
        curve._fitted_points = points.detach().clone()  # shape [B, N, 2]
        return curve


class KulfanModifiedCST(TorchCSTCurve):
    """Airfoil specific CST curve with Kulfan leading- and trailing-edge
    modifications.

    Does not use dependency injection to keep the interface simpler.
    Probably should have it take a base CST curve instance instead of creating
    it but I don't wan't to instantiate two objects every time.

    The modified ordinate reads:
        y_mod(x) = y_base(x)
                   + w_le * x * (1 - x)^{n + 0.5}
                   + s_te * t_te * x / 2

    where:
        - y_base(x) is the standard CST curve (super-class implementation).
        - w_le is the leading-edge weight parameter (per curve).
        - t_te is the trailing-edge thickness parameter (per curve).
        - s_te is +1 for the upper surface and -1 for the lower surface.
        - n is the Bernstein polynomial degree (n = n_coefficients - 1).

    The class fully supports batching, automatic differentiation, and analytic
    derivatives by extending the TorchCSTCurve base implementation.

    Attributes:
        leading_edge_weight (torch.Tensor):
            Kulfan leading-edge weight ``w_le`` of shape [B].

        trailing_edge_thickness (torch.Tensor):
            Kulfan trailing-edge thickness ``t_te`` of shape [B].

        surface_type (str):
            Either ``"upper"`` or ``"lower"``, determined from leading-edge
            slope.

        te_sign (float):
            Sign of the TE offset, +1.0 for upper surface, -1.0 for lower
            surface.
    """

    def __init__(
        self,
        coefficients: Union[torch.Tensor, np.ndarray],
        leading_edge_weight: Union[torch.Tensor, float] = 0.0,
        trailing_edge_thickness: Union[torch.Tensor, float] = 0.0,
        surface_type: Optional[str] = None,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
    ) -> None:
        """
        Initialize a Kulfan-modified CST curve.

        Args:
            coefficients (torch.Tensor | np.ndarray):
                CST coefficients ``a_k`` of shape [K] or [B, K].

            leading_edge_weight (torch.Tensor | float):
                Kulfan LE weight ``w_le`` (scalar or [B]).
                Default is 0.0, meaning no LE modification.

            trailing_edge_thickness (torch.Tensor | float):
                Kulfan TE thickness ``t_te`` (scalar or [B]).
                Default is 0.0, meaning no TE modification.
                Absolute value is used to ensure non-negative thickness.

            surface (str):
                Either ``"upper"`` (positive TE thickness) or ``"lower"`` (negative).
                Default is None, and type is inferred from leading-edge slope.

            n1 (float):
                Leading-edge class exponent.
                Default is 0.5 for conventional airfoils.

            n2 (float):
                Trailing-edge class exponent.
                Default is 1.0 for conventional airfoils.

            device (Optional[torch.device | str]):
                Optional device for all tensors.
                Default is CUDA if available.

        Raises:
            ValueError: If surface is not "upper" or "lower", or parameters fail to broadcast.
        """
        super().__init__(coefficients, n1=n1, n2=n2, device=device)

        # Determine surface type
        self.surface_type = surface_type or self.infer_surface_type()

        # little validation
        if self.surface_type not in ("upper", "lower"):
            raise ValueError("surface_type must be 'upper' or 'lower'.")
        if not torch.all(trailing_edge_thickness >= 0):
            warnings.warn(
                "Trailing edge thickness must be non-negative. "
                "Absolute value will be used."
            )

        # Prepare LE/TE parameters so they broadcast across batch size B
        # Register as buffers (non-trainable but part of state_dict)
        self.register_buffer(
            "leading_edge_weight",      # [B]
            self._prepare_modifier_parameter(leading_edge_weight),
        )
        self.register_buffer(
            "trailing_edge_thickness",  # [B]
            self._prepare_modifier_parameter(
                torch.abs(trailing_edge_thickness)  # trailing edge thickness always non-negative
            ),
        )

    def _prepare_modifier_parameter(
        self,
        x: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """
        Broadcast scalar or 1D tensor to match batch size [B, 1].
        Unsqueezed last dimension for broadcasting in computations later.

        Args:
            value: Scalar or tensor to broadcast.
            name: Parameter name used in error messages.

        Returns:
            torch.Tensor: Shape [B, 1].

        Raises:
            ValueError: If broadcasting to [B] is impossible.
        """
        x = torch.as_tensor(
            x,
            dtype=self.coefficients.dtype,
            device=self.device,
        )

        # single value: expand to batch size
        if x.ndim == 0 or (x.ndim == 1 and x.shape[0] == 1):
            x = x.expand(self.batch_size)       # [B]
        # 1D tensor: validate shape
        if x.shape[0] != self.batch_size:
            raise ValueError(
                f"Dimension mismatch: batch size is {self.batch_size}, "
                f"but LE or TE modifier has shape {tuple(x.shape)} instead."
            )
        return x.unsqueeze(-1)

    def infer_surface_type(self) -> str:
        """Determine if the curve represents an upper or lower airfoil surface.

        Based on the sign of the leading-edge slope at x=0:
            - Positive slope indicates upper surface.
            - Negative slope indicates lower surface.

        Slope is the first derivative of the *unmodified* CST curve,
        and signs are cross-checked with the first coefficient and y-component
        of the tangent vector.

        Returns:
            str: "upper" if TE thickness is positive, "lower" if negative.

        Raises:
            ValueError: If mixed surface types are detected in batch.
        """
        if torch.all(torch.sign(self.coefficients[..., 0]) > 0):
            return "upper"
        elif torch.all(torch.sign(self.coefficients[..., 0]) < 0):
            return "lower"
        else:
            raise ValueError(
                "Inconsistent signs of leading-edge slope and first coefficient. "
                "Potentially mixed upper/lower surfaces in batch."
            )

    @property
    def parameters(self) -> torch.Tensor:
        """Get Kulfan modification parameters as a single tensor.

        Returns:
            torch.Tensor: Shape [B, 2] with columns [w_le, t_te].
        """
        return torch.cat(
            [
                self.coefficients,
                self.leading_edge_weight,
                self.trailing_edge_thickness,
            ],
            dim=-1,
        )  # [B>1, n_coefficients + 2]

    @property
    def te_sign(self) -> float:
        """Sign applied to trailing-edge thickness depending on surface.
        Unsqueezed last dimension for broadcasting during computations later.

        Returns:
            torch.Tensor: Shape [B, 1]
                Values +1.0 for upper surface, -1.0 for lower surface.
        """
        return torch.sign(self.coefficients[:, 0]).unsqueeze(-1)  # [B, 1]

    def leading_edge_mod(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Kulfan leading-edge modification term.

        :math:`\delta_{LE}(x) = w_{le} \cdot x \cdot (1 - x)^{n + 0.5}`

        Args:
            x: Chordwise locations [X].

        Returns:
            LE modification [B, X].

        Notes:
            Could exand x to [B, X] with
            .unsqueeze(0).expand(self.batch_size, -1), but works fine as is, as
            long as leading edge weight is unsqueezed to [B, 1].
        """
        x = self._prepare_input(x)                      # [X]
        one_minus_x = (1.0 - x).clamp_min(self.eps)     # [X]
        w_le = self.leading_edge_weight                 # [B, 1]
        return (
            w_le * x * torch.pow(one_minus_x, self.degree + 0.5)
        ).squeeze()                                     # [X] or [B, X]

    def trailing_edge_mod(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Kulfan trailing-edge modification term.

        Args:
            x: Chordwise locations [X].

        Returns:
            torch.Tensor: TE modification, shape [B, X].
        """
        x = self._prepare_input(x)                          # [X]
        t_te = self.trailing_edge_thickness                 # [B, 1]
        return  (self.te_sign * t_te * x / 2.0).squeeze()   # [X] or [B, X]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Kulfan modified curve at chordwise locations ``x``.

        Dimension flow:
            - x: [X]
            - base_y = super().forward(x): [X] (if B=1) or [B, X]
            - base_y reshaped to [B, X]
            - modifications applied element-wise
            - output squeezed to [X] if B=1, else [B, X]

        Args:
            x: Evaluation points in [0, 1].

        Returns:
            torch.Tensor: Modified ordinates with shape [X] or [B, X].
        """
        x = self._prepare_input(x)          # [X]
        return (
            super().forward(x)              # [X] or [B, X]
            + self.leading_edge_mod(x)      # [X] or [B, X]
            + self.trailing_edge_mod(x)     # [X] or [B, X]
        )

    def leading_edge_mod_first_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute first derivative of the Kulfan leading-edge modification term.

        :math:`\delta_{LE}'(x) = w_le * [ (1 - x)^{n+0.5} - (n+0.5) x (1 - x)^{n-0.5} ]`

        Args:
            x: Chordwise locations [X].

        Returns:
            torch.Tensor: \delta_{LE}'(x), shape [B, X].
        """
        x = self._prepare_input(x)                      # [X]
        one_minus_x = (1.0 - x).clamp_min(self.eps)     # [X]
        w_le = self.leading_edge_weight                 # [B, 1]
        exponent = self.degree + 0.5                    # scalar
        return w_le * (
            torch.pow(one_minus_x, exponent)
            - exponent * x * torch.pow(one_minus_x, exponent - 1.0)
        ).squeeze()                                     # [B, X] or [X]

    def leading_edge_mod_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the second derivative of the Kulfan leading-edge modification.

        ..math::
            \delta_{LE}''(x) = w_{le} \left[
            -2 (n + 0.5) (1 - x)^{n-0.5}
            + (n + 0.5)(n - 0.5) x (1 - x)^{n-1.5}
            \right]

        Dimension flow mirrors the first derivative:
            result is [B, X] (squeezed to [X] if B=1).
        """
        x = self._prepare_input(x)                      # [X]
        one_minus_x = (1.0 - x).clamp_min(self.eps)     # [X]
        w_le = self.leading_edge_weight                 # [B, 1]
        exponent = self.degree + 0.5                    # scalar
        return w_le * (
            -2.0 * exponent * torch.pow(one_minus_x, exponent - 1.0)
            + exponent * (exponent - 1.0) * x * torch.pow(one_minus_x, exponent - 2.0)
        ).squeeze()                                     # [B, X] or [X]

    def trailing_edge_mod_first_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the first derivative of the Kulfan trailing-edge modification.
        This is a constant value.

        :math:`\delta_{TE}'(x) = s_{te} \cdot t_{te} / 2`

        Args:
            x (torch.Tensor): Chordwise locations [X].

        Returns:
            torch.Tensor: \delta_{TE}'(x), shape [B, X].

        Dimension flow:
            - constant: [B, 1]
            - ones_like(x): [X]
            - Broadcasting yields [B, X] (squeezed to [X] if B=1).
        """
        x = self._prepare_input(x)              # [X]
        t_te = self.trailing_edge_thickness     # [B, 1]
        return (
            self.te_sign * t_te / 2.0 * torch.ones_like(x)
        ).squeeze()                             # [B, X] or [X]

    def trailing_edge_mod_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the second derivative of the Kulfan trailing-edge modification.

        :math:`\delta_{TE}''(x) = 0`

        Args:
            x (torch.Tensor): Chordwise locations [X].

        Returns:
            torch.Tensor: \delta_{TE}''(x), shape [B, X].

        Returns a zero tensor with shape [B, X] (squeezed to [X] if B=1).
        """
        x = self._prepare_input(x)  # [X]
        zeros = torch.zeros(        # [B, X]
            (self.batch_size, x.shape[0]),
            dtype=self.coefficients.dtype,
            device=self.device,
        )
        return zeros.squeeze()      # [B, X] or [X]

    # Aliases
    dLE_mod = leading_edge_mod_first_derivative
    d2LE_mod = leading_edge_mod_second_derivative
    dTE_mod = trailing_edge_mod_first_derivative
    d2TE_mod = trailing_edge_mod_second_derivative

    def first_derivative_at(
        self,
        x: torch.Tensor,
        mode: str = "analytic",
    ) -> torch.Tensor:
        """
        Compute first derivative of the modified curve.

        ..math::
            dy_mod/dx = dy_base/dx + dLE_mod/dx + dTE_mod/dx
            dLE_mod/dx = w_le * [ (1 - x)^{n+0.5} - (n+0.5) x (1 - x)^{n-0.5} ]
            dTE_mod/dx = s_te * t_te / 2

        Args:
            x: Evaluation points [X].
            mode: "analytic" (default) or "autograd".

        Returns:
            torch.Tensor of shape [X] or [B, X].

        Raises:
            ValueError: When mode is invalid.
        """
        if mode == "analytic":
            x = self._prepare_input(x)  # [X]
            # dy = super().first_derivative_at(x, mode="analytic")
            # return dy + self.dLE_mod(x) + self.dTE_mod(x)
            return (
                super().first_derivative_at(x)
                + self.leading_edge_mod_first_derivative(x)
                + self.trailing_edge_mod_first_derivative(x)
            ).squeeze()

        if mode == "autograd":
            x = (
                self._prepare_input(x)
                .clamp_min(self.eps)
                .detach()
                .clone()
                .requires_grad_(True)
            )

            def grad_fn(x: torch.Tensor) -> torch.Tensor:
                y = self.forward(x)  # [X] or [B, X]
                return torch.autograd.grad(
                    y, x, torch.ones_like(y), create_graph=True
                )[0]

            if self.batch_size == 1:
                return grad_fn(x)

            return vmap(grad_fn)(x.unsqueeze(0).expand(self.batch_size, -1))

        raise ValueError("mode must be 'analytic' or 'autograd'.")

    def second_derivative_at(
        self,
        x: torch.Tensor,
        mode: str = "analytic",
    ) -> torch.Tensor:
        """
        Compute second derivative of the modified curve.

        ..math::
            d2y_mod/dx2 = d2y_base/dx2 + d2LE_mod/dx2 + d2TE_mod/dx2
            d2LE_mod/dx2 = w_le [ -2 (n + 0.5) (1 - x)^{n-0.5}
                                + (n + 0.5)(n - 0.5) x (1 - x)^{n-1.5} ]
            d2TE_mod/dx2 = 0

        Args:
            x: Evaluation points [X].
            mode: "analytic" (default) or "autograd".

        Returns:
            torch.Tensor of shape [X] or [B, X].

        Raises:
            ValueError: When mode is invalid.
        """
        if mode == "analytic":
            x = self._prepare_input(x)  # [X]
            return (
                super().second_derivative_at(x)
                + self.leading_edge_mod_second_derivative(x)
                # + self.trailing_edge_mod_second_derivative(x)  # zero anyway
            ).squeeze()

        if mode == "autograd":
            x = (
                self._prepare_input(x)
                .clamp_min(self.eps)
                .detach()
                .clone()
                .requires_grad_(True)
            )

            def grad_fn(x: torch.Tensor) -> torch.Tensor:
                y = self.forward(x)
                dy = torch.autograd.grad(
                    y, x, torch.ones_like(y), create_graph=True
                )[0]
                ddy = torch.autograd.grad(
                    dy, x, torch.ones_like(dy), create_graph=True
                )[0]
                return ddy

            if self.batch_size == 1:
                return grad_fn(x)

            return vmap(grad_fn)(x.unsqueeze(0).expand(self.batch_size, -1))

        raise ValueError("mode must be 'analytic' or 'autograd'.")

    def curvature_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        """
        Compute curvature of the modified curve:

            κ(x) = y'' / (1 + (y')^{2})^{3/2}

        Args:
            x: Evaluation points [X].
            mode: Derivative strategy for y' and y''.

        Returns:
            torch.Tensor: Curvature with shape [X] or [B, X].
        """
        dy = self.first_derivative_at(x, mode=mode)
        d2y = self.second_derivative_at(x, mode=mode)
        return d2y / torch.pow(1.0 + dy**2, 1.5)

    @classmethod
    def fit(
        cls,
        points: torch.Tensor,
        n_coefficients: int = 8,
        surface_type: Literal["upper", "lower"] = None,
        n1: float = 0.5,
        n2: float = 1.0,
        device: Optional[torch.device | str] = None,
        rcond: Optional[float] = None,
    ) -> "KulfanModifiedCST":
        """
        Jointly fit CST coefficients and Kulfan LE/TE parameters via least squares.

        The linear system solves for ``[a_0, ..., a_{K-1}, w_le, t_te]`` in one step.
        The routine is fully differentiable with respect to the input points.

        Args:
            points: Sampled coordinates, shape [N, 2] or [B, N, 2].
            n_coefficients: Number of CST coefficients (K). Default is 8.
            surface: "upper" or "lower" to control TE sign.
            n1 / n2: Class function exponents for the base CST curve.
            device: Optional torch.device override.
            rcond: Cutoff passed to ``torch.linalg.lstsq``.

        Returns:
            KulfanModifiedCST instance with fitted parameters.

        Raises:
            ValueError: For invalid inputs or insufficient sample points.

        Dimension overview:
            - points: [B, N, 2]
            - x, y: [B, N]
            - design matrix M: [B, N, K+2]
            - solution: [B, K+2] → split into coefficients, w_le, t_te
        """
        device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        points = torch.as_tensor(points.copy(), dtype=torch.float32, device=device)

        # Validate input shape
        if points.ndim not in (2, 3) or points.shape[-1] != 2:
            raise ValueError(
                "points must have shape [N, 2] or [B, N, 2]; "
                f"received {tuple(points.shape)}."
            )

        # Add batch dimension for single curve
        if points.ndim == 2:
            points = points.unsqueeze(0)  # [B=1, N, 2]

        # validate sufficient points
        B, N, _ = points.shape
        if N < n_coefficients:
            raise ValueError(
                f"Need at least {n_coefficients + 2} points to fit "
            f"{n_coefficients} coefficients + 2 modifiers; received N={N}."
            )

        dtype = points.dtype
        x = points[..., 0]  # [B, N]
        y = points[..., 1]  # [B, N]

        # validate x range
        if torch.any((x < 0) | (x > 1)):
            raise ValueError("x coordinates must lie within [0, 1].")

        # clamp x values to avoid numerical issues at endpoints
        eps = torch.finfo(dtype).eps
        x = x.clamp(min=eps, max=1.0 - eps)  # Avoid singularities at endpoints

        # Parameters needed for basis construction
        k = torch.arange(n_coefficients, device=device, dtype=dtype)  # [K]
        n = n_coefficients - 1  # degree, scalar
        x = x.unsqueeze(-1)     # [B, N, 1]
        y = y.unsqueeze(-1)     # [B, N, 1]

        # Binomial coefficients, log-space for numerical stability
        n_plus_1 = torch.full((n_coefficients,), n + 1.0, dtype=dtype, device=device)
        log_binom = (
            torch.lgamma(n_plus_1)
            - torch.lgamma(k + 1.0)
            - torch.lgamma(n_plus_1 - k)
        )  # [K]
        # Take exp() to leave log-space, and expand for broadcasting
        binom = torch.exp(log_binom).unsqueeze(0).unsqueeze(0)  # [1, 1, K]

        Cx = torch.pow(x, n1) * torch.pow(1.0 - x, n2)    # [B, N, 1]
        Bx = torch.pow(x, k) * torch.pow(1.0 - x, n - k)  # [B, N, K]

        # Weighted Basis Matrix of the base CST curve
        Mx = Cx * Bx * binom                              # [B, N, K]

        # Now the tensor of leading edge modifiers
        le_mod = x * torch.pow(1.0 - x, n + 0.5)          # [B, N, 1]

        # # Tensor of trailing edge modifiers
        # if surface_type == "upper":
        #     te_signs = torch.ones(B, 1, dtype=dtype, device=device)     # [B, 1]
        # elif surface_type == "lower":
        #     te_signs = -torch.ones(B, 1, dtype=dtype, device=device)    # [B, 1]
        # else:
        #     te_signs = torch.sign(torch.diff(points[:, :2, 1], dim=1))  # [B, 1]

        # [B, 1, 1] * [B, N, 1]
        # te_mod = te_signs.unsqueeze(1) * x / 2.0                        # [B, N, 1]

        te_mod = x / 2.0  # [B, N, 1]

        # Full Design matrix including LE and TE modifiers
        Mx_mod = torch.cat([Mx, le_mod, te_mod], dim=-1)  # [B, N, K+2]

        solution = torch.linalg.lstsq(Mx_mod, y, rcond=rcond).solution.squeeze(-1)  # [B, K+2]

        # split solution and squeeze away empty batch dim if needed
        coeffs = solution[..., :n_coefficients].squeeze()      # [B, K] or [K]
        le_weights = solution[..., -2].squeeze()               # [B] or [1]
        te_thickness = solution[..., -1].squeeze()             # [B] or [1]

        # # Validate surface type consistency
        # first_coeff_signs = torch.sign(coeffs[..., 0])  # [B]
        # te_signs = torch.sign(te_thickness)             # [B]
        # if not torch.all(first_coeff_signs == te_signs):
        #     raise ValueError(
        #         "Inconsistent surface orientation: first coefficient sign does not "
        #         "match fitted trailing-edge thickness sign.\n"
        #         f"First coeff signs: {first_coeff_signs.squeeze().tolist()}\n"
        #         f"TE thickness signs: {te_signs.squeeze().tolist()}\n"
        #         "This suggests mixed upper/lower surfaces in batch."
        #     )

        # If user provided surface_type, validate it matches the fitted sign
        # if surface_type is not None:
        #     expected_sign = 1.0 if surface_type == "upper" else -1.0
        #     if not torch.all(te_signs == expected_sign):
        #         raise ValueError(
        #             f"Provided surface_type ('{surface_type}') conflicts with fitted solution.\n"
        #             f"Expected sign: {expected_sign}\n"
        #             f"Fitted signs: {te_signs.tolist()}"
        #         )

        curve = cls(
            coefficients=coeffs,
            leading_edge_weight=le_weights,
            trailing_edge_thickness=te_thickness.abs(),
            surface_type=surface_type,
            n1=n1,
            n2=n2,
            device=device,
        )
        curve._fitted_points = points.detach().clone()  # shape [B, N, 2]
        return curve


class TorchPARSECCurve(nn.Module):
    """Differentiable PARSEC surface parameterization (batched).

    This class implements the classic PARSEC *per-surface* half-integer polynomial:

    $$
    y(x) = a_1 x^{1/2} + a_2 x + a_3 x^{3/2} + a_4 x^2 + a_5 x^{5/2} + a_6 x^3,
    \quad x \in [0, 1]
    $$

    Nomenclature (requested):
        - `r_le`: leading-edge radius
        - `x_z`, `y_z`: crest ("z") location and ordinate
        - `k_z`: crest curvature using $k := y''(x_z)$
        - `y_te`: trailing-edge ordinate (a.k.a. `z_te` in some references)
        - `dy_te`: trailing-edge slope $y'(1)$

    Constraint system (per surface):
        1) Leading edge radius sets the first coefficient magnitude:
           $a_1 = s \sqrt{2 r_{le}}$ where $s=+1$ for upper and $s=-1$ for lower.
        2) Crest point constraint: $y(x_z) = y_z$
        3) Crest slope constraint: $y'(x_z) = 0$
        4) Crest curvature constraint: $y''(x_z) = k_z$
        5) Trailing edge ordinate: $y(1) = y_{te}$
        6) Trailing edge slope: $y'(1) = dy_{te}$

    Notes:
        - Derivatives near the leading edge are singular due to $x^{-1/2}$ and
          $x^{-3/2}$. Analytic derivative methods clamp $x$ by `eps` to avoid
          inf/NaN.
        - The coefficient solve uses `torch.linalg.solve` and is differentiable.

    Args:
        r_le: Leading-edge radius (scalar or [B]).
        x_z: Crest x-location (scalar or [B]).
        y_z: Crest ordinate (scalar or [B]).
        k_z: Crest curvature (scalar or [B]).
        dy_te: Trailing-edge slope (scalar or [B]).
        y_te: Trailing-edge ordinate (scalar or [B]). Default is 0.
        surface_type: Either "upper" or "lower".
        device: Optional torch.device override.
        dtype: torch.dtype for all tensors. Default is torch.float32.

    Attributes:
        r_le: Leading-edge radius tensor [B].
        x_z: Crest x-location tensor [B].
        y_z: Crest ordinate tensor [B].
        k_z: Crest curvature tensor [B]. [B].
        y_te: Trailing-edge ordinate tensor [B].
        dy_te: Trailing-edge slope tensor [B].
        surface_type: "upper" or "lower".
        device: Optional torch.device override.
        dtype: torch.dtype for all tensors. Default is torch.float32.
        leading_edge_radius: Alias for `r_le`.
        x_crest: Alias for `x_z`.
        y_crest: Alias for `y_z`.
        yxx_crest: Alias for `k_z`.
        crest_curvature: Alias for `k_z`.
        batch_size: Number of surfaces in batch.
        parameters: PARSEC parameters as a single tensor [B, 6].
    """

    def __init__(
        self,
        r_le: Union[torch.Tensor, np.ndarray, float],
        x_z: Union[torch.Tensor, np.ndarray, float],
        y_z: Union[torch.Tensor, np.ndarray, float],
        k_z: Union[torch.Tensor, np.ndarray, float],
        dy_te: Union[torch.Tensor, np.ndarray, float],
        y_te: Union[torch.Tensor, np.ndarray, float] = 0.0,
        *,
        surface_type: Literal["upper", "lower"],
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        if surface_type not in ("upper", "lower"):
            raise ValueError("surface_type must be either 'upper' or 'lower'.")
        self.surface_type: Literal["upper", "lower"] = surface_type

        # Prepare & broadcast parameters to a common batch size.
        params = [r_le, x_z, y_z, k_z, y_te, dy_te]
        tensors = [self._as_1d_tensor(p) for p in params]
        batch_size = max(t.numel() for t in tensors)
        tensors = [self._broadcast_1d(t, batch_size) for t in tensors]

        r_le_t, x_z_t, y_z_t, k_z_t, y_te_t, dy_te_t = tensors

        # Register parameters as buffers for state_dict inclusion.
        self.register_buffer("r_le", r_le_t)
        self.register_buffer("x_z", x_z_t)
        self.register_buffer("y_z", y_z_t)
        self.register_buffer("k_z", k_z_t)
        self.register_buffer("y_te", y_te_t)
        self.register_buffer("dy_te", dy_te_t)

        self.eps = torch.finfo(dtype).eps

        coeffs = self._compute_coefficients(
            r_le=r_le_t,
            x_z=x_z_t,
            y_z=y_z_t,
            k_z=k_z_t,
            y_te=y_te_t,
            dy_te=dy_te_t,
            surface_type=surface_type,
        )
        self.register_buffer("coefficients", coeffs)  # [B, 6]

    @property
    def leading_edge_radius(self) -> torch.Tensor:
        """Alias for `r_le` (compatibility with other code paths)."""
        return self.r_le

    @property
    def x_crest(self) -> torch.Tensor:
        """Alias for `x_z` (compatibility with earlier drafts)."""
        return self.x_z

    @property
    def y_crest(self) -> torch.Tensor:
        """Alias for `y_z` (compatibility with earlier drafts)."""
        return self.y_z

    @property
    def yxx_crest(self) -> torch.Tensor:
        """Alias for `k_z` (compatibility with earlier drafts)."""
        return self.k_z

    @property
    def crest_curvature(self) -> torch.Tensor:
        """Alias for `k_z` (compatibility with earlier drafts)."""
        return self.k_z

    @property
    def batch_size(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def parameters(self) -> torch.Tensor:
        """Return PARSEC surface parameters as a single tensor [B, 6].

        Order:
            [ r_le, x_z, y_z, k_z, y_te, dy_te ]
        """
        return torch.stack(
            [
                self.r_le,
                self.x_z,
                self.y_z,
                self.k_z,
                self.y_te,
                self.dy_te,
            ],
            dim=-1,
        )

    def _as_1d_tensor(
        self, x: Union[torch.Tensor, np.ndarray, float]
    ) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if t.ndim == 0:
            return t.unsqueeze(0)
        if t.ndim != 1:
            raise ValueError("PARSEC parameters must be scalars or 1D tensors.")
        return t

    def _broadcast_1d(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.numel() == 1:
            return t.expand(batch_size)
        if t.numel() != batch_size:
            raise ValueError(
                f"Batch mismatch: expected scalar or length {batch_size}, got {t.numel()}."
            )
        return t

    def _prepare_input(self, x: Union[torch.Tensor, np.ndarray, float]) -> torch.Tensor:
        x = (
            x.to(dtype=self.dtype, device=self.device)
            if torch.is_tensor(x)
            else torch.as_tensor(x, dtype=self.dtype, device=self.device)
        )
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.ndim != 1:
            raise ValueError("Input x must be 1D or scalar")
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("x must be in the range [0, 1]")
        return x

    @staticmethod
    def _basis(x: torch.Tensor) -> torch.Tensor:
        """Return PARSEC basis matrix [X, 6]."""
        sqrtx = torch.sqrt(x)
        x1 = x
        x3_2 = x * sqrtx
        x2 = x * x
        x5_2 = x2 * sqrtx
        x3 = x2 * x
        return torch.stack([sqrtx, x1, x3_2, x2, x5_2, x3], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        basis = self._basis(x)  # [X, 6]
        y = self.coefficients @ basis.T  # [B, X]
        return y.squeeze(0)

    def evaluate_at(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        y = self.forward(x)
        if y.ndim == 1:
            return torch.stack([x, y], dim=-1)
        x_b = x.unsqueeze(0).expand(self.batch_size, -1)
        return torch.stack([x_b, y], dim=-1)

    def first_derivative_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        if mode == "autograd":
            x = (
                self._prepare_input(x)
                .clamp_min(self.eps)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            y = self.forward(x)
            dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
            return dy

        if mode != "analytic":
            raise ValueError("mode must be 'analytic' or 'autograd'.")

        x = self._prepare_input(x)
        x_safe = x.clamp_min(self.eps)
        sqrtx = torch.sqrt(x_safe)
        inv_sqrtx = 1.0 / sqrtx
        x1 = x_safe
        x3_2 = x_safe * sqrtx
        x2 = x_safe * x_safe

        a1, a2, a3, a4, a5, a6 = self.coefficients.unbind(dim=-1)  # [B]
        dy = (
            0.5 * a1.unsqueeze(-1) * inv_sqrtx
            + a2.unsqueeze(-1)
            + 1.5 * a3.unsqueeze(-1) * sqrtx
            + 2.0 * a4.unsqueeze(-1) * x1
            + 2.5 * a5.unsqueeze(-1) * x3_2
            + 3.0 * a6.unsqueeze(-1) * x2
        )
        return dy.squeeze(0)

    def second_derivative_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        if mode == "autograd":
            x = (
                self._prepare_input(x)
                .clamp_min(self.eps)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            y = self.forward(x)
            dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
            d2y = torch.autograd.grad(dy, x, torch.ones_like(dy), create_graph=True)[0]
            return d2y

        if mode != "analytic":
            raise ValueError("mode must be 'analytic' or 'autograd'.")

        x = self._prepare_input(x)
        x_safe = x.clamp_min(self.eps)
        sqrtx = torch.sqrt(x_safe)
        inv_sqrtx = 1.0 / sqrtx
        inv_x3_2 = inv_sqrtx / x_safe

        a1, _, a3, a4, a5, a6 = self.coefficients.unbind(dim=-1)  # [B]
        d2y = (
            -0.25 * a1.unsqueeze(-1) * inv_x3_2
            + 0.75 * a3.unsqueeze(-1) * inv_sqrtx
            + 2.0 * a4.unsqueeze(-1)
            + 3.75 * a5.unsqueeze(-1) * sqrtx
            + 6.0 * a6.unsqueeze(-1) * x_safe
        )
        return d2y.squeeze(0)

    def curvature_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        dy = self.first_derivative_at(x, mode=mode)
        d2y = self.second_derivative_at(x, mode=mode)
        return d2y / torch.pow(1.0 + dy**2, 1.5)

    def radius_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        return 1.0 / (torch.abs(self.curvature_at(x, mode=mode)) + self.eps)

    def tangent_at(self, x: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        dy = self.first_derivative_at(x, mode=mode)
        return torch.atan(dy)

    def normal_at(
        self,
        x: torch.Tensor,
        mode: str = "analytic",
        direction: Literal["outward", "inward"] = "outward",
    ) -> torch.Tensor:
        theta = self.tangent_at(x, mode=mode)
        if direction == "outward":
            return theta + torch.pi / 2
        if direction == "inward":
            return theta - torch.pi / 2
        raise ValueError("direction must be 'outward' or 'inward'.")

    def _compute_coefficients(
        self,
        *,
        r_le: torch.Tensor,
        x_z: torch.Tensor,
        y_z: torch.Tensor,
        k_z: torch.Tensor,
        y_te: torch.Tensor,
        dy_te: torch.Tensor,
        surface_type: Literal["upper", "lower"],
    ) -> torch.Tensor:
        """Solve for the PARSEC polynomial coefficients per batch element.

        Unknowns: $[a_2, a_3, a_4, a_5, a_6]$.
        The first coefficient $a_1$ is fixed by the leading-edge radius.
        """
        r_le = torch.abs(r_le).clamp_min(self.eps)  # [B]
        x_z = x_z.clamp(min=self.eps, max=1.0 - self.eps)  # [B]

        sign = 1.0 if surface_type == "upper" else -1.0
        a1 = sign * torch.sqrt(2.0 * r_le)  # [B]

        sqrt_xc = torch.sqrt(x_z)
        xc = x_z
        xc_3_2 = xc * sqrt_xc
        xc2 = xc * xc
        xc_5_2 = xc2 * sqrt_xc
        xc3 = xc2 * xc
        inv_sqrt_xc = 1.0 / sqrt_xc
        inv_xc_3_2 = inv_sqrt_xc / xc

        B = x_z.shape[0]
        ones = torch.ones((B,), dtype=self.dtype, device=self.device)
        zeros = torch.zeros((B,), dtype=self.dtype, device=self.device)

        row1 = torch.stack([ones, ones, ones, ones, ones], dim=-1)
        row2 = torch.stack(
            [
                ones,
                1.5 * ones,
                2.0 * ones,
                2.5 * ones,
                3.0 * ones,
            ],
            dim=-1,
        )
        row3 = torch.stack([xc, xc_3_2, xc2, xc_5_2, xc3], dim=-1)
        row4 = torch.stack(
            [ones, 1.5 * sqrt_xc, 2.0 * xc, 2.5 * xc_3_2, 3.0 * xc2],
            dim=-1,
        )
        row5 = torch.stack(
            [zeros, 0.75 * inv_sqrt_xc, 2.0 * ones, 3.75 * sqrt_xc, 6.0 * xc],
            dim=-1,
        )
        A = torch.stack([row1, row2, row3, row4, row5], dim=1)  # [B, 5, 5]

        b1 = y_te - a1
        b2 = dy_te - 0.5 * a1
        b3 = y_z - a1 * sqrt_xc
        b4 = -0.5 * a1 * inv_sqrt_xc
        b5 = k_z + 0.25 * a1 * inv_xc_3_2
        b = torch.stack([b1, b2, b3, b4, b5], dim=-1)  # [B, 5]

        sol = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)  # [B, 5]
        a2, a3, a4, a5, a6 = sol.unbind(dim=-1)
        return torch.stack([a1, a2, a3, a4, a5, a6], dim=-1)


if __name__ == "__main__":

    batchcoeffs = torch.randn(32, 4)  # [B=32, K=8]
    batchcurve = TorchCSTCurve.from_numpy(batchcoeffs, device="cpu")

    # coeffs = np.array([0.2, 0.4, 0.3, 0.1])
    coeffs = batchcoeffs[0].numpy()
    curve = TorchCSTCurve.from_numpy(coeffs, device="cpu")
    curve2 = TorchCSTCurve.from_numpy(-coeffs, device="cpu")

    fig,ax = curve.plot()#[0].savefig("test")
    curve2.plot(fig=fig, ax=ax)[0].savefig("test2", )

    bkcurve = KulfanModifiedCST(
        coefficients=torch.rand(32, 4),
        leading_edge_weight=torch.randn(32),
        trailing_edge_thickness=torch.rand(32)/5,
        device="cpu",
    )
    kcurve = KulfanModifiedCST(
        coefficients=coeffs,
        leading_edge_weight=torch.randn(1),
        trailing_edge_thickness=torch.rand(1)/5,
        device="cpu",
    )

    x = torch.linspace(0, 1, 100)
    y = curve(x)
    dy_dx = curve.first_derivative(x)
    d2y_dx2 = curve.second_derivative(x)
    curvature = curve.curvature(x)

    print("x:", x)
    print("y:", y)
    print("dy/dx:", dy_dx)
    print("d2y/dx2:", d2y_dx2)
    print("curvature:", curvature)
