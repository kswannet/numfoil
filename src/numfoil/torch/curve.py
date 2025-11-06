import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Optional


class TorchCSTCurve(nn.Module):
    """
    PyTorch implementation of a CST curve for differentiable airfoil design.

    The standard CST formulation: y(x) = C(x) * S(x)
    where:
        C(x) = x^n1 (1 - x)^n2  (class function)
        S(x) = ∑_{k=0}^n a_k B_{n,k}(x)  (shape function)

    in this context:
        - k is the index of the Bernstein polynomial
        - n is the degree of the polynomial (n = number of coefficients - 1)
        - n1, n2 are the class function exponents
        - a_k are the shape coefficients (learnable parameters)
        - B_{n,k}(x) are the Bernstein basis polynomials of degree n
        - x is the normalized chordwise position in [0, 1]


    Attributes:
        coefficients (torch.Tensor): Learnable coefficients a_k for the shape function
        n1 (float): Exponent for the class function C(x)
        n2 (float): Exponent for the class function C(x)
        device (torch.device): Device to store tensors on (CPU or GPU)
        k_vec (torch.Tensor): Vector of k values for Bernstein polynomials
        binomial_weights (torch.Tensor): Precomputed binomial coefficients for
            Bernstein polynomials

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
        device: Optional[torch.device] = None,
    ):
        """Initialize the TorchCSTCurve
        Args:
            coefficients (Union[torch.Tensor, np.ndarray]):
                Coefficients a_k for the shape function
            n1 (float): Exponent for the class function C(x)
            n2 (float): Exponent for the class function C(x)
        """
        super().__init__()
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        coefficients = torch.as_tensor(coefficients, dtype=torch.float32)
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be a 1D tensor or array.")

        # Register coefficients as a parameter so they can be optimized
        self.register_parameter("coefficients", nn.Parameter(coefficients.float()))

        # Class function exponents
        self.n1 = n1
        self.n2 = n2

        # Degree of the Bernstein polynomial
        self.register_buffer(
            "k_vec", torch.arange(self.n_coefficients, dtype=torch.float32)
        )

        # Precompute binomial coefficients
        self.register_buffer(
            "binomial_weights", self._compute_binomial_weights()
        )

    # alias
    @property
    def k(self) -> torch.Tensor:
        """Alias for k vector for Bernstein polynomials"""
        return self.k_vec

    @property
    def n(self) -> int:
        """Alias for degree of the Bernstein polynomial"""
        return self.degree

    @property
    def n_coefficients(self) -> int:
        """Number of coefficients in the shape function"""
        return self.coefficients.shape[0]

    @property
    def degree(self) -> int:
        """Degree of the Bernstein polynomial"""
        return self.n_coefficients - 1

    @staticmethod
    def _ensure_1d(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 0 else x

    def _compute_binomial_weights(self) -> torch.Tensor:
        """Compute binomial coefficients for Bernstein polynomials

        :math:`\binom{n}{k} = C(n,k) = n! / (k! (n-k)!)`

        where
            - C(n,k) are the coefficients
            - n is the degree of the polynomial (n = number of coefficients - 1)
            - k is the index of the Bernstein polynomial

        returns:
            torch.Tensor: Binomial coefficients of shape (n+1,)
        """
        # n = self.degree
        # k = self.k_vec

        # Use log-space to avoid numerical issues with large factorials
        # log_binom = (
        #     torch.lgamma(torch.tensor(n + 1.0))
        #     - torch.lgamma(k + 1.0)
        #     - torch.lgamma(torch.tensor(n + 1.0) - k)
        # )

        # Use log-space to avoid numerical issues with large factorials
        n_plus_1 = torch.full((self.n_coefficients,), self.n + 1.0, device=self.device)
        log_binom = (
            torch.lgamma(n_plus_1)
            - torch.lgamma(self.k + 1.0)
            - torch.lgamma(n_plus_1 - self.k)
        )
        return torch.exp(log_binom)

    def class_function(self, x: torch.Tensor) -> torch.Tensor:
        """Class function for the CST curve

        :math:`C(x) = x^{n_1} (1 - x)^{n_2}`

        args:
            x (torch.Tensor): Input tensor of shape (...,) with values in [0, 1]

        returns:
            torch.Tensor: Output tensor of shape (...,) representing C(x)
        """
        x = self._ensure_1d(x)
        x = x.to(self.coefficients.dtype)
        return torch.pow(x, self.n1) * torch.pow(1.0 - x, self.n2)
        # return torch.pow(x.clamp_min(0.0), self.n1) * torch.pow(
        #     (1.0 - x).clamp_min(0.0), self.n2
        # )

    def bernstein_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the Bernstein polynomial basis functions

        :math:`B_{n,k}(x) = x^k (1-x)^{n-k}`

        where
            - n is the degree of the polynomial (n = number of coefficients - 1)
            - k is the index of the Bernstein polynomial

        args:
            x (torch.Tensor): Input tensor of shape (...,) with values in [0, 1]

        returns:
            torch.Tensor: Output tensor of shape (..., n+1) representing the
                basis functions. The last dimension corresponds to k=0,...,n.
                So column k contains B_{n,k}(x).
        """
        # Reshape x for broadcasting
        x = self._ensure_1d(x).unsqueeze(-1)
        return torch.pow(x, self.k) * torch.pow(1.0 - x, self.n - self.k)

    def weighted_basis_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the weighted Bernstein polynomial basis matrix
        (I forgot why I added this... )

        :math:`W_{n,k}(x) = \binom{n}{k} B_{n,k}(x)`
        :math:`M(x) = C(x) * B(x) * binom_coeffs`

        where
            - n is the degree of the polynomial (n = number of coefficients - 1)
            - k is the index of the Bernstein polynomial

        args:
            x (torch.Tensor): Input tensor of shape (...,) with values in [0, 1]

        returns:
            torch.Tensor: Output tensor of shape (..., n+1) representing the
                weighted basis functions. The last dimension corresponds to k=0,...,n.
                So column k contains W_{n,k}(x).
        """
        x = self._ensure_1d(x).unsqueeze(-1)
        return self.class_function(x) * self.binomial_weights * self.bernstein_basis(x)

    def shape_function(self, x: torch.Tensor) -> torch.Tensor:
        """Shape function S(x) = ∑_{k=0}^n a_k B_{n,k}(x)"""
        return torch.sum(
                self.coefficients
                * self.binomial_weights
                * self.bernstein_basis(x),
            dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the CST curve at points x"""
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("x must be in the range [0, 1]")

        return self.class_function(x) * self.shape_function(x)

    def evaluate_points(self, x: torch.Tensor) -> torch.Tensor:
        """Return (x,y) coordinates of the curve"""
        y = self.forward(x)
        if x.dim() == 1:
            return torch.stack([x, y], dim=1)
        else:
            return torch.stack([x, y], dim=2)

    def first_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute first derivative using autograd"""
        x_requires_grad = x.detach().requires_grad_(True)
        y = self.forward(x_requires_grad)

        return torch.autograd.grad(
            outputs=y,
            inputs=x_requires_grad,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
        )[0]

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute second derivative using autograd"""
        x_requires_grad = x.detach().requires_grad_(True)
        dy_dx = self.first_derivative(x_requires_grad)

        return torch.autograd.grad(
            outputs=dy_dx,
            inputs=x_requires_grad,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True,
        )[0]

    def curvature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute curvature using first and second derivatives"""
        dy_dx = self.first_derivative(x)
        d2y_dx2 = self.second_derivative(x)

        return d2y_dx2 / torch.pow(1.0 + dy_dx**2, 1.5)

    @classmethod
    def from_numpy(cls, coefficients: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> 'TorchCSTCurve':
        """Create a TorchCSTCurve from numpy coefficients"""
        return cls(torch.tensor(coefficients, dtype=torch.float32), n1, n2)
