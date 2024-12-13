import numpy as np
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod
from functools import cached_property
import scipy.optimize as opt
import scipy.interpolate as si

from .geom2d import normalize_2d, rotate_2d_90ccw
from ..util import cosine_spacing, chebyshev_nodes, ensure_1d_vector


class Curve(ABC):
    """Abstract base class for curve definitions.
    Evaluations are either done at a location defined by a parameter ``u``
    indicating a parametric curve, or a location ``x`` indicating the location
    along the x-axis (chord line in case of airfoils).
    """

    def evaluate_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve at locations ``u``."""
        raise NotImplementedError

    def tangent_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Compute tangent vectors at locations ``u``."""
        return normalize_2d(self.first_deriv_at(u))

    def normal_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Compute normal vectors at locations ``u``."""
        return rotate_2d_90ccw(self.tangent_at(u))

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve's first derivative(s) at ``u``."""
        raise NotImplementedError

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve's second derivative(s) at ``u``."""
        raise NotImplementedError

    def curvature_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Calculate the spline curvature at ``u``
        k=|y"(x)| / (1+(y'(x))^2)^{3/2}
        """
        dx, dy = self.first_deriv_at(u).T
        ddx, ddy = self.second_deriv_at(u).T
        numerator = ddy * dx - ddx * dy  # det([dx, dy], [ddx, ddy])
        denominator = (dx**2 + dy**2) ** 1.5  # abs([dx, dy])^3
        return np.where(denominator != 0, numerator / denominator, 0)

    def radius_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Return the radius of curvature on the curve at given location ``u``."""
        curvature = self.curvature_at(u)
        return np.where(curvature != 0, 1 / curvature, np.inf)


class BSpline2D(Curve):
    """Creates a splined representation of a set of points.

    Args:
        points: A set of 2D row-vectors
        degree: Degree of the spline. Defaults to 3 (cubic spline).
    """

    def __init__(
        self,
        points: np.ndarray,
        degree: Optional[int] = 3,
        smoothing: Optional[float] = 0.0,
    ):
        # self.points = points
        super().__init__(points)
        self.degree = degree
        self.smoothing = smoothing

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            tck: Tuple of knots, the B-spline coefficients,
            degree of the spline.
        """
        return si.splprep(self.points.T, s=self.smoothing, k=self.degree)[0]

    def evaluate_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline point(s) at ``u``."""
        return np.array(si.splev(u, self.spline, der=0), dtype=np.float64).T

    def find_u(self, x: float = None, y: float = None) -> float:
        """Find the parametric value ``u`` for a given point ``(x, y)``."""

        def objective(u):
            return np.linalg.norm(self.evaluate_at(u) - np.array([x, y]))

        result = opt.minimize(
            objective, 0.5, bounds=[(0.0, 1)], method="SLSQP"
        )
        if not result.success:
            print("Failed to find u!")
        return

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline's first derivative(s) at ``u``."""
        return np.array(si.splev(u, self.spline, der=1), dtype=np.float64).T

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline's second derivative(s) at ``u``."""
        return np.array(si.splev(u, self.spline, der=2), dtype=np.float64).T

    @property
    def max_curvature(self) -> Tuple[float, float]:
        """Finds maximum curvature of the spline
        Returns:
            Tuple[float, np.float]: [u, curvature]: max curvature location on
        on the spline (u) and the maximum curvature value
        """
        result = opt.minimize(
            lambda u: -self.curvature_at(u[0]) ** 2,
            0.51,
            bounds=[(0.0, 1)],
            method="SLSQP",
        )
        if not result.success:
            print("Failed to find max curvature!")
        return result.x[0], (
            np.sqrt(-result.fun) if result.success else float("nan")
        )


class Bezier(Curve):
    """Represents a Bezier curve using control points."""

    def __init__(
        self,
        control_points: np.ndarray,
        points: np.ndarray = None,
    ):
        """
        Initialize a BezierCurve instance.

        Args:
            points (np.ndarray): Input coordinates for fitting the curve.
            n_control_points (int): Number of control points to fit.
            spacing (str): Spacing method for control points ("linear" or "cosine").
        """
        if control_points.ndim != 2 or control_points.shape[1] != 2:
            raise ValueError(
                "control_points must be a 2D array with shape (n, 2)."
            )

        self.control_points = control_points
        self.n_control_points = len(control_points)
        self.points = points

    @cached_property
    def degree(self) -> int:
        """Return the degree of the Bézier curve.
        degree k = n_control_points - 1"""
        return self.n_control_points - 1

    @cached_property
    def coefficient_matrix(self) -> np.ndarray:
        """Compute the characteristic matrix of coefficients for a Bézier curve.

        This method calculates the matrix of coefficients used in the construction
        of a Bézier curve of a given degree. The matrix is computed using binomial
        coefficients and combinatorial logic.

        Returns:
            np.ndarray: A 2D numpy array representing the characteristic matrix of
            coefficients for the Bézier curve.

        Raises:
            ValueError: If the degree of the Bézier curve or the number of control
            points is not properly defined.
        """
        n = self.degree
        matrix = np.zeros((self.n_control_points, self.n_control_points))
        for i in range(self.n_control_points):
            for j in range(i + 1):
                matrix[i, j] = (-1) ** (i - j) * np.math.comb(n, i) * np.math.comb(i, j)
        return matrix

    # @cached_property
    # def first_derivative_coefficient_matrix(self) -> np.ndarray:
    #     """Compute the matrix of coefficients for the derivative Bernstein polynomials."""
    #     n = self.degree
    #     matrix = np.zeros((self.n_control_points - 1, self.n_control_points))
    #     for i in range(self.n_control_points - 1):
    #         matrix[i, i] = -n
    #         matrix[i, i + 1] = n
    #     return matrix

    # @cached_property
    # def second_derivative_coefficient_matrix(self) -> np.ndarray:
    #     """Compute the matrix of coefficients for the second derivative Bernstein polynomials."""
    #     n = self.degree
    #     second_derivative_matrix = np.zeros(
    #         (self.n_control_points - 2, self.n_control_points)
    #     )
    #     for i in range(self.n_control_points - 2):
    #         second_derivative_matrix[i, i] = n * (n - 1)
    #         second_derivative_matrix[i, i + 1] = -2 * n * (n - 1)
    #         second_derivative_matrix[i, i + 2] = n * (n - 1)
    #     return second_derivative_matrix

    def evaluate_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the Bézier curve at parametric location(s) ``u``.

        Args:
            u (Union[float, np.ndarray]): Parametric location(s) to evaluate. Must be in the range [0, 1].

        Returns:
            np.ndarray: Evaluated curve point(s) as a 2D array. For scalar input, returns shape (2,).
        """
        u = ensure_1d_vector(u)
        # Basis matrix [1, t, t^2, ..., t^n]
        # basis = u[:, None] ** np.arange(self.n_control_points)
        basis = np.power.outer(u, np.arange(self.n_control_points))
        return basis @ self.coefficient_matrix @ self.control_points

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve's first derivative(s) at ``u``."""
        u = ensure_1d_vector(u)
        # Basis matrix for (n-1)-degree Bernstein polynomials
        # basis = np.array(
        #     [
        #         i * u ** (i - 1) if i > 0 else np.zeros_like(u)
        #         for i in range(self.n_control_points)
        #     ]
        # ).T
        powers = np.arange(self.n_control_points)
        basis = powers * u[:, None] ** (powers - 1)
        return (basis @ self.coefficient_matrix @ self.control_points )

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve's second derivative(s) at ``u``."""
        u = ensure_1d_vector(u)
        # Basis matrix for (n-2)-degree Bernstein polynomials
        # basis = np.array(
        #     [
        #         i * (i - 1) * u ** (i - 2) if i > 1 else np.zeros_like(u)
        #         for i in range(self.n_control_points)
        #     ]
        # ).T
        powers = np.arange(self.n_control_points)
        basis = powers * (powers - 1) * u[:, None] ** (powers - 2)
        return (basis @ self.coefficient_matrix @ self.control_points )

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        n_control_points: int,
        clamped: bool = True,
        method: str = "least_squares",
    ) -> "Bezier":
        """Fit a Bézier curve to a given set of points.

        Args:
            points (np.ndarray): Input points of shape (m, 2).
            n_control_points (int): Number of control points to use.

        Returns:
            Bezier: Fitted :py:class:`Bézier` curve.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a 2D array with shape (m, 2).")
        if len(points) < n_control_points:
            raise ValueError(
                "Number of points must be greater than or equal to n_control_points."
            )

        if clamped:
            n_control_points -= 2 # endpoints are fixed to data

        # initial guesses are taken from data points
        init_guess = points[
            np.linspace(0, len(points) - 1, n_control_points, dtype=int)
        ]

        # Parametric locations of data on curve approximated by arc length
        u_target = cls.arc_lengths(points, normalize=True)

        def objective(flat_control_points):
            control_points = flat_control_points.reshape(-1, 2)  # unflatten
            if clamped:
                control_points = np.vstack(
                    (points[0], control_points, points[-1])
                )
            curve_points = cls(control_points).evaluate_at(u_target)
            residuals = points - curve_points
            match method:
                case "least_squares":
                    return residuals.ravel()  # flatten again
                case "L2_norm":
                    return (
                        np.linalg.norm(residuals, axis=1).sum()
                        # +
                        # 1e-2 * np.linalg.norm(control_points, axis=1).sum()
                    )
                case _:
                    raise ValueError(
                        "Invalid method. Use 'least_squares' or 'L2_norm'."
                    )

        match method:
            case "least_squares":
                results = opt.least_squares(
                    objective, init_guess.ravel()
                    )
            case "L2_norm":
                results = opt.minimize(
                    objective, init_guess.ravel()
                    )
            case _:
                raise ValueError(
                    "Invalid method. Use 'least_squares' or 'L2_norm'."
                )
        control_points = results.x.reshape(-1, 2)
        if clamped:
            control_points = np.vstack((points[0], control_points, points[-1]))
        return cls(control_points, points)

    @staticmethod
    def arc_lengths(points: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Approximate parameter values `u` for input points based on cumulative
        arc length.

        Args:
            points (np.ndarray): Input points of shape (m, 2).

        Returns:
            np.ndarray: Approximate `t` values in [0, 1] for each input point.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a 2D array with shape (m, 2).")

        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_lengths = np.insert(np.cumsum(distances), 0, 0)
        return (
            cumulative_lengths / cumulative_lengths[-1]
            if normalize
            else cumulative_lengths
        )


class Bezier2D(Bezier):
    """Represents a Bezier curve using control points.
    Tailored for 2D airfoil coordinates specifcally."""

    def __init__(
        self,
        points: np.ndarray,
        n_control_points: int,
        spacing: Union[str, np.ndarray] = "cosine",
    ):
        """
        Initialize a BezierCurve instance.

        Args:
            points (np.ndarray): Input coordinates for fitting the curve.
            n_control_points (int): Number of control points to fit.
            spacing (str): Spacing method for control points ("linear" or "cosine").
        """
        # super().__init__(points, n_control_points, spacing)

    # def _fit_control_points(self) -> np.ndarray:
    #     """Fit control points to best approximate the input data."""
    #     # initial guess is approximate midpoints for upper and lower surfaces
    #     y_init = np.hstack((
    #         self.points[len(self.points) * 1 // 4, 1],  # ~25% index
    #         self.points[len(self.points) * 3 // 4, 1]   # ~75% index
    #     ))

    # def combine_x_y(y_values):
    #     y_upper, y_lower = np.split(y_values, 2)
    #     control_points = np.vstack((
    #         np.column_stack((
    #             self.x_control_points[::-1],
    #             y_upper
    #         )),
    #         np.array([0,0]),
    #         np.column_stack((
    #             self.x_control_points,
    #             y_lower
    #         ))
    #     ))
    #     return control_points

    # def objective(y_values):
    #     control_points = combine_x_y(y_values)
    #     curve_points = self.evaluate_at(self.u_values)
    #     distances = np.linalg.norm(self.points - curve_points, axis=1)
    #     return distances

    # # Fit y-values of control points via least squares
    # initial_guess = control_points[:, 1]
    # result = least_squares(objective, y_init)
    # control_points[:, 1] = result.x
    # return control_points

    class BBezier(Curve):
        """
        create a tck tuple to represent a B-spline curve which can be evaluated
        using splev, but behaves like a Bezier curve. This is done by fixing the
        knot vector and number of control points.

        k = degree = n_control_points - 1
        t = knot vector = [0, 0, 0, 1, 1, 1]
        """

        pass

    class BBezier(Curve):
        """
        create a tck tuple to represent a B-spline curve which can be evaluated
        using splev, but behaves like a Bezier curve. This is done by fixing the
        knot vector and number of control points.

        k = degree = n_control_points - 1
        t = knot vector = [0, 0, 0, 1, 1, 1]
        """

        pass
