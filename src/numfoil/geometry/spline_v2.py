import numpy as np
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod
from functools import cached_property
import scipy.optimize as opt
import scipy.interpolate as si
from scipy.spatial import KDTree


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

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve's first derivative(s) at ``u``."""
        raise NotImplementedError

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the curve's second derivative(s) at ``u``."""
        raise NotImplementedError

    def tangent_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Compute tangent vectors at locations ``u``."""
        return normalize_2d(self.first_deriv_at(u))

    def normal_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Compute normal vectors at locations ``u``."""
        return rotate_2d_90ccw(self.tangent_at(u))

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

    def find_u(self, points: np.ndarray, method: str = "least_squares") -> float:
        """Find the parametric value ``u`` for a given point ``(x, y)``."""
        # def objective(u):
        #     """distance(s) between points"""
        #     return sum(np.linalg.norm(self.evaluate_at(u) - points, axis=1)**2)
        # result = opt.minimize(
        #     objective,
        #     [0.5]*len(points),
        #     bounds=[(0., 1)]*len(points),
        #     method="SLSQP"
        #     )
        # if not result.success:
        #     print("Failed to find u!")
        # return result.x

        init_guess = np.array([0.5]*len(points))

        def objective(u):
            residuals = points - self.evaluate_at(u)

            match method:
                case "least_squares":
                    return residuals.ravel()

                case "L2_norm":
                    return np.linalg.norm(residuals, axis=1).sum()

                case _:
                    raise ValueError(
                        "Invalid method. Use 'least_squares' or 'L2_norm'."
                    )

        match method:
            case "least_squares":
                result = opt.least_squares(
                    objective, init_guess
                    )
            case "L2_norm":
                result = opt.minimize(
                    objective,
                    init_guess,
                    )
            case _:
                raise ValueError(
                    "Invalid method. Use 'least_squares' or 'L2_norm'."
                )

        if not result.success:
            # print("Failed to find u!")
            raise ValueError(
                "failed to find u, optimization success {result.success}"
                )
        return result.x


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
        self.points = points
        # super().__init__(points)
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
        endpoint_clamping: bool = True,
        method: str = "least_squares",
        beta: float = 0,
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

        if endpoint_clamping:
            n_control_points -= 2 # endpoints are fixed to data

        # initial guesses are taken from data points
        init_guess = points[
            np.linspace(0, len(points) - 1, n_control_points, dtype=int)
        ]

        # Parametric locations of data on curve approximated by arc length
        u_target = cls.arc_lengths(points, normalize=True)

        def objective(flat_control_points):
            control_points = flat_control_points.reshape(-1, 2)  # unflatten
            if endpoint_clamping:
                control_points = np.vstack(
                    (points[0], control_points, points[-1])
                )
            curve_points = cls(control_points).evaluate_at(u_target)
            residuals = points - curve_points

            # control point location regularization term
            smoothness_penalty = beta * np.sum(
                np.linalg.norm(
                    # np.diff(control_points, axis=0),
                    control_points,
                    axis=1,
                    ) ** 2
            )

            match method:
                case "least_squares":
                    return (
                        residuals.ravel()       # flatten again
                        + smoothness_penalty    # regularize
                    )
                case "L2_norm":
                    return (
                        np.linalg.norm(residuals, axis=1).sum()
                        + smoothness_penalty
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
        if endpoint_clamping:
            control_points = np.vstack((points[0], control_points, points[-1]))
        return cls(control_points, points)


class AirfoilBezier(Bezier):
    """Bezier curve subclass tailored for airfoil geometry."""

    def __init__(
        self,
        control_points: np.ndarray,
        points: np.ndarray = None,
    ):
        """
        Initialize an AirfoilBezier instance.

        Args:
            control_points (np.ndarray): Control points defining the airfoil curve.
            points (np.ndarray, optional): Input coordinates for fitting the curve.
        """
        super().__init__(control_points, points)

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        n_control_points: int = 6,
        spacing: Union[str, np.ndarray] = "cosine",
        method: str = "least_squares",
        endpoint_clamping = False,
    ) -> "AirfoilBezier":
        """
        Fit an AirfoilBezier curve to the given airfoil points.

        Args:
            points (np.ndarray): Input airfoil points of shape (m, 2).
                Must follow the order: trailing edge -> upper surface ->
                leading edge -> lower surface -> trailing edge.
            n_control_points (int): Number of control points per airfoil
                surface. Total number of control points will be 2*n+1, n for the
                upper surface, n for the lower surface, and one explicitly in
                the origin (leading edge).
            spacing (np.ndarray or str): Predefined x-locations for control
                points in array (from [0,1], without repeated points), or a
                string defining the type of spacing used.
            method (str): Fitting method, either "least_squares" or "L2_norm".

        Returns:
            AirfoilBezier: Fitted airfoil-specific Bézier curve.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a 2D array with shape (m, 2).")

        if len(points) < n_control_points * 2 + 1:
            raise ValueError(
                "Number of input points must be >= (2 * len(x_control_points) + 1)."
            )

        if endpoint_clamping:
            n_control_points -= 2 # endpoints are fixed to data

        # constant initial y-values guess, taken at 25% and 75% indices
        # init_guess = np.hstack((
        #     [points[len(points) * 1 // 4, 1]] * n_control_points,  # Approx. upper mid-point
        #     [points[len(points) * 3 // 4, 1]] * n_control_points   # Approx. lower mid-point
        # ))

        # initial guesses are taken from data points
        init_guess = points[
            np.linspace(0, len(points) - 1, n_control_points * 2, dtype=int)
        ][:,0]

        # Parametric locations of data on curve approximated by arc length
        u_target = cls.arc_lengths(points, normalize=True)

        if isinstance(spacing, np.ndarray):
            x_control_points = spacing
        elif isinstance(spacing, str):
            match spacing:
                case "cosine":
                    x_control_points = cosine_spacing(0, 1, n_control_points)
                case "linear":
                    x_control_points = np.linspace(0, 1, n_control_points)
                case "chebyshev":
                    x_control_points = chebyshev_nodes(0, 1, n_control_points )

        def combine_x_y(y_values):
            """
            Combine fixed x-values with optimized y-values into control points.
            Args:
                y_values (np.ndarray): Optimized y-values (concatenated upper and lower).

            Returns:
                np.ndarray: Complete control points, including clamped leading edge.
            """
            y_upper, y_lower = np.split(y_values, 2)
            control_points = np.vstack((
                np.column_stack((x_control_points[::-1], y_upper)),  # Upper surface
                [0, 0],                                              # Leading edge (clamped)
                np.column_stack((x_control_points, y_lower)),        # Lower surface
            ))
            return control_points

        # Define the objective function for fitting
        def objective(y_values):
            control_points = combine_x_y(y_values)
            if endpoint_clamping:
                control_points = np.vstack(
                    (points[0], control_points, points[-1])
                )
            curve_points = cls(control_points).evaluate_at(u_target)
            residuals = points - curve_points
            match method:
                case "least_squares":
                    return residuals.ravel()  # Flatten residuals
                case "L2_norm":
                    return np.linalg.norm(residuals, axis=1).sum()
                case _:
                    raise ValueError("Invalid method. Use 'least_squares' or 'L2_norm'.")

        # Perform optimization
        match method:
            case "least_squares":
                results = opt.least_squares(objective, init_guess)
            case "L2_norm":
                results = opt.minimize(objective, init_guess)
            case _:
                raise ValueError("Invalid method. Use 'least_squares' or 'L2_norm'.")

        # Generate final control points
        y_optimized = results.x
        control_points = combine_x_y(y_optimized)

        return cls(control_points, points)


class SplevBezier(Bezier):
    """Bezier curve subclass tailored for airfoil geometry."""

    def __init__(
        self,
        tck: Tuple[np.ndarray, np.ndarray, int],
        points: np.ndarray = None,
    ):
        """
        Initialize an AirfoilBezier instance.

        Args:
            control_points (np.ndarray): Control points defining the airfoil curve.
            points (np.ndarray, optional): Input coordinates for fitting the curve.
        """
        # super().__init__(points)
        self.points = points
        self.tck = tck

        # if len(self.knots) != self.n_control_points + self.degree + 1:
        #     raise ValueError(
        #         f"Invalid knot vector length. Expected {self.n_control_points + self.degree + 1}, got {len(self.knots)}."
        #     )

        # Validate monotonicity of knot vector
        if not all(self.knots[i] <= self.knots[i + 1] for i in range(len(self.knots) - 1)):
            raise ValueError("Knot vector must be non-decreasing.")

    @cached_property
    def knots(self):
        return self.tck[0]

    @cached_property
    def control_points(self):
        return self.tck[1].T

    @cached_property
    def degree(self):
        return self.tck[2]

    @cached_property
    def n_control_points(self):
        return len(self.control_points)

    def evaluate_at(self, u):
        return np.array(si.splev(u, self.tck)).T

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        n_control_points: int = 6,
        spacing: Union[str, np.ndarray] = "cosine",
        method: str = "least_squares",
        endpoint_clamping = False,
        beta = 0,
    ) -> "AirfoilBezier":
        """
        Fit an Bspline-based composite bezier curve through given airfoil
        coordinates.

        Args:
            points (np.ndarray): Input airfoil points of shape (m, 2).
                Must follow the order: trailing edge -> upper surface -> leading
                edge -> lower surface -> trailing edge.
            n_control_points (int): Number of control points per airfoil
                surface. Total number of control points will be 2*n+1, n for the
                upper surface, n for the lower surface, and one explicitly in
                the origin (leading edge).
            spacing (np.ndarray or str): Predefined x-locations for control
                points in array (from [0,1], without repeated points), or a
                string defining the type of spacing used.
            method (str): Fitting method, either "least_squares" or "L2_norm".

        Returns:
            AirfoilBezier: Fitted airfoil-specific Bézier curve.
        """

        degree = n_control_points - 1
        # input is control points per bezier part (upper and lower surface)
        n_control_points_per_side = n_control_points
        # consider leading edge control point is fixed, not all are optimized
        variables_per_side = n_control_points_per_side - 1
        # the actual total number of control points for the entire surface
        n_control_points = n_control_points * 2 - 1  # remove duplicated leading edge control point

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a 2D array with shape (m, 2).")

        if len(points) < n_control_points:
            raise ValueError(
                "Number of input points must be >= (2 * len(n_control_points) + 1)."
            )

        if isinstance(spacing, np.ndarray):
            x_control_points = spacing
        elif isinstance(spacing, str):
            match spacing:
                case "cosine":
                    x_control_points = cosine_spacing(0, 1, variables_per_side)
                case "linear":
                    x_control_points = np.linspace(0, 1, variables_per_side)
                case "chebyshev":
                    x_control_points = chebyshev_nodes(0, 1, variables_per_side )

        # in case the endpoints are fixed to data
        if endpoint_clamping:
            variables_per_side -= 2
            x_control_points = x_control_points[:-1]

        # initial guesses for y values are taken from data points
        init_guess = points[
            np.linspace(
                0, len(points) - 1,
                variables_per_side * 2,
                dtype=int
            ),
            1
        ] * 2
        # init_guess = np.concatenate((
        #     [0.5] * variables_per_side,
        #     [-0.5] * variables_per_side,
        # ))

        knot_vector = np.hstack(
            [
            [0.0] * (degree + 1),
            [0.5] * (degree),
            [1.0] * (degree + 1)
            ]
        )

        # Parametric locations of data on curve approximated by arc length
        u_target = cls.arc_lengths(points, normalize=True)

        def combine_x_y(y_values):
            """
            Combine fixed x-values with optimized y-values into control points.
            Args:
                y_values (np.ndarray): Optimized y-values (concatenated upper and lower).

            Returns:
                np.ndarray: Complete control points, including clamped leading edge.
            """
            y_upper, y_lower = np.split(y_values, 2)
            control_points = np.vstack((
                np.column_stack((x_control_points[::-1], y_upper)),  # Upper surface
                [0, 0],                                              # Leading edge (clamped)
                np.column_stack((x_control_points, y_lower)),        # Lower surface
            ))
            return control_points

        # Define the objective function for fitting
        def objective(y_values):
            control_points = combine_x_y(y_values)
            if endpoint_clamping:
                control_points = np.vstack(
                    (points[0], control_points, points[-1])
                )
            tck = (knot_vector, control_points.T, degree)
            curve_points = cls(tck).evaluate_at(u_target)
            residuals = points - curve_points

            smoothness_penalty = beta * np.sum(np.diff(y_values) ** 2)
            smoothness_penalty = beta * np.sum(np.linalg.norm(np.diff(control_points, axis=0), axis=1) ** 2)

            match method:
                case "least_squares":
                    return residuals.ravel() + smoothness_penalty  # Flatten residuals
                case "L2_norm":
                    return np.linalg.norm(residuals, axis=1).sum() + smoothness_penalty
                case "kdtree":
                    curve_points = cls(tck).evaluate_at(np.linspace(0, 1, 1000))
                    distances, _ = KDTree(curve_points).query(points)
                    return np.sum(distances)**2
                case _:
                    raise ValueError("Invalid method. Use 'least_squares' or 'L2_norm'.")

        # Perform optimization
        match method:
            case "least_squares":
                results = opt.least_squares(objective, init_guess, verbose=2)
            case "L2_norm" | "kdtree":
                results = opt.minimize(objective, init_guess)
            case _:
                raise ValueError(f"Invalid method {method}. Use 'least_squares' or 'L2_norm'.")

        # Generate final control points
        y_optimized = results.x
        control_points = combine_x_y(y_optimized)

        tck = (knot_vector, control_points.T, degree)

        return cls(tck, points)

class BBezier(Curve):
    """
    create a tck tuple to represent a B-spline curve which can be evaluated
    using splev, but behaves like a Bezier curve. This is done by fixing the
    knot vector and number of control points.

    k = degree = n_control_points - 1
    t = knot vector = [0, 0, 0, 1, 1, 1]
    """

    pass

class BSpline(Curve):
    """
    create a tck tuple to represent a B-spline curve which can be evaluated
    using splev, but behaves like a Bezier curve. This is done by fixing the
    knot vector and number of control points.

    k = degree = n_control_points - 1
    t = knot vector = [0, 0, 0, 1, 1, 1]
    """

    pass
