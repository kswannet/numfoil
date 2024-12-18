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

    To try and keep things general,``x`` is used as the location parameter here.

    Evaluations are either done at a location ``x`` indicating the location
    along the x-axis on a 1D curve, or a location defined by a parameter ``u``
    indicating the location on a 2D parametric curve.
    """

    @cached_property
    def points(self) -> np.ndarray:
        """Retrieve the (input) points used to define the curve."""
        raise NotImplementedError

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

    def radius_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Return the radius of curvature on the curve at given location ``u``."""
        curvature = self.curvature_at(x)
        return np.where(curvature != 0, 1 / curvature, np.inf)

    @staticmethod
    def arc_lengths(points: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Approximate parameter values `u` for input points based on cumulative
        arc length with linear approximation.

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


class ParametricCurve(Curve, ABC):
    """More specific curve baseclass for parametric curves."""

    @cached_property
    def spline(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Retrieve or create the spline representation of the curve."""
        raise NotImplementedError

    @cached_property
    def control_points(self) -> np.ndarray:
        """Retrieve the control points of the spline."""
        return self.spline[1].T

    @cached_property
    def knots(self) -> np.ndarray:
        """Retrieve the knot vector of the spline."""
        return self.spline[0]

    @cached_property
    def degree(self) -> int:
        """Retrieve the degree of the spline."""
        if self.spline[2] != self.n_control_points - 1:
            raise ValueError(
                "Degree and number of control points do not match."
                )
        return self.n_control_points - 1

    @cached_property
    def n_control_points(self):
        """
        Retrieve the number of control points in the spline.

        Returns:
            int: The number of control points.
        """
        return len(self.control_points)

    def find_u(
            self,
            points: np.ndarray,
            method: str = "least_squares",
            output: str = "u",
        ) -> float:
        """Find the parameter value ``u`` of the location on the curve closest
        to a given point ``(x, y)``.

        Args:
            points (np.ndarray):
                A set of points to find the parametric value for.

            method (str):
                Optimization method to use.
                Options are "least_squares" or "L2_norm".
                Defaults to "least_squares".

            output (str):
                What to return. Options are "u" (or "x") or "result".
                Defaults to "u".

        Returns:
            float: The parameter value ``u`` for the given point.
            OR
            opt.OptimizeResult: The full optimization result.
        """
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
                    # bounds=[(0., 1)]*len(points),
                    # method="SLSQP"
                    )
            case _:
                raise ValueError(
                    "Invalid method. Use 'least_squares' or 'L2_norm'."
                )

        if not result.success:
            raise ValueError(
                "failed to find u, optimization success {result.success}"
                )
        match output:
            # in case you'd want the full result...
            case "result":
                return result
            # ...but probably you just want this
            case "u" | "x":
                return result.x


class BSpline2D(ParametricCurve):
    """Creates a splined representation of a set of points.
    This uses the scipy.interpolate.splprep function to create a 2D
    -INTERPOLATING- spline. This means that the spline will pass through all
    points given,

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
        """
        Evaluate the curve at specified parameter values ``u``.

        Args:
            u (float or array-like): Parameter value(s) at which to evaluate
                the curve.

        Returns:
            ndarray: The evaluated point(s) on the spline.
        """
        return np.array(si.splev(u, self.spline, der=0), dtype=np.float64).T

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the curve's first derivative(s) at specified parameter values
        ``u``.

        Args:
            u (float or array-like): Parameter value(s) at which to evaluate
                the curve.

        Returns:
            ndarray: The evaluated point(s) on the spline.
        """
        return np.array(si.splev(u, self.spline, der=1), dtype=np.float64).T

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the curve's second derivative(s) at specified parameter values
        ``u``.

        Args:
            u (float or array-like): Parameter value(s) at which to evaluate
                the curve.

        Returns:
            ndarray: The evaluated point(s) on the spline.
        """
        return np.array(si.splev(u, self.spline, der=2), dtype=np.float64).T

    @property
    def max_curvature(self) -> Tuple[float, float]:
        """Finds maximum curvature of the curve.

        Returns:
            Tuple: tuple of the location and value of the maximum curvature:
                [0]: float: parametric location ``u`` of max curvature.
                [1]: float: maximum curvature value \kappa_{max}.

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


class SplevCBezier(BSpline2D):
    """Represents a B-spline curve (defined by tck tuple), which mimics a
    composite Bezier (CBezier) curve, clamped at the endpoints and the leading
    edge.

    Use fit method to fit the curve to a set of airfoil coordinate points in
    selig format.
    """
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
        self.points = points
        self.tck = tck

    @cached_property
    def spline(self):
        """
        Retrieve the spline representation (tck tuple).

        The tck format consists of:
            - t: Knot vector (non-decreasing sequence).
            - c: Control points.
            - k: Degree of the spline.

        Returns:
            tuple: A tuple (t, c, k) representing the spline.
        """
        return self.tck

    @cached_property
    def knots(self):
        """
        Retrieve the knot vector of the spline.

        The knot vector must be non-decreasing. This method validates the monotonicity
        before returning the knot vector.

        Raises:
            ValueError: If the knot vector is not non-decreasing.

        Returns:
            list or ndarray: Knot vector of the spline.
        """
        # Validate monotonicity of knot vector
        if not all(self.knots[i] <= self.knots[i + 1] for i in range(len(self.knots) - 1)):
            raise ValueError("Knot vector must be non-decreasing.")
        return self.tck[0]

    @cached_property
    def control_points(self):
        """
        Retrieve the control points of the spline.

        Returns:
            ndarray: An array of control points for the spline.
        """
        return self.tck[1].T

    @cached_property
    def degree(self):
        """
        Retrieve the degree of the spline.

        The degree defines the polynomial order of the spline segments.

        Returns:
            int: The degree of the spline.
        """
        return self.tck[2]

    @cached_property
    def n_control_points(self):
        """
        Retrieve the number of control points in the spline.

        Returns:
            int: The number of control points.
        """
        return len(self.control_points)

    # TODO add knot/clamp locations as input?
    # TODO remove other methods from ``fit`` method, only kdtree works properly
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
        # TODO : remove other methods from ``fit`` method, only kdtree works properly
        method = "kdtree"  # the other ones dont work properly, but im leaving them in for now

        degree = n_control_points - 1
        # input is control points per bezier part (upper and lower surface)
        n_control_points_per_side = n_control_points
        # consider leading edge control point is fixed, not all are optimized
        variables_per_side = n_control_points_per_side - 1
        # the actual total number of control points for the entire surface
        n_control_points = n_control_points * 2 - 1  # remove duplicated leading edge control point

        # knot vector for a composite bezier curve
        # - endpoints are clamped using multiplicity of (degree + 1)
        # - middle knot (leading edge) is clamped using multiplicity (degree)
        # - knot value is kept at 0.5, so evaluating at u = 0.5 should always
        #   return the leading edge
        # - evaluating [0, 0.5[ should return the upper surface,
        #   ]0.5, 1] the lower surface
        knot_vector = np.hstack(
            [
            [0.0] * (degree + 1),
            [0.5] * (degree),
            [1.0] * (degree + 1)
            ]
        )

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
            # indexing linearly spaced points from the input data
            np.linspace(
                0, len(points) - 1,
                # need 2x the number of variables per side
                variables_per_side * 2,
                # indeces must be integers
                dtype=int
            # indexing only the y values
            ),1
            # double values to move control points outwards (arbitrary)
            ] * 2

        # # alternate fixed initial guesses
        # init_guess = np.concatenate((
        #     [0.5] * variables_per_side,
        #     [-0.5] * variables_per_side,
        # ))

        # Parametric locations of data on curve approximated by arc length
        u_target = cls.arc_lengths(points, normalize=True)

        # TODO : Add knot/clamp locations as input?
        # TODO | Then split control points in k+1 parts, and account for
        # TODO | variability when combining x and y values with the clamped
        # TODO | points?
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
        def objective(y_values: np.ndarray) -> np.ndarray:
            """Objective function for control point fitting optimization.

            Args:
                y_values (np.ndarray): y values of control points to be found.

            Returns:
                np.ndarray: residuals between data points and curve points, or
                    the L2 norm of residuals depending on the chosen method.
            """
            control_points = combine_x_y(y_values)
            if endpoint_clamping:
                control_points = np.vstack(
                    (points[0], control_points, points[-1])
                )
            tck = (knot_vector, control_points.T, degree)
            curve_points = cls(tck).evaluate_at(u_target)
            residuals = points - curve_points

            smoothness_penalty = beta * np.sum(np.diff(y_values) ** 2)
            # smoothness_penalty = beta * np.sum(np.linalg.norm(np.diff(control_points, axis=0), axis=1) ** 2)

            match method:
                case "least_squares":
                    return residuals.ravel() + smoothness_penalty  # Flatten residuals
                case "L2_norm":
                    return np.linalg.norm(residuals, axis=1).sum() + smoothness_penalty
                case "kdtree":
                    curve_points = cls(tck).evaluate_at(np.linspace(0, 1, 500))
                    distances, _ = KDTree(curve_points).query(points)
                    return np.sum(distances)**2 + smoothness_penalty
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
        self.points = points

    @cached_property
    def degree(self) -> int:
        """Return the degree of the Bézier curve.
        degree k = n_control_points - 1

        Returns:
            int: The degree of the Bézier curve.
        """
        return self.n_control_points - 1

    @cached_property
    def n_control_points(self):
        """
        Retrieve the number of control points in the spline.

        Returns:
            int: The number of control points.
        """
        return len(self.control_points)

    @cached_property
    def spline(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Retrieve the Bspline spline representation of the Bézier curve, which
        can be used with scipy.interpolate.splev to evaluate the curve.

        This has no real purpose for this Bezier class definition, but could be
        useful for interfacing with other code that expects a spline.

        Bspline knots are defined as the parameter boundary points, with
        multiplicity (degree + 1) to clamp the endpoints.

        Returns:
            tuple: A tuple (t, c, k) representing the spline.
        """
        knot_vector = np.hstack(
            [
            [0.0] * (self.degree + 1),
            [1.0] * (self.degree + 1)
            ]
        )
        return (knot_vector, self.control_points, self.degree)

    @cached_property
    def knots(self):
        """
        Bezier curve does not have knots, but this property is defined for
        consistency with other classes. Additionaly, a Bspline definition of
        the Bezier curve can be obtained by calling the spline property.

        Alternative implementation:
            Retrieve the knot vector of the spline.
            ``t = self.spline[0]``

            Returns:
                np.ndarray: Knot vector of the spline.
        """
        # return self.spline[0]
        print("Knots are not defined for a Bézier curve. "
            + "For a Bspline defintion of the bezier curve,"
            + " call the spline property."
        )

    @cached_property
    def coefficient_matrix(self) -> np.ndarray:
        """Compute the characteristic matrix of coefficients for a Bézier curve.

        This method calculates the matrix of coefficients used in the construction
        of a Bézier curve of a given degree. The matrix is computed using binomial
        coefficients and combinatorial logic.

        Returns:
            np.ndarray: A 2D numpy array representing the characteristic matrix of
            coefficients for the Bézier curve.
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
            u (float or array-like): Parametric location(s) to evaluate. Must be in the range [0, 1].

        Returns:
            np.ndarray: Evaluated curve point(s) as a 2D array. For scalar input, returns shape (2,).
        """
        u = ensure_1d_vector(u)
        # Basis matrix [1, t, t^2, ..., t^n]
        # basis = u[:, None] ** np.arange(self.n_control_points)
        basis = np.power.outer(u, np.arange(self.n_control_points))
        return basis @ self.coefficient_matrix @ self.control_points

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the Bézier curve's first derivative at parametric
        location(s) ``u``.

        Args:
            u (float or array-like): Parametric location(s) to evaluate.
            Must be in the range [0, 1].

        Returns:
            np.ndarray: Evaluated curve point(s) as a 2D array. For scalar
            input, returns shape (2,).
        """
        u = ensure_1d_vector(u)
        # Basis matrix for (n-1)-degree Bernstein polynomials
        # Basis matrix [0, 1, 2t, ..., (n-1)t^(n-2)]
        powers = np.arange(self.n_control_points)
        basis = powers * u[:, None] ** (powers - 1)
        return (basis @ self.coefficient_matrix @ self.control_points )

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the Bézier curve's second derivative at parametric
        location(s) ``u``.

        Args:
            u (float or array-like): Parametric location(s) to evaluate.
            Must be in the range [0, 1].

        Returns:
            np.ndarray: Evaluated curve point(s) as a 2D array. For scalar
            input, returns shape (2,).
        """
        u = ensure_1d_vector(u)
        # Basis matrix for (n-2)-degree Bernstein polynomials
        # Basis matrix [0, 0, 2, 6t, ..., (n-2)(n-1)t^(n-3)]
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
        """Fit be the Bézier curve to the given points.

        Args:
            points (np.ndarray):
                coordinate points to fit the curve to.

            n_control_points (int):
                number of control points to fit.

            endpoint_clamping (bool, optional):
                whether the endpoints (control points) should be forced to
                coincide with the first and lastdatapoint.
                Defaults to True.

            method (str, optional):
                which optimization method to use.
                Options are "least_squares" or "L2_norm".
                Defaults to "least_squares".

            beta (float, optional):
                regularization weight parameter for control point location.
                Defaults to 0.

        Raises:
            ValueError:
                input points must be a 2D array with shape (m, 2).

            ValueError:
                Number of points must be greater than or equal to
                n_control_points.

            ValueError:
                invalid optimization method.

        Returns:
            Bezier: _description_
        """

        # input validation
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a 2D array with shape (m, 2).")
        if len(points) < n_control_points:
            raise ValueError(
                "Number of points must be greater than or equal to n_control_points."
            )

        # adjust number of control points if endpoints are fixed
        if endpoint_clamping:
            n_control_points -= 2 # endpoints are fixed to data

        # initial guesses are taken from data points
        init_guess = points[
            np.linspace(0, len(points) - 1, n_control_points, dtype=int)
        ]

        # Parametric locations of data on curve approximated by arc length
        u_target = cls.arc_lengths(points, normalize=True)

        def objective(flat_control_points: np.ndarray) -> np.ndarray:
            """Objective function for control point fitting optimization.

            Args:
                flat_control_points (array): flattened control points.

            Returns:
                np.ndarray: residuals between data points and curve points.
            """
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
                    )
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
                case "kdtree":
                    curve_points = cls(control_points).evaluate_at(
                        np.linspace(0, 1, 500)
                    )
                    distances, _ = KDTree(curve_points).query(points)
                    return (
                        np.sum(distances)**2
                        + smoothness_penalty
                    )

        match method:
            case "least_squares":
                results = opt.least_squares(
                    objective, init_guess.ravel()
                    )
            case "L2_norm" | "kdtree":
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

