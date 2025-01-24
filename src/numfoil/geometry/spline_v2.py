from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
from scipy.spatial import KDTree
from scipy.special import comb
from warnings import warn as warning

from .geom2d import normalize_2d, rotate_2d_90ccw, Point2D, Geom2D
from ..util import cosine_spacing, chebyshev_nodes, ensure_1d_vector
# from .data import NormalizedAirfoilCoordinates

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
        # ! this definition is not fixed yet, in previous versions `points`
        # ! meant the resampled points of the curve, not the oringal input points
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
        numerator = ddy * dx - ddx * dy         # det([dx, dy], [ddx, ddy])
        denominator = (dx**2 + dy**2) ** 1.5    # abs([dx, dy])^3
        return np.where(denominator != 0, numerator / denominator, 0)

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


class Bezier(Curve):
    """Represents a Bezier curve using control points.

    Refs:
        https://www.youtube.com/watch?v=jvPPXbo87ds
    """

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

        Bspline knots are defined as the parameter boundary points (0,1), with
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
        message = (
            "Knots are not defined for a Bézier curve. "
            + "For a Bspline defintion of the bezier curve,"
            + " call the spline property."
        )
        # print(message)
        # this is messy but draws more attention (maybe?)
        try:
            raise(NotImplementedError(message))
        except NotImplementedError as e:
            print(repr(e))

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
        return_results: bool = False,
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

            return_results (bool, optional):
                whether to return the optimization results.
                Defaults to False.

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
        return (cls(control_points, points), results) if return_results else cls(control_points, points)


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


class SplevCBezier(BSpline2D):
    """Airfoil Surface `Composite Bezier` B-spline.
    Represents a B-spline curve (defined by tck tuple), which mimics a composite
    Bezier (CBezier) curve, clamped at the endpoints (x=1) and the leading
    edge (x=0, y=0).

    Refs:
        https://math.stackexchange.com/questions/2960974/convert-continuous-bezier-curve-to-b-spline

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
        # Validate number of knots
        if len(self.knots) != self.n_control_points + self.degree + 1:
            raise ValueError(
                f"Invalid number of knots. {len(self.knots)} provided but {self.n_control_points + self.degree + 1} required"
                )
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
        # validate degree
        if self.tck[2] != self.n_control_points - 1:
            raise ValueError("Degree and number of control points do not match.")
        return self.tck[2]

    @cached_property
    def n_control_points(self):
        """
        Retrieve the number of control points in the spline.

        Returns:
            int: The number of control points.
        """
        return len(self.control_points)


class SplevCBezierAirfoilSurface(SplevCBezier):
    """Airfoil Surface `Composite Bezier` B-Spline (CBBS).
    Represents a B-spline curve (defined by tck tuple), which mimics a composite
    Bezier (CBezier) curve, clamped at the endpoints (x=1) and the leading
    edge (x=0, y=0).
    """
    def __init__(self, tck, points = None):
        super().__init__(tck, points)

    @property
    def trailing_edge(self) -> Tuple[float, np.ndarray]:
        """Calculates the trailing edge point.

        Returns:
            float: The parameter value at the trailing edge (u).
            np.ndarray: The trailing edge point ([x, y]).
        """
        start_point = self.evaluate_at(0)
        end_point = self.evaluate_at(1)

        res1 = opt.minimize(lambda u: -np.linalg.norm(np.array([0,0])-self.evaluate_at(u)[0]), 0, bounds=[(0, 1)])
        res2 = opt.minimize(lambda u: -np.linalg.norm(np.array([0,0])-self.evaluate_at(u)[0]), 1, bounds=[(0, 1)])
        # if the maximum x value found is not the same at both ends of the
        # spline, the trailing edge is not properly defined and doubles back on
        # itself or the coordinates are missing one of the endpoints
        # ! this is still not ideal. If a trailing edge point is missing somehow
        # ! extrapolating might lead to a better result than just taking the
        # ! maximum x value. This is a quick fix for now.
        if abs(res1.fun - res2.fun) > 1e-5:
            # take location u with maximum x value, most likely to be trailing edge
            u_TE = res1.x[0] if -res1.fun>-res2.fun else res2.x[0]
            return u_TE, self.evaluate_at(u_TE)

        # if endpoints are both at same x-coordinate, return midpoint
        elif abs(start_point[0] - end_point[0]) < 1e-5:
            # todo: fix x value to 1 here (if close already)?
            return 0.5 * (start_point + end_point)
        else:
            raise ValueError("Trailing edge not properly defined, possible unaccounted edge case")
            # return start_point if start_point[0] > end_point[0] else end_point

    @property
    def leading_edge(self) -> Tuple[float, np.ndarray]:
        """Finds the leading edge point by maximizing the distance from the
        trailing edge.

        Additional options are availabe in
        :py:class:`.data.AirfoilProcessor.get_leading_edge`.

        returns:
            float: the parameter value at the leading edge (u) np.ndarray: the
            leading edge point ([x, y])
        """
        if trailing_edge is None:
            trailing_edge = self.trailing_edge

        # initial guess is midway the surface curve/spline
        init_guess = 0.5

        def objective(u):
            residuals = trailing_edge - self.evaluate_at(u)
            return -np.linalg.norm(residuals)

        result = opt.minimize(
            objective,
            init_guess,
            bounds=[(0, 1)],
            # method="SLSQP",
            )

        return result.x[0], self.evaluate_at(result.x[0])

    @property
    def chord_vector(self) -> np.ndarray:
            """Calculates the chord vector from the leading to trailing edge."""
            return self.trailing_edge - self.leading_edge

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        n_control_points: int = 12,
        spacing: Union[str, np.ndarray] = "cosine",
        w_damping = 1e-3,
        w_overlap = 1e3,
        clamp_origin = True,
    ) -> BSpline2D:
        """
        Fit an Bspline-based composite bezier curve through given airfoil
        coordinates. (Bspline which mimics a composite Bezier curve)

        knot vector for a composite bezier curve:
        - endpoints are clamped using multiplicity of (degree + 1)
        - middle knot (leading edge) is clamped using multiplicity (degree)
        - knot value is kept at 0.5, so evaluating at u = 0.5 should always
        return the leading edge
        - evaluating [0, 0.5[ should return the upper surface,
        ]0.5, 1] the lower surface

        Refs:
            https://math.stackexchange.com/questions/2960974/convert-continuous-bezier-curve-to-b-spline

        Args:
            points (np.ndarray):
                Input airfoil points of shape (m, 2). Must follow the order:
                trailing edge -> upper surface -> leading edge -> lower surface
                -> trailing edge.

            n_control_points (int):
                Number of control points per airfoil surface. Total number of
                control points will be 2*n+1, n for the upper surface, n for the
                lower surface, and one explicitly in the origin (leading edge).

            spacing (np.ndarray or str):
                Predefined x-locations for control points in array (from [0,1],
                without repeated points), or a string defining the type of
                spacing used.
                Defaults to "cosine".

            w_damping (float):
                Weight for the damping term in the optimization objective.
                Defaults to 1e-3.

            w_overlap (float):
                Weight for the overlap penalty in the optimization objective.
                Defaults to 1e-3.

            clamp_origin (bool):
                If True, the leading edge is clamped to the origin.
                This requires normalized input data
                Defaults to True.

            return_results (bool): If True, return the optimization results.
            TODO: return_results will be removed at some point.

        Returns:
            AirfoilBezier: Fitted airfoil-specific Bézier curve.
        """
        # if clamp_origin and not isinstance(points, NormalizedAirfoilCoordinates):
        #     warning(
        #         "Clamping the leading edge to the origin requires normalized input data.\n"
        #         + "If the input data is not an instance of the :py:class:`NormalizedAirfoilCoordinates` this can not be verified"
        #     )

        # First redefine some of the parameters needed
        # The degree is per definition:
        degree = n_control_points - 1

        # input number of ctrl points is control points per bezier part,
        # so both upper and lower surface have the defined number of ctrl points
        # Note: this still includes the leading edge point for both surfaces
        n_control_points_per_side = n_control_points

        # Considering leading edge control point is fixed, not all are optimized
        variables_per_side = n_control_points_per_side - 1

        # The actual total number of control points for the entire surface is
        # twice the number of points per side, but remember to remove duplicated
        # leading edge control point
        n_control_points = n_control_points * 2 - 1

        # The knot vector for a composite bezier curve.
        # The knot value (leading edge) is fixed at 0.5
        knot_vector = np.hstack(
            [
            [0.0] * (degree + 1),
            [0.5] * (degree),
            [1.0] * (degree + 1)
            ]
        )

        # Validate input points
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a 2D array with shape (m, 2).")

        # Check if there are enough points to fit the curve
        if len(points) < n_control_points:
            raise ValueError(
                "Number of input points must be >= (2 * len(n_control_points) + 1)."
                + f"\n Got {len(points)} points, expected at least {2 * len(n_control_points) + 1} for a curve with {n_control_points}."
            )

        # Check how spacing is defined, and generate x values accordingly
        if isinstance(spacing, np.ndarray):
            x_control_points = spacing

        elif isinstance(spacing, str):
            # if the curve is not clamped to the origin, control point spacing
            # must account for the non-normalized input data
            if not clamp_origin:
                x_values = points.T[0]
                x_min = x_values.min()
                x_max = x_values.max()
            else:
                x_min = 0
                x_max = 1
            match spacing:
                case "cosine":
                    x_control_points = cosine_spacing(x_min, x_max, variables_per_side)
                case "linear":
                    x_control_points = np.linspace(x_min, x_max, variables_per_side)
                case "chebyshev":
                    x_control_points = chebyshev_nodes(x_min, x_max, variables_per_side )


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
            # double values to move control points outward (arbitrary)
            ] * 2

        # # alternate fixed initial guesses. While this should be the safer
        # # approach, it does not always work as well as the sampled guess.
        # init_guess = np.concatenate((
        #     [0.2] * variables_per_side,
        #     [-0.2] * variables_per_side,
        # ))

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
            return control_points.view(Point2D)

        def objective(y_values: np.ndarray) -> np.ndarray:
            """Objective function for control point fitting optimization.

            Args:
                y_values (np.ndarray): y values of control points to be found.

            Returns:
                np.ndarray: residuals between data points and curve points, or
                    the L2 norm of residuals depending on the chosen method.
            """
            control_points = combine_x_y(y_values)

            # create spline definition
            tck = (knot_vector, control_points.T, degree)

            # Penalize oscillations in y values
            # smoothness_penalty = w_damping * np.sum(np.diff(y_values) ** 2)
            smoothness_penalty = w_damping * np.mean(np.diff(y_values)) ** 2

            # Penilize corssover / intersection of upper and lower surface
            overlap_penalty = w_overlap * np.sum(
                np.maximum(
                    0,
                    - cls(tck).evaluate_at(np.linspace(0, 0.2, 200)).T[1]
                    + cls(tck).evaluate_at(np.linspace(0.8, 1, 200)).T[1][::-1]
                )
            ) ** 2

            # get a dens evaluation of the curve to create a KDTree
            curve = cls(tck).evaluate_at(np.linspace(0, 1, 5000))

            # query the KDTree to get distances to the input data points
            distances, _ = KDTree(curve).query(points)

            # final objective is the square of the sum of the distances,
            # plus the smoothness and overlap penalties
            return np.sum(distances)**2 + smoothness_penalty + overlap_penalty

        # Define constraints for optimization
        constraints = [
                {
                    # Trailing edge points must have symmetric y values
                    # This helps properly define the trialing edge
                    "type": "eq",
                    "fun": lambda y: y[0] + y[-1]  # = 0
                },
                {
                    # Upper trailing edge point must have y >= 0
                    # Probably redundant, but just to be sure
                    "type": "ineq",
                    "fun": lambda y: y[0]  # >= 0
                }
            ]

        # Perform optimization
        results = opt.minimize(
            objective, init_guess,
            constraints=constraints,
            options={
                # "method": "L-BFGS-B",
                "disp": True,
                # "ftol": 1e-9,
                "maxiter": 1000,
                },
            tol=1e-6
        )

        # if something groes wrong
        if not results.success:
            print(results)
            raise ValueError(f"Curve fit failed: {results.message}")

        # Generate final control points
        y_optimized = results.x
        control_points = combine_x_y(y_optimized)

        # return the fitted curve
        return cls((knot_vector, control_points.T, degree), points)


class BezierAirfoilSurface(Bezier):
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


class CSTCurve(Curve):
    """
    A class representing a single-valued CST curve: y(x).

    The standard CST formulation for y(x) can be written as:
        :math:`y(x) = C(x) * S(x)`
    where
        :math:`C(x) = x^n1 (1 - x)^n2`
        :math:`S(x) = \sum_{k=0}^n \binom{n}{k} a_k x^k (1 - x)^{n-k}`

    Args:
        coefficients (list or array of floats):
            The shape (Bernstein polynomial) coefficients a_k.

        n1, n2 (float):
            The exponents in the C(x) term.
            Default values are n1 = 0.5, n2 = 1.0

        verbose (bool):
            Print optimization results.
            Default is False.

    Attributes:
        n (int): Number of Bernstein terms.
        k_vec (np.ndarray): Index vector for Bernstein polynomials.
        binomial_weights (list): List of binomial coefficients.
    """
    def __init__(self, coefficients: np.ndarray, n1: float = 0.5, n2: float = 1.0):
        self.n1 = n1
        self.n2 = n2
        self.coefficients = np.array(coefficients, dtype=float)

        # Polynomial degree (number of Bernstein terms)
        self.n = len(coefficients) - 1
        # index vector for Bernstein polynomials
        self.k_vec = np.arange(self.n + 1)

    def class_function(self, x: np.ndarray) -> np.ndarray:
        """Class function for CST curve.
        C(x) = x^n1 (1-x)^n2

        :math:`C(x) = x^{n1} (1-x)^{n2}`

        Args:
            x (np.ndarray): Input array. shape (N,).

        Returns:
            np.ndarray: Class function values. shape (N,)
        """
        return np.power(x, self.n1) * np.power(1.0 - x, self.n2)

    @cached_property
    def binomial_weights(self):
        """Precompute binomial coefficients for faster evaluation.

        :math:`\binom{n}{k}`

        Returns:
            list: List of binomial coefficients.
        """
        return comb(self.n, self.k_vec)

    def bernstein_basis(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Bernstein polynomial.
        B(x, n, k) = x^k (1 - x)^(n-k), where k is the current index.
        vectorized this becomes:
        B(x) = x^k (1 - x)^(n-k), where k = [0, 1, ..., n] is a vector.

        :math:`B_{n,k}(x) = x^k (1 - x)^{n-k}`

        Args:
            x (np.ndarray): Input array. Shape (N,).

        Returns:
            np.ndarray: Bernstein polynomial values.
                Shape (N, n+1) where column k is B_k(x).
        """
        x = np.array(x).reshape(-1, 1) # if np.array(x).ndim == 1 else x
        return np.power(x, self.k_vec) * np.power(1.0 - x, self.n - self.k_vec)

    def weighted_basis_matrix (self, x: np.ndarray) -> np.ndarray:
        """Construct the weighted Bernstein Basis matrix.

        :math:`M(x) = C(x) * B(x) * binom_coeffs`

        Args:
            x (np.ndarray): Input array. Shape (N,).

        Returns:
            np.ndarray: Bernstein Basis matrix. Shape (N, n+1)
        """
        x = np.array(x).reshape(-1, 1) # if np.array(x).ndim == 1 else x
        return (
            self.class_function(x)
            * self.bernstein_basis(x)
            * self.binomial_weights
        )

    def shape_function(self, x: np.ndarray) -> np.ndarray:
        """
        Shape function for CST curve.

        :math:`S(x) = \sum_{k=0}^n a_k B_{n,k}(x)`

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Shape function values.
        """
        return np.sum(
                self.coefficients
                * self.binomial_weights
                * self.bernstein_basis(x),
            axis=-1,
        )

    # Aliases
    C = class_function
    S = shape_function
    B = bernstein_basis
    M = weighted_basis_matrix

    def evaluate_at(self, x: np.ndarray | float) -> Point2D:
        """
        Evaluate the CST curve at a given x in [0, 1].
        """
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be in the range [0, 1]")

        y = self.class_function(x) * self.shape_function(x)
        return np.array([x, y]).T.view(Point2D)

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        An alternative evaluation of the CST curve at x in [0, 1].

        :math:`y(x) = M(x) * a`
        """
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("x must be in the range [0, 1]")
        return self.M(x) @ self.coefficients

    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        num_coefficients: int = 6,
        n1: float = 0.5,
        n2: float = 1.0,
        verbose: bool = False,
    ):
        """
        Fit the CST coefficients to the provided 2D data points using direct
        least-squares when n1 and n2 are fixed.

        Args:
            data (np.ndarray): 2D data points to fit curve to.
            num_coefficients (int): Number of Bernstein coefficients.
                Default is 6.
            n1 (float): Exponent for C(x) term.
                Default is 0.5.
            n2 (float): Exponent for C(x) term.
                Default is 1.0.
            verbose (bool): Print fitting diagnostics.
                Default is False.

        Returns:
            CSTCurve: A fitted CSTCurve instance.
        """
        if not isinstance(data, Geom2D):
            data = data.view(Point2D)

        x, y = data.x[:, None], data.y

        n = num_coefficients - 1        # Polynomial degree (number of Bernstein terms)
        k_vec = np.arange(n + 1)        # index vector for Bernstein polynomials
        binom_coeffs = comb(n, k_vec)   # binomial coefficients

        Cx = np.power(x, n1) * np.power(1.0 - x, n2)            # Class function
        Bx = np.power(x, k_vec) * np.power(1.0 - x, n - k_vec)  # Bernstein polynomial matrix

        Mx = Cx * Bx * binom_coeffs     # Weighted Bernstein basis matrix

        # Solve for the coefficients using least squares
        cst, residuals, rank, svals = np.linalg.lstsq(Mx, y, rcond=None)

        if verbose:
            print(f"Residuals: {residuals}")
            print(f"Rank of M: {rank}")
            print(f"Singular values: {svals}")

        return cls(cst, n1=n1, n2=n2)

    # Ugly stuff
    def class_first_deriv(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """ First derivative of the class function

        :math:`\frac{dC(x)}{dx} = n1 x^{n1 - 1} (1 - x)^n2 - n2 x^n1 (1 - x)^{n2 - 1}`

        Args:
            x (np.ndarray): Input array chordwise location.

        Returns:
            np.ndarray: First derivative of the class function
        """
        return(
        self.n1 * x**(self.n1 - 1) * (1 - x)**self.n2
        - self.n2 * x**self.n1 * (1 - x)**(self.n2 - 1)
    )

    def class_second_deriv(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """ Second derivative of the class function

        :math:`\frac{d^2C(x)}{dx^2} = n1 (n1 - 1) x^{n1 - 2} (1 - x)^n2 - 2 n1
        n2 x^{n1 - 1} (1 - x)^{n2 - 1} + n2 (n2 - 1) x^n1 (1 - x)^{n2 - 2}`

        Args:
            x (np.ndarray): Input array chordwise location.

        Returns:
            np.ndarray: Second derivative of the class function
        """
        return(
            self.n1 * (self.n1 - 1) * x**(self.n1 - 2) * (1 - x)**self.n2
            - 2 * self.n1 * self.n2 * x**(self.n1 - 1) * (1 - x)**(self.n2 - 1)
            + self.n2 * (self.n2 - 1) * x**self.n1 * (1 - x)**(self.n2 - 2)
        )

    def shape_first_deriv(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """ First derivative of the shape function

        :math:`\frac{dS(x)}{dx} = \sum_{k=0}^n a_k \binom{n}{k} (k x^{k - 1} (1
        - x)^{n - k} - (n - k) x^k (1 - x)^{n - k - 1})`

        Args:
            x (np.ndarray): Input array chordwise location.

        Returns:
            np.ndarray: First derivative of the shape function
        """
        return np.sum(
            self.coefficients
            * self.binomial_weights
            * (
                self.k_vec
                * np.power(x, self.k_vec - 1)
                * np.power(1 - x, self.n - self.k_vec)
            ),
            axis=-1,
        )

    def shape_second_deriv(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """ Second derivative of the shape function

        :math:`\frac{d^2S(x)}{dx^2} = \sum_{k=0}^n a_k \binom{n}{k} ((k - 1) k
        x^{k - 2} (1 - x)^{n - k} - 2 (n - k) k x^{k - 1} (1 - x)^{n - k - 1} +
        (n - k - 1) (n - k) x^k (1 - x)^{n - k - 2})`

        Args:
            x (np.ndarray): Input array chordwise location.

        Returns:
            np.ndarray: Second derivative of the shape function.
        """
        return np.sum(
            self.coefficients
            * self.binomial_weights
            * self.k_vec * (self.k_vec - 1)
            * np.power(x, self.k_vec - 2)
            * np.power(1 - x, self.n - self.k_vec),
            axis=-1,
        )

    def first_deriv_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """ First derivative of the CST curve at x.
        :math:`\frac{dy}{dx} = \frac{dC(x)}{dx} S(x) + C(x) \frac{dS(x)}{dx}`

        Args:
            x (Union[float, np.ndarray]): chordwise location

        Returns:
            np.ndarray: First derivative of the CST curve at x.
        """
        return (
            self.class_first_deriv(x) * self.shape_function(x)
            + self.class_function(x) * self.shape_first_deriv(x)
        )

    def second_deriv_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """ Second derivative of the CST curve at x.

        :math:`\frac{d^2y}{dx^2} = \frac{d^2C(x)}{dx^2} S(x) + 2 \frac{dC(x)}{dx} \frac{dS(x)}{dx} + C(x) \frac{d^2S(x)}{dx^2}`

        Args:
            x (Union[float, np.ndarray]): Input array chordwise location.

        Returns:
            np.ndarray: Second derivative of the CST curve at x.
        """
        return (
            self.class_second_deriv(x) * self.shape_function(x)
            + 2 * self.class_first_deriv(x) * self.shape_first_deriv(x)
            + self.class_function(x) * self.shape_second_deriv(x)
        )

    # Aliases
    dC = class_first_deriv
    d2C = class_second_deriv
    dS = shape_first_deriv
    d2S = shape_second_deriv


class CSTAirfoilSurface:
    """
    A parametric CST-based airfoil curve, x(u), y(u), covering upper and lower surfaces.

    Here, we define a parameter u in [0,1] that 'walks' around the airfoil from the trailing edge,
    around the leading edge, and back to the trailing edge.

    For simplicity, we assume:
      - 0 <= u <= 0.5 describes the upper surface,
      - 0.5 <= u <= 1.0 describes the lower surface.

    This is just one of many ways to define a closed parametric curve using CST.
    """
    def __init__(
        self,
        upper_surface: CSTCurve,
        lower_surface: CSTCurve,
    ):
        """
        We store two separate single-valued CST curves internally for the upper and lower surfaces.
        """
        self.upper_part = upper_surface
        self.lower_part = lower_surface

    @classmethod
    def fit(cls, data: np.ndarray, num_coefficients: int = 6, n1: float = 0.5, n2: float = 1.0):
        """
        Fit the CST coefficients to the provided 2D data points using direct
        least-squares when n1 and n2 are fixed.

        Args:
            data (np.ndarray): 2D data points to fit curve to.
            num_coefficients (int): Number of Bernstein coefficients.
                Default is 6.

        Returns:
            AirfoilCST: A fitted AirfoilCST instance.
        """
        if not isinstance(data, Geom2D):
            data = data.view(Point2D)

        # upper = CSTCurve.fit(data[:len(data)//2][::-1], num_coefficients, n1, n2)
        # lower = CSTCurve.fit(data[len(data)//2:], num_coefficients, n1, n2)

        switch_idx = np.where(np.diff(np.sign(np.diff(data[:, 0]))))[0][0] + 1

        upper = CSTCurve.fit(data[:switch_idx][::-1], num_coefficients, n1, n2)
        lower = CSTCurve.fit(data[switch_idx:], num_coefficients, n1, n2)

        return cls(upper, lower)

    def x(self, u):
        """
        Example parameter-to-x mapping for airfoil.
        By default, we can just let x decrease from 1 at the trailing edge (u=0)
        to 0 at the leading edge (u=0.5), and then increase back to 1 at the trailing edge (u=1).
        One simple linear mapping is:
            x = 1.0 - 2*abs(u - 0.5)
        for u in [0, 1].
        """
        return np.abs(1 - 2 * u)

    def evaluate_at(self, u):
        """
        Evaluate x(u), y(u) at a parametric point u in [0,1].

        Returns:
            np.ndarray: [x_val, y_val]
        """
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError("Parameter u must be in [0,1].")

        x = self.x(u)
        if np.array(x).ndim == 0:
            y = self.upper_part(x) if u <= 0.5 else self.lower_part(x)
        else:
            y = np.zeros_like(x)
            y[u <= 0.5] = self.upper_part(x[u <= 0.5])
            y[u >= 0.5] = self.lower_part(x[u >= 0.5])
        return np.array([x, y]).T.view(Point2D)

    def __call__(self, u):
        return self.evaluate_at(u)

