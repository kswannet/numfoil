"""Contains py:class:`BSpline` for splining 2D points."""

from functools import cached_property
from typing import Optional, Union, Tuple

import numpy as np
import scipy.interpolate as si
from scipy.optimize import minimize

from scipy.spatial import KDTree

from .geom2d import normalize_2d, rotate_2d_90ccw
from ..util import cosine_spacing


class BSpline2D:
    """Creates a splined representation of a set of points.

    Args:
        points: A set of 2D row-vectors
        degree: Degree of the spline. Defaults to 3 (cubic spline).
    """

    def __init__(self, points: np.ndarray, degree: Optional[int] = 3, smoothing: Optional[float] = 0.0):
        self.points = points
        self._degree = degree
        self.smoothing = smoothing

    @property
    def degree(self) -> int:
        return self._degree

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
                #//[1]: Parametric points, u, used to create the spline
        """
        return si.splprep(self.points.T, s=self.smoothing, k=self.degree)[0]

    def evaluate_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline point(s) at ``u``."""
        return np.array(si.splev(u, self.spline, der=0), dtype=np.float64).T

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        return np.array(si.splev(u, self.spline, der=1), dtype=np.float64).T

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        return np.array(si.splev(u, self.spline, der=2), dtype=np.float64).T

    def tangent_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline tangent(s) at ``u``."""
        # return normalize_2d(
        #     np.array(si.splev(u, self.spline, der=1), dtype=np.float64).T
        # )
        return normalize_2d(self.first_deriv_at(u))

    def normal_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline normals(s) at ``u``."""
        return rotate_2d_90ccw(self.tangent_at(u))

    def curvature_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Calculate the spline curvature at ``u``
        k=|y"(x)| / (1+(y'(x))^2)^{3/2}
        """
        # a = np.abs(self.second_deriv_at(u)[1])
        # b = (1+self.first_deriv_at(u)[1]**2)**(3/2)
        # return a/b if b != 0 else 0
        dx, dy = self.first_deriv_at(u).T
        ddx, ddy = self.second_deriv_at(u).T
        return ddy * dx - ddx * dy / (dx**2 + dy**2)**1.5

    def radius_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        # TODO can this be vectorized?
        """returns the radius of curvature on the spline at given location u

        Args:
            u (Union[float, np.ndarray]): locaiton on the spline u

        Returns:
            np.ndarray: curvature values
        """
        curvature = self.curvature_at(u)
        return 1 / curvature if curvature != 0 else np.inf

    # @cached_property
    @property
    def max_curvature(self) -> Tuple[float, float]:
        """Finds maximum curvature of the spline
        Returns:
            Tuple[float, np.float]: [u, curvature]: max curvature location on
        on the spline (u) and the maximum curvature value
        """
        result = minimize(
            lambda u: -self.curvature_at(u[0])**2,
            0.5,
            bounds=[(0., 1)],
            method="SLSQP"
            )
        if not result.success:
            print("Failed to find max curvature!")
        return result.x[0], np.sqrt(-result.fun) if result.success else float("nan")

    @property
    def crest(self) -> Tuple[float, np.ndarray[float, float]]:
        """Return lowest point of the airfoil.
        Based on the PARSEC parameter.

        Raises:
            Exception: Finding crest failed.

        Returns:
            Tuple[float, np.ndarray]: [u, [x, y]]: crest location and coordinates.
        """
        result = minimize(lambda u: -self.evaluate_at(u[0])[1]**2,
                0.5,
                bounds=[(0, 1)]
                )

        if result.success:
            return result.x[0], self.evaluate_at(result.x[0])
        else:
            raise Exception("Finding lower crest failed: " + result.message)

    @property
    def crest_curvature(self) -> float:
        """Return the curvature at the crest."""
        return self.curvature_at(self.crest[0])

    def find_u(self, x: float = None, y: float = None) -> float:
        """Find the parametric value ``u`` for a given point ``(x, y)``."""
        if x is None and y is None:
            raise ValueError("At least one of x or y should be provided!")

        result = minimize(
            lambda u: (
                lambda a: np.linalg.norm(a - np.array([
                    x if x is not None else a[0],
                    y if y is not None else a[1]
                ]))
            )(self.evaluate_at(u[0])),
            0.5,
            bounds=[(0., 1)],
            method="SLSQP"
        )

        if result.success:
            return result.x[0]
        else:
            print("Failed to find u!")
            return float("nan")


# TODO split in upper/lower surface and merged surface classes
class CompositeBezierBspline(BSpline2D):
    """Creates a composite B-spline representation of a set of points or a set
    of ClampedBezierCurves.
    Bsplines are clamped in the first, last, and middle control points to
    function as Bezier curves.

    Meant to define the entire surface of an airfoil.

    Args:
        points: A set of 2D row-vectors
        n_control_points: Number of control points (n) per spline segment (upper and
                          lower surfaces). Defaults to 6.
        control_point_spacing: Either Spacing between control points when location is
                                fixed but not predefined (currently supports
                                "cosine" and "linear" spacing), or an array of
                                control points x-ordinates.
                                Defaults to "Cosine" spacing.
    """
    def __init__(
            self, points: np.ndarray,
            degree: Optional[int] = None,
            n_control_points: Optional[int] = None,
            control_point_spacing: Union[str, np.ndarray] = None,
        ):

        if not n_control_points and not degree and not isinstance(control_point_spacing, np.ndarray):
            raise ValueError(
                "Either degree or n_control_points must be provided."
                )

        elif degree and n_control_points:
            if degree != n_control_points - 1:
                raise ValueError(
                    "If both degree and n_control_points are provided, they must satisfy the relation degree = n_control_points - 1."
                    )

        if isinstance(control_point_spacing, np.ndarray):
            if len(control_point_spacing) != self.n_control_points:
                raise ValueError(
                    "When specifying control point x-ordinates, Length of control_point_spacing must be equal to n_control_points."
                    )
            if control_point_spacing[0] != 0 or control_point_spacing[-1] != 1:
                raise ValueError(
                    "When specifying control point x-ordinates, the first and last control points must be located at 0 and 1, respectively."
                    )

        self.points = points
        self._n_control_points = n_control_points
        self._degree = degree
        self.control_point_spacing = control_point_spacing

    @cached_property
    def degree(self) -> int:
        """
        If degree is specified, return it.
        Otherwise, calculate the degree of the spline based on the number of
        control points per segment (n). The degree is the number of control
        points per segment minus 1: p =(n-1).

        Returns:
            int: spline degree per segment (uppper/lower surface)
        """
        return self._degree if self._degree else self.n_control_points - 1

    @cached_property
    def n_control_points(self) -> int:
        """
        If number of control points (n) is specified, return it.
        Otherwise, calculate the number of control points based on the specified
        spline degree (p). The number of control points is the degree of the
        spline plus 1: n = (p+1).

        Returns:
            int: number of control points per segment (uppper/lower surface)
        """
        if self._n_control_points:
            return self._n_control_points
        elif isinstance(self.control_point_spacing, np.ndarray):
            return len(self.control_point_spacing)
        else:
            return self.degree + 1

    @cached_property
    def n_knots(self):
        """
        Returns:
            int: number of knots required to define the spline.
        """
        return self.n_control_points + self.degree + 1

    @cached_property
    def x_control_points(self) -> np.ndarray:
        """
        Return the x-coordinates of the control points.
        - If control point spacing is specified as an array (with control point
            x-ordinates), return it.
        - If control point spacing is specified as a string, return the control
            points x-ordinates based on the specified spacing.
        - If control point spacing is not specified, raise a ValueError.

        Raises:
            ValueError: Missing control point spacing specification.

        Returns:
            np.ndarray: x-coordinates of the control points.
        """
        if isinstance(self.control_point_spacing, np.ndarray):
            return self.control_point_spacing

        elif isinstance(self.control_point_spacing, str):
            match self.control_point_spacing:
                case "cosine":
                    return np.insert(cosine_spacing(0, 1, self.n_control_points), 0, 0)
                case "linear":
                    return np.insert(np.linspace(0, 1, self.n_control_points), 0, 0)

        else:
            raise ValueError("Missing control point spacing specification.")

    @cached_property
    def y_control(self):
        """Calculate the y-coordinates of the upper surface control points."""
        def cost_function(y_control_points):
            y_control_points = np.insert(y_control_points, len(y_control_points)//2, 0)
            y_control_points = np.insert(y_control_points, 0, 0.001)
            y_control_points = np.insert(y_control_points, len(y_control_points), 0.001)
            x_control_points = np.append(self.x_control_points[::-1], self.x_control_points[1:])
            control_points = np.column_stack((x_control_points, y_control_points))
            tck = np.array([self.combined_knots, control_points.T, self.degree])
            u_fine = np.linspace(0, 1, 400)
            sample = np.array(si.splev(u_fine, tck)).T
            target = np.array(si.splev(u_fine, super().spline)).T
            return np.sum((np.linalg.norm(sample-target)*100) ** 2)
            # tree = KDTree(sample)
            # distances, _ = tree.query(target)
            # return np.sum(distances**2)

        result = minimize(
            cost_function,
            np.hstack((
                [1]*(self.n_control_points-1),
                [-1]*(self.n_control_points-1)
            )),
            method='L-BFGS-B'
        )
        return result.x


    @cached_property
    def combined_control_points(self):
        """Combine the upper and lower surface control points."""
        x = np.concatenate(
            (
                cosine_spacing(1, 0, self.n_control_points - 1),
                [0],
                cosine_spacing(1, 0, self.n_control_points - 1)[1:],
            )
        )
        y = np.concatenate(
            (
                [0.001],
                self.y_control[:len(self.y_control)//2:-1],
                [0],
                self.y_control[len(self.y_control)//2:],
                [0.001]
            )
        )
        return np.column_stack((x, y))

    @cached_property
    def knot_value(self):
        """Calculate the knot value for the combined spline.
        This should be the leading edge location of the airfoil."""
        distances = np.sqrt(np.sum(np.diff(self.combined_control_points, axis=0)**2, axis=1))  # Compute distances between control points# Compute distances between control points
        parameters = np.concatenate(([0], np.cumsum(distances) / np.sum(distances)))           # Compute parameter values proportional to distances
        return parameters[len(parameters)//2]                                                  # Compute knot value at knot point (middle of parameters)

    @cached_property
    def combined_knots(self) -> np.ndarray:
        """Create a single, continuous knot vector for the combined spline.

        Number of knot points should be number of control points + degree + 1
        (n+p+1). The degree of the spline is the number of control points per
        segment minus 1: p =(n-1). There must be at least one more distinct knot
        value than the number of spline segments (in this case 2: upper and
        lower surface).

        To clamp the endpoints, the first and last knot point values must repeat
        p+1 times, or in other words the first and last p+1 knot values must be
        0 and 1, respectively. The knot point connecting the two curves should
        repeat p times for C2 continuity.

        returns:
            np.ndarray: combined knot vector
        """
        return np.concatenate((
            [0]*(self.degree+1),            # clamp the first control point (TE)
            [self.knot_value]*self.degree,  # clamp the middle control point(LE)
            [1]*(self.degree+1)             # clamp the last control point (TE)
            ))

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
        """
        return np.array([self.combined_knots, self.combined_control_points.T, self.degree])



class ClampedBezierCurve(BSpline2D):
    """
    Creates a B-spline definition which function as a Bezier curve by
    clamping the first and last control points.

    Meant to define the upper or lower surface of an airfoil.

    Args:
        points: A set of 2D row-vectors
        n_control_points: Number of control points (n) per spline segment (upper and
                          lower surfaces). Defaults to 6.
        control_point_spacing: Spacing between control points when location is
                                fixed but not predefined. Defaults to None.
        trailing_edge_gap: Trailing edge thickness.
        surface_type: 'upper' or 'lower' to specify the surface.
    """
    def __init__(
            self, points: np.ndarray,
            degree: Optional[int] = None,
            n_control_points: Optional[int] = None,
            control_point_spacing: Union[str, np.ndarray] = None,
            control_point_values: Optional[np.ndarray] = None,
            trailing_edge_gap: Optional[float] = 0.0,
            surface_type: str = 'upper'
        ):

        if not n_control_points and not degree and not isinstance(control_point_spacing, np.ndarray):
            raise ValueError(
                "Either degree or n_control_points must be provided."
                )
        elif degree and n_control_points:
            if degree != n_control_points - 1:
                raise ValueError(
                    "If both degree and n_control_points are provided, they must satisfy the relation degree = n_control_points - 1."
                    )

        if isinstance(control_point_spacing, np.ndarray):
            if len(control_point_spacing) != n_control_points:
                raise ValueError(
                    "When specifying control point x-ordinates, Length of control_point_spacing must be equal to n_control_points."
                    )
            if control_point_spacing[0] != 0 or control_point_spacing[-1] != 1:
                raise ValueError(
                    "When specifying control point x-ordinates, the first and last control points must be located at 0 and 1, respectively."
                    )
            if isinstance(control_point_values, np.ndarray):
                if len(control_point_values) != n_control_points:
                    raise ValueError(
                        "When specifying control point y-ordinates, Length of control_point_values must be equal to n_control_points."
                        )

        self.points = points
        self._n_control_points = n_control_points
        self._degree = degree
        self.control_point_spacing = control_point_spacing
        self.control_point_values = control_point_values
        self.trailing_edge_gap = trailing_edge_gap
        self.surface_type = surface_type

    @cached_property
    def degree(self) -> int:
        """
        If degree is specified, return it.
        Otherwise, calculate the degree of the spline based on the number of
        control points per segment (n). The degree is the number of control
        points per segment minus 1: p =(n-1).

        Returns:
            int: spline degree per segment (uppper/lower surface)
        """
        return self._degree if self._degree else self.n_control_points - 1

    @cached_property
    def n_control_points(self) -> int:
        """
        If number of control points (n) is specified, return it.
        Otherwise, calculate the number of control points based on the specified
        spline degree (p). The number of control points is the degree of the
        spline plus 1: n = (p+1).

        Returns:
            int: number of control points per segment (uppper/lower surface)
        """
        if self._n_control_points:
            return self._n_control_points
        elif isinstance(self.control_point_spacing, np.ndarray):
            return len(self.control_point_spacing)
        else:
            return self.degree + 1

    @cached_property
    def n_knots(self) -> int:
        """
        Returns:
            int: number of knots required to define the spline.
        """
        return self.n_control_points + self.degree + 1

    @cached_property
    def x_control_points(self) -> np.ndarray:
        """
        Return the x-coordinates of the control points.
        - If control point spacing is specified as an array (with control point
            x-ordinates), return it.
        - If control point spacing is specified as a string, return the control
            points x-ordinates based on the specified spacing.
        - If control point spacing is not specified, raise a ValueError.
        """
        if isinstance(self.control_point_spacing, np.ndarray):
            return self.control_point_spacing  # Not including leading and trailing points

        elif isinstance(self.control_point_spacing, str):
            match self.control_point_spacing:
                case "cosine":
                    return cosine_spacing(0, 1, self.n_control_points - 1)  # Using total n_control_points minus the manual additions
                case "linear":
                    return np.linspace(0, 1, self.n_control_points - 1)

        else:
            raise ValueError("Missing control point spacing specification.")

    @cached_property
    def control_points(self):
        """Calculate the y-coordinates of the control points including constraints."""
        def objective_function(y_values):
            """ Compute the sum of squared distances from airfoil points to the spline. """
            control_points = np.concatenate((
                np.array([[0,0]]),
                np.column_stack((self.x_control_points[:-1], y_values)),
                np.array([[1,0]])
            ))
            tck = (self.knots, control_points.T, self.degree)
            u = np.linspace(0, 1, 400)
            curve_points = np.array(si.splev(u, tck)).T
            tree = KDTree(curve_points)
            distances, _ = tree.query(self.points)
            return np.sum(distances**2)

        result = minimize(
            objective_function,
            np.array([0.5] * (self.n_control_points - 2)),
            method='SLSQP', #'L-BFGS-B',
            options={'maxiter': 1e6, 'ftol': 1e-9}
        )

        return np.concatenate((
                np.array([[0,0]]),
                np.column_stack((self.x_control_points[:-1], result.x)),
                np.array([[1,0]])
            ))


    @cached_property
    def knots(self):
        """Create a single, continuous knot vector for the combined spline.

        Number of knot points should be number of control points + degree + 1
        (n+p+1). The degree of the spline is the number of control points per
        segment minus 1: p =(n-1). There must be at least one more distinct knot
        value than the number of spline segments (in this case 2: upper and
        lower surface).

        To clamp the endpoints, the first and last knot point values must repeat
        p+1 times, or in other words the first and last p+1 knot values must be
        0 and 1, respectively. The knot point connecting the two curves should
        repeat p times for C2 continuity.
        """
        return np.concatenate(([0] * (self.degree + 1), [1] * (self.degree + 1)))

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
        """
        return (self.knots, self.control_points.T, self.degree)
