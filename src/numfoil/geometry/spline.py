"""Contains py:class:`BSpline` for splining 2D points."""

from functools import cached_property
from typing import Optional, Union, Tuple

import numpy as np
import scipy.interpolate as si
import scipy.optimize as opt

from scipy.spatial import KDTree

from .geom2d import normalize_2d, rotate_2d_90ccw
from ..util import cosine_spacing, chebyshev_nodes

# TODO add CST based spline class

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
        return (ddy * dx - ddx * dy) / (dx**2 + dy**2)**1.5

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
        result = opt.minimize(
            lambda u: -self.curvature_at(u[0])**2,
            0.51,
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
        result = opt.minimize(lambda u: -self.evaluate_at(u[0])[1]**2,
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

        result = opt.minimize(
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


# TODO: implement this with proper bezier curve defintion instead of splev(?)
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
            # degree: Optional[int] = None,
            n_control_points: Optional[int] = 8,
            control_point_spacing: Union[str, np.ndarray] = 'cosine',
            control_point_values: Optional[np.ndarray] = None,
        ):

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
        self.control_point_spacing = control_point_spacing
        self.control_point_values = control_point_values

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
        # return self._degree if self._degree else self.n_control_points - 1
        return self.n_control_points - 1

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
        Return the chordwise locations (x-coordinates) of the control points.
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
                    return cosine_spacing(0, 1, self.n_control_points - 2)  # Using total n_control_points minus the manual additions
                case "linear":
                    return np.linspace(0, 1, self.n_control_points - 2)
                case "chebyshev":
                    return chebyshev_nodes(0, 1, self.n_control_points - 2)

        else:
            raise ValueError("Missing control point spacing specification.")

    def combine_control_points(self, y_values: np.ndarray) -> np.ndarray:
        """Combine the x and y control points."""
        return np.concatenate((
                np.array([[0,0]]),
                # np.column_stack((self.x_control_points[:-1], y_values)), # #this is to force the trailling edge to y=0
                # np.array([[1,0.0005*np.sign(np.median(self.points.T[1]))]])
                np.column_stack((self.x_control_points, y_values)), # this sets y at trailing edge to fit coordinates
                np.array([[1,0]]) # this closes the trailing edge with a curve at the end (requires n-2 isntead of n-1)
            ))

    @cached_property
    def control_points(self):
        """Calculate the y-coordinates of the control points including constraints."""
        def objective_function(y_values):
            """ Compute the sum of squared distances from airfoil points to the spline. """
            control_points = np.concatenate((
                np.array([[0,0]]),
                # np.column_stack((self.x_control_points[:-1], y_values)), # #this is to force the trailling edge to y=0
                # np.array([[1,0.0005*np.sign(np.median(self.points.T[1]))]])
                np.column_stack((self.x_control_points, y_values)), # this sets y at trailing edge to fit coordinates
                # np.array([[1,0]]) # this closes the trailing edge with a curve at the end (requires n-2 isntead of n-1)
                np.array([[1,0.0005*np.sign(np.median(self.points.T[1]))]])
            ))
            tck = (self.knots, control_points.T, self.degree)
            u = cosine_spacing(0, 1, 400)
            curve_points = np.array(si.splev(u, tck)).T
            distances, _ = KDTree(curve_points).query(self.points)
            # distances = np.min(np.linalg.norm(curve_points[:, None, :] - self.points[None, :, :], axis=2), axis=0)
            # target_points = BSpline2D(self.points).evaluate_at(u)
            # distances = np.min(np.linalg.norm(curve_points[:, None, :] - target_points[None, :, :], axis=2), axis=0)
            return np.sum(distances)**2

        sign = np.sign(np.median(self.points.T[1]))
        bounds = [(None, None)] * (self.n_control_points - 2)
        bounds[0] = bounds[-1] = bounds[-2] = (0.0005*sign, None) if sign >= 0 else (None, 0.0005*sign)

        result = opt.minimize(
            objective_function,
            # np.array([np.mean(self.points)] * (self.n_control_points - 1)),
            self.points.T[1][np.linspace(5, len(self.points)-2, num=self.n_control_points-2, dtype=int)],
            method='SLSQP', #'L-BFGS-B',
            options={'maxiter': 1e6, 'ftol': 1e-12},
            bounds=bounds
        )

        return np.concatenate((
                np.array([[0,0]]), # always there
                # np.column_stack((self.x_control_points[:-1], result.x)), # this is to force the trailling edge to y=0
                np.column_stack((self.x_control_points, result.x)), # this one to have 2 contorl points at x=1 to close the trailing edge with a curve
                # np.array([[1,0]])
                np.array([[1,0.0005*np.sign(np.median(self.points.T[1]))]])
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
        # return np.concatenate(([0] * (self.degree + 1), [1] * (self.degree + 1)))
        # Create a uniform knot vector in the middle
        middle_knots = np.linspace(0, 1, self.n_knots - 2 * (self.degree + 1))

        # Repeat the first and last knot values to create a clamped knot vector
        knot_vector = np.concatenate(([0] * (self.degree + 1), middle_knots, [1] * (self.degree + 1)))

        return knot_vector

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
        """
        return (self.knots, self.control_points.T, self.degree)

    # TODO: This is currently unused, but could be useful later. idk if it is
    # TODO  worth replacing the current use of splev with this instead to get a
    # TODO  more proper bezier curve definition.
    def bezier_curve_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Bezier curve evaluation of the spline at location u.

        Args:
            u (float or array): parametric location on the spline

        Returns:
            np.ndarray: Bezier curve points
        """
        if not isinstance(u, np.ndarray):
            u = np.array([u])
        curve_points = np.zeros((len(u), 2))
        for i in range(self.degree + 1):
            binomial_coeff = np.math.factorial(self.degree) / (np.math.factorial(i) * np.math.factorial(self.degree - i))
            curve_points[:, 0] += binomial_coeff * (u ** i) * ((1 - u) ** (self.degree - i)) * self.control_points[i, 0]
            curve_points[:, 1] += binomial_coeff * (u ** i) * ((1 - u) ** (self.degree - i)) * self.control_points[i, 1]
        return curve_points


class CompositeBezierBspline(BSpline2D):
    """Creates a composite B-spline representation of an airfoil from the knot
    points of the upper and lower surface"""
    def __init__(
            self,
            control_points: np.ndarray,
            u_leading_edge: float,
        ):
        self.control_points = control_points
        self.knot_value = u_leading_edge

    @cached_property
    def degree(self) -> int:
        return self.n_control_points - 1

    @cached_property
    def n_control_points(self) -> int:
        """
        Returns:
            int: number of control points per segment (upper/lower surface)
        """
        return len(self.control_points)//2+1

    @cached_property
    def n_knots(self):
        """
        Returns:
            int: number of knots required to define the spline.
        """
        return self.n_control_points + self.degree + 1


    @cached_property
    def knot_value(self):
        """Calculate the knot value for the combined spline.
        This should be the leading edge location of the airfoil."""
        distances = np.sqrt(np.sum(np.diff(self.control_points, axis=0)**2, axis=1))  # Compute distances between control points# Compute distances between control points
        parameters = np.concatenate(([0], np.cumsum(distances) / np.sum(distances)))           # Compute parameter values proportional to distances
        return parameters[len(parameters)//2]                                                  # Compute knot value at knot point (middle of parameters)

    @cached_property
    def knots(self) -> np.ndarray:
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
        return [self.knots, self.control_points.T, self.degree]
