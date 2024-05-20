"""Contains py:class:`BSpline` for splining 2D points."""

from functools import cached_property
from typing import Optional, Union

import numpy as np
import scipy.interpolate as si
import scipy.optimize as opt

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
        self.degree = degree
        self.smoothing = smoothing

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
                [1]: Parametric points, u, used to create the spline
        """
        return si.splprep(self.points.T, s=self.smoothing, k=self.degree)

    # @cached_property
    # def exact_interpolator(self):
    #     pts = self.evaluate_at(np.linspace(0,1,num=200))
    #     return si.pchip_interpolate(pts[:,0], pts[:,1], self.x, der=0, axis=0)

    def evaluate_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline point(s) at ``u``."""
        return np.array(si.splev(u, self.spline[0], der=0), dtype=np.float64).T

    def first_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        return np.array(si.splev(u, self.spline[0], der=1), dtype=np.float64).T

    def second_deriv_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        return np.array(si.splev(u, self.spline[0], der=2), dtype=np.float64).T

    def tangent_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the spline tangent(s) at ``u``."""
        # return normalize_2d(
        #     np.array(si.splev(u, self.spline[0], der=1), dtype=np.float64).T
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

    @cached_property
    def max_curvature(self) -> float:
        """Finds maximum curvature of the spline
        Returns:
            Tuple[float, np.float]: [u, curvature]: max curvature location on
            the spline (u) and the maximum curvature value
        """
        result = opt.minimize(
            lambda u: -self.curvature_at(u[0])**2,
            0.5,
            bounds=[(0., 1)],
            method="SLSQP"
            )
        if not result.success:
            print("Failed to find max curvature!")
        return result.x[0], np.sqrt(-result.fun) if result.success else float("nan")


class CompositeBezierBspline(BSpline2D):
    """Creates a composite B-spline representation of a set of points.
    Bsplines are clamped in the first, last, and middle control points to
    function as Bezier curves.

    Args:
        points: A set of 2D row-vectors
        n_control_points: Number of control points per spline segment (upper and
                          lower surfaces). Defaults to 6.
        control_point_spaceing: Spacing between control points when location is
                                fixed but not predefined. Defaults to None.
    """

    def __init__(
            self, points: np.ndarray,
            n_control_points: Optional[int] = 6,
            control_point_spaceing: Union[str, np.ndarray] = None,
            x_control_points: Optional[np.ndarray] = None,
        ):
        self.points = points
        self._n_control_points = n_control_points
        self._x_control_points = x_control_points

    @cached_property
    def x_control_points(self):
        if self._x_control_points:
            return self._x_control_points
        elif self.control_point_spaceing:
            match self.control_point_spaceing:
                case "cosine":
                    return cosine_spacing(0, 1, self.n_control_points)
                case "linear":
                    return np.linspace(0, 1, self.n_control_points)
        else:
            return np.linspace(0, 1, self.n_control_points)


    @cached_property
    def degree(self):
        return self.n_control_points - 1

    @cached_property
    def n_control_points(self):
        if self._n_control_points:
            return self._n_control_points
        else:
            return self.degree + 1

    @cached_property
    def n_knots(self):
        return self.n_control_points + self.degree + 1

    @cached_property
    def combined_control_points(self):
        """Combine the upper and lower surface control points."""
        return np.concatenate(
                (
                    self.control_points_upper,
                    self.control_points_lower[1:]
                )
            )

    @cached_property
    def knot_value(self):
        """Calculate the knot value for the combined spline.
        This should be the leading edge location of the airfoil."""
        distances = np.sqrt(np.sum(np.diff(self.combined_control_points, axis=0)**2, axis=1))  # Compute distances between control points# Compute distances between control points
        parameters = np.concatenate(([0], np.cumsum(distances) / np.sum(distances)))           # Compute parameter values proportional to distances
        return parameters[len(parameters)//2]                                                  # Compute knot value at knot point (middle of parameters)

    @cached_property
    def knots(self):
        """Create a single, continuous knot vector for the combined spline.

        Number of knot points should be number of control points + degree + 1
        (n+p+1). The degree of the spline is the number of control points per
        segment minus 1 (p-1). There must be at least one more distinct knot
        value than the number of spline segments (in this case 2: upper and
        lower surface).

        To clamp the endpoints, the first and last knot point values must repeat
        p+1 times, or in other words the first and last p+1 knot values must be
        0 and 1, respectively. The knot point connecting the two curves should
        repeat p times for C2 continuity.
        """
        return np.concatenate(([0]*(self.degree+1), [self.knot_value]*self.degree, [1]*(self.degree+1)))     # Create knot vector with knot value at knot point

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
        """
        return np.array([self.combined_knots, self.combined_control_points.T, self.degree])

