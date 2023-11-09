"""Contains py:class:`BSpline` for splining 2D points."""

from functools import cached_property
from typing import Optional, Union

import numpy as np
import scipy.interpolate as si
import scipy.optimize as opt

from .geom2d import normalize_2d, rotate_2d_90ccw


class BSpline2D:
    """Creates a splined representation of a set of points.

    Args:
        points: A set of 2D row-vectors
        degree: Degree of the spline. Defaults to 3 (cubic spline).
    """

    def __init__(self, points: np.ndarray, degree: Optional[int] = 3):
        self.points = points
        self.degree = degree

    @cached_property
    def spline(self):
        """1D spline representation of :py:attr:`points`.

        Returns:
            Scipy 1D spline representation:
                [0]: Tuple of knots, the B-spline coefficients, degree
                     of the spline.
                [1]: Parametric points, u, used to create the spline
        """
        return si.splprep(self.points.T, s=0.0, k=self.degree)

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
        dx, dy = self.first_deriv_at(u).T
        ddx, ddy = self.second_deriv_at(u).T
        return np.abs(ddy * dx - ddx * dy) / (dx**2 + dy**2)**1.5

    def radius_at(self, u: Union[float, np.ndarray]) -> np.ndarray:
        curvature = self.curvature_at(u)
        return 1 / curvature if curvature != 0 else np.inf

    @cached_property
    def max_curvature(self, bounds=[(0, 1)]) -> float:
        result = opt.minimize(lambda u: -self.curvature_at(u[0]), 0.5, bounds=bounds)
        if not result.success:
            print("Failed to find max curvature!")
        return result.x[0], self.curvature_at(result.x[0]) if result.success else float("nan")
