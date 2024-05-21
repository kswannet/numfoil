"""Contains py:class:`BSpline` for splining 2D points."""

from functools import cached_property
from typing import Optional, Union, Tuple

import numpy as np
import scipy.interpolate as si
from scipy.optimize import minimize

from .geom2d import normalize_2d, rotate_2d_90ccw


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
            print("At least one of x or y should be provided!")
            return float("nan")

        result = minimize(
            lambda u: np.linalg.norm(
                a := self.evaluate_at(u) - np.array([
                    x if x is not None else a[0],
                    y if y is not None else a[1]
                    ])
            ),
            0.5,
            bounds=[(0., 1)],
            method="SLSQP"
        )

        if result.success:
            return result.x[0]
        else:
            print("Failed to find u!")
            return float("nan")
