import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as si
import scipy.optimize as opt

from functools import cached_property
from typing import Union, Tuple
from abc import ABCMeta, abstractmethod

from .data import AirfoilProcessor, AirfoilDataFile
from ..util import cosine_spacing, chebyshev_nodes, ensure_1d_vector
from .spline_v2 import BSpline2D, SplevCBezier, ParametricCurve
from .geom2d import Point2D


class AirfoilBase(metaclass=ABCMeta):
    """Abstract Base Class definition of an :py:class:`Airfoil`.
    ...
    """

    @abstractmethod
    @cached_property
    def data_points(self) -> np.ndarray:
        """Returns the original input data points used to define the airfoil."""

    @property
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered."""
        raise NotImplementedError

    @abstractmethod
    def camberline_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line points at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """

    @abstractmethod
    def upper_surface_at(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)

        """

    @abstractmethod
    def lower_surface_at(
        self, x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower airfoil ordinates at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """

    def plot(self):
        """Plots the airfoil geometry."""
        raise NotImplementedError


class Airfoil(AirfoilBase):
    """Unified airfoil class handling points-based airfoils with inconsistent or
    missing data."""

    def __init__(
        self,
        data_points: np.ndarray,
        normalized_points: np.ndarray = None,
        name: str = None,
        full_name: str = None,
    ):
        # the original input points, mainly for reference
        self.data_points = data_points
        # the data points after processing, used for fitting
        self.normalized_points = normalized_points or AirfoilProcessor.normalize(data_points)

        # the shortened name of the airfoil, usually the filename
        self.name = name
        # the full name of the airfoil, usually from the file header
        self.full_name = full_name or name

    @classmethod
    def from_array(cls, points: np.ndarray, name: str = None):
        """Creates an Airfoil object from an array of points.
        Mainly used for input validation.

        Args:
            points (np.ndarray): Array of airfoil points.

        Raises:
            TypeError: Input must be a numpy array.
            ValueError: Invalid shape for input array.

        Returns:
            Airfoil: Airfoil object initialized with input
        """
        if not isinstance(points, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if points.shape[1] != 2:
            raise ValueError("Input array must have shape (n, 2).")
        # normalized_points = AirfoilProcessor.normalize(points)
        return cls(
            data_points=points,
            name=name
        )

    @classmethod
    def from_file(cls, filepath: str):
        """Returns an Airfoil object from a data file.
        Coordinates are normalized before passing to the Airfoil object.

        Includes setting the name and full name of the airfoil based on the
        filename and header, respectively.

        Args:
            filepath (str): path to the data file.

        Returns:
            Airfoil: Airfoil object initialized with data from file.
        """
        if AirfoilDataFile.has_header(filepath):
            # read the header and points separately
            header = AirfoilDataFile.header(filepath)
            points = AirfoilDataFile.load_from_file(filepath, header=1)
        else:
            # no header, just read the points
            header = None
            points = AirfoilDataFile.load_from_file(filepath, header=0)

        filename = AirfoilDataFile.filename(filepath)
        points = np.genfromtxt(filepath, skip_header=1).view(Point2D)

        # ! normalization call moved to __init__ (for now)
        # normalized_points = AirfoilProcessor.normalize(points)

        return cls(
            data_points=points,
            name=filename,
            full_name=header
        )

    # TODO add option for other curve types
    @cached_property
    def surface(self) -> ParametricCurve:
        """Construct the surface spline of the airfoil."""
        return SplevCBezier(self.normalized_points)

    @cached_property
    def trailing_edge(self) -> np.ndarray:
        """Returns the [x,y] coordinate of the trailing edge.

        Trailing edge is taken as the midpoint between surface spline ends.

        Returns:
            np.ndarray: The [x,y] coordinate of the trailing edge.
        """
        return 0.5*(self.surface.evaluate_at(0) + self.surface.evaluate_at(1))


    @cached_property
    def u_leading_edge(self) -> np.ndarray:
        """Determines the leading edge as the point on the spline furthest from
        the trailing edge.
        """
        result = opt.minimize(
            lambda u: -np.linalg.norm(
                self.trailing_edge - self.surface.evaluate_at(u[0])
            ),
            0.5, # initial guess
            bounds=[(0, 1)],
            # method="SLSQP",
            )

        if not result.success:
            print(result)
            raise RuntimeError("Failed to find leading edge.")
        return result.x[0]

    @cached_property
    def leading_edge(self) -> np.ndarray:
        """Determines the leading edge as the point on the spline furthest from the trailing edge."""
        return self.surface.evaluate_at(self.u_leading_edge)

    @cached_property
    def upper_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for upper surface spline.
        Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: upper surface y ordinate at x.
        """
        points = self.surface.evaluate_at(
            cosine_spacing(0, self.u_leading_edge, num=500)
        ).round(5)
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=False)

    @cached_property
    def lower_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for lower surface spline.
        Returns lower airfoil ordinates at the supplied ``x``.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: lower surface y ordinate at x.
        """
        points = self.surface.evaluate_at(
            cosine_spacing(self.u_leading_edge, 1, num=500)
        )[::-1].round(5)
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=False)

    @cached_property
    def camber_line(self) -> si.PchipInterpolator:
        """Returns the interpolator for the camber line."""
        x = np.linspace(0, 1, 200)
        y_upper = self.upper_surface(x)
        y_lower = self.lower_surface(x)
        y_camber = 0.5 * (y_upper + y_lower)
        return si.PchipInterpolator(x, y_camber, extrapolate=False)

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber at specified x locations."""
        return self.camber_line(x)

    def thickness_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Calculates the thickness at specified x locations."""
        y_upper = self.upper_surface(x)
        y_lower = self.lower_surface(x)
        return y_upper - y_lower

    def plot(self, n_points: int = 200, show: bool = True):
        """Plots the airfoil geometry."""
        x = np.linspace(0, 1, n_points)
        y_upper = self.upper_surface(x)
        y_lower = self.lower_surface(x)
        y_camber = self.camber_line(x)

        fig, ax = plt.subplots()
        ax.plot(x, y_upper, label="Upper Surface")
        ax.plot(x, y_lower, label="Lower Surface")
        ax.plot(x, y_camber, label="Camber Line", linestyle="--")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        if show:
            plt.show()

        return fig, ax

    @property
    def max_thickness(self) -> Tuple[float, float]:
        """Finds the location and value of maximum thickness."""
        result = opt.minimize(
            lambda x: -self.thickness_at(x), 0.5, bounds=[(0, 1)]
        )
        if result.success:
            return result.x[0], -result.fun
        else:
            raise RuntimeError("Failed to find maximum thickness.")

    @property
    def max_camber(self) -> Tuple[float, float]:
        """Finds the location and value of maximum camber."""
        result = opt.minimize(lambda x: -self.camber_at(x), 0.5, bounds=[(0, 1)])
        if result.success:
            return result.x[0], -result.fun
        else:
            raise RuntimeError("Failed to find maximum camber.")
