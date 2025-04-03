import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as si
import scipy.optimize as opt

from functools import cached_property
from typing import Union, Tuple
from abc import ABCMeta, abstractmethod

from .data import AirfoilDataFile, AirfoilNormalizer
from ..util import cosine_spacing, chebyshev_nodes, ensure_1d_vector
from .spline import *
from .geom2d import Point2D


class AirfoilBase(metaclass=ABCMeta):
    """Abstract Base Class definition of an :py:class:`Airfoil`.
    ...
    """

    @abstractmethod
    def surface(self):
        """Returns a parametric surface curve representing the entire airfoil"""

    @abstractmethod
    def upper_surface(self):
        """Returns the upper surface curve"""

    @abstractmethod
    def lower_surface(self):
        """Returns the lower surface curve"""

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

    @abstractmethod
    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line points at the supplied ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """

    @property
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered."""
        raise NotImplementedError

    def plot(self):
        """Plots the airfoil geometry."""
        raise NotImplementedError


class BezierAirfoil(AirfoilBase):
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
        datafile = AirfoilDataFile(filepath)

        # normalized_points = AirfoilProcessor.normalize(datafile.points)

        return cls(
            data_points=datafile.points,
            name=datafile.filename,
            full_name=datafile.header,
        )

    # TODO add option for other curve types
    @cached_property
    def surface(self) -> ParametricCurve:
        """Construct the surface spline of the airfoil."""
        return SplevCBezier(self.normalized_points)

    @property
    def cambered(self) -> bool:
        return True if self.max_camber[1] > 0 else False


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

    @property
    def upper_surface(self):
        raise NotImplementedError

    @property
    def lower_surface(self):
        raise NotImplementedError


class CSTAirfoil(AirfoilBase):
    def __init__(
        self,
        data_points: np.ndarray,
        name: str = None,
        full_name: str = None,
        normalize: bool = True,
    ):
        # the original input points, mainly for reference
        self.data_points = data_points
        # the data points after processing, used for fitting
        self.surface_points = AirfoilProcessor.normalize(data_points) if normalize else data_points

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
        datafile = AirfoilDataFile(filepath)
        return cls(
            data_points=datafile.points,
            name=datafile.filename,
            full_name=datafile.header,
        )

    # TODO add option for other curve types
    @cached_property
    def surface(self) -> CSTAirfoilSurface:
        """Construct the surface spline of the airfoil."""
        return CSTAirfoilSurface.fit(self.surface_points)

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
    def upper_surface(self) -> CSTCurve:
        return self.surface.upper_part

    @cached_property
    def lower_surface(self) -> CSTCurve:
        return self.surface.lower_part

    @cached_property
    def upper_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for upper surface spline.
        Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: upper surface y ordinate at x.
        """
        return self.upper_surface.evaluate_at

    @cached_property
    def lower_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for upper surface spline.
        Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: upper surface y ordinate at x.
        """
        return self.lower_surface.evaluate_at

    @cached_property
    def camber_line(self) -> si.PchipInterpolator:
        """Returns the interpolator for the camber line."""
        x = np.linspace(0, 1, 500)
        y_upper = self.upper_surface(x)
        y_lower = self.lower_surface(x)
        camber = 0.5 * (y_upper + y_lower)
        return CSTCurve.fit(camber, num_coefficients=6, n1=1.0, n2=1.0,)

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber at specified x locations."""
        return self.camber_line(x)

    @property
    def cambered(self) -> bool:
        return True if self.max_camber[1] > 0 else False

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


class Airfoil(AirfoilBase):
    def __init__(
        self,
        data_points: np.ndarray,
        surface: Curve = None,
        upper_surface: Curve = None,
        lower_surface: Curve = None,
        name: str = None,
        full_name: str = None,
    ):
        # the original input points, only for reference
        self.data_points = data_points

        # curve definition of the airfoil geometry
        self.surface = surface              # the entire airfoil in a single curve
        self.upper_surface = upper_surface  # the upper surface curve
        self.lower_surface = lower_surface  # the lower surface curve

        # the shortened name of the airfoil, usually the filename
        self.name = name
        # the full name of the airfoil, usually from the file header
        self.full_name = full_name or name


    @classmethod
    def from_array(
        cls,
        points: np.ndarray,
        curve_type: str = "bezier",
        name: str = None,
    ):
        spline = AirfoilNormalizer.normalize(points)
        upper_points = spline.evaluate_at(
            cosine_spacing(0, spline.u_leading_edge, num=100)
        )

        match curve_type:
            case "bezier":
                surface = SplevBezier(spline)
            case "cst":
                surface = CSTCurve.fit(spline)
            case _:
                raise ValueError(f"Unknown curve type: {curve_type}")

    @classmethod
    def from_file(
        cls,
        filepath: str,

    ):
        """Returns an Airfoil object from a data file.
        Coordinates are normalized before passing to the Airfoil object.

        Args:
            filepath (str): path to the data file.

        Returns:
            Airfoil: Airfoil object initialized with data from file.
        """
        datafile = AirfoilDataFile(filepath)

    @cached_property
    def surface(self) -> ParametricCurve:
        return self.surface_class.fit(self.data_points)

    @cached_property
    def upper_surface(self) -> CSTCurve:
        return CSTCurve.fit(self.data_points[:len(self.data_points) // 2][::-1])

    @cached_property
    def lower_surface(self) -> CSTCurve:
        return CSTCurve.fit(self.data_points[len(self.data_points) // 2:])

    @cached_property
    def upper_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for upper surface spline.
        Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: upper surface y ordinate at x.
        """
        return self.upper_surface.evaluate_at

    @cached_property
    def lower_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for upper surface spline.
        Returns upper airfoil ordinates"""


class NACA4Airfoil(Airfoil):
    def __init__(self):
        pass
