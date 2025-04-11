import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as si
import scipy.optimize as opt

from functools import cached_property
from typing import Union, Tuple
from abc import ABCMeta, abstractmethod,ABC

from .data import AirfoilDataFile, AirfoilNormalizer
from ..util import cosine_spacing, chebyshev_nodes, ensure_1d_vector, selig
from .spline import *
from .geom2d import Point2D


class AirfoilBase(ABC):
    """Abstract Base Class definition of an :py:class:`Airfoil`.
    ...
    """

    @property
    @abstractmethod
    def surface(self):
        """Returns a parametric surface curve representing the entire airfoil"""

    @property
    @abstractmethod
    def upper_surface(self):
        """Returns the upper surface curve"""

    @property
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
        surface_curve: ParametricCurve,
        name: str = None,
        description: str = None,
    ):
        # save the surface spline object
        self.surface_curve = surface_curve
        # the original input points, mainly for reference
        self.data_points = surface_curve.points
        # the shortened name of the airfoil, usually the filename
        self.name = name
        # the full name of the airfoil, usually from the file header
        self.description = description or name

    @classmethod
    def from_array(
        cls,
        points: np.ndarray,
        name: str = None,
        description: str = None,
        normalize: bool = True
    ):
        """Creates an Airfoil object from an array of points.

        Args:
            points (np.ndarray):
                Array of airfoil coordinate points.
            name (str):
                Name of the airfoil.
            description (str):
                Description of the airfoil. (Any additional text)
            normalize (bool):
                Whether to normalize the points before fitting.
                Defaults to True.

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


        # To improve the fitting of the new spline, points are resampled after
        # normalization
        if normalize:
            spline = AirfoilNormalizer.normalize(points) # returns normalized BSpline2D
            points = spline.evaluate_at(
                np.hstack([
                    cosine_spacing(0, spline.u_leading_edge, num=100),
                    cosine_spacing(spline.u_leading_edge, 1, num=100)[1:],
                ])
            )
        surfacespline = SplevCBezier.fit(points, 12, spacing='linear', verbose=False, w_damping=1e-3)
        surfacespline.points = spline.points  # add original points again for reference

        return cls(
            surfacespline,
            name=name,
            description=description
        )

    @classmethod
    def from_file(cls,
            filepath: str,
            normalize: bool = True
    ):
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

        return cls.from_array(
            datafile.points,
            name=datafile.filename,
            description=datafile.header,
            normalize=normalize,
        )

    @property
    def surface(self) -> ParametricCurve:
        """Returns a parametric surface curve representing the entire airfoil."""
        return self.surface_curve

    @cached_property
    def upper_surface(self):
        return SplevBezier(
            self.surface.control_points[self.surface.n_control_points//2::-1]
        )

    @cached_property
    def lower_surface(self):
        return SplevBezier(
            self.surface.control_points[self.surface.n_control_points//2:]
        )

    @cached_property
    def camber_line(self):
        """Returns the curve object for the camber line of the airfoil.

        The camber line curve is obtained by placing control points midway
        between the upper and lower control points of the airfoil surface.

        Previously this was done by evaluating the upper and lower surfaces,
        finding the midpoints defining the camber line, and fitting a curve
        through these points. Using the control points already available
        simplifies the process.

        Returns:
            ParametricCurve : The camber line object of the airfoil.
        """
        # y_upper = self.upper_surface_at(cosine_spacing(0, 1, 1000))
        # y_lower = self.lower_surface_at(cosine_spacing(0, 1, 1000))
        # y_camber = 0.5 * (y_upper + y_lower)
        # return SplevBezier.fit(
        #     np.column_stack([x, y_camber])
        # )
        camber_control_points = np.column_stack([
            self.upper_surface.control_points[:,0],
            0.5 * (
                self.upper_surface.control_points[:,1] +
                self.lower_surface.control_points[:,1]
                )
        ])
        return SplevBezier(camber_control_points)

    @cached_property
    def thickness_distribution(self):
        """Returns the thickness distribution of the airfoil."""
        # x = cosine_spacing(0, 1, 1000)
        # y_upper = self.upper_surface_at(x)
        # y_lower = self.lower_surface_at(x)
        # return y_upper - y_lower
        thickness_control_points = np.column_stack([
            self.upper_surface.control_points[:,0],
            (
                self.upper_surface.control_points[:,1] -
                self.lower_surface.control_points[:,1]
            )
        ])
        return SplevBezier(thickness_control_points)


    @cached_property
    def upper_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for upper surface spline.
        Returns upper airfoil ordinates at ``x``.

        Usage:
            `obj.upper_surface_at(x: Union[float, np.ndarray]) -> np.ndarray`

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: upper surface y ordinate at x.
        """
        points = self.upper_surface.evaluate_at(
            cosine_spacing(0, 1, num=2000)
        )[::-1]
        # filter out all points with non-increasing x-coordinates
        # this prevents a lot of headaches
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=False)

    @cached_property
    def lower_surface_at(self) -> si.PchipInterpolator:
        """Callable interpolator for lower surface spline.
        Returns lower airfoil ordinates at ``x``.

        Usage:
            `obj.lower_surface_at(x: Union[float, np.ndarray]) -> np.ndarray`

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            float, np.ndarray: lower surface y ordinate at x. (interpolator results)
        """
        points = self.lower_surface.evaluate_at(
            cosine_spacing(0, 1, num=2000)
        )
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=False)

    @cached_property
    def camber_line_at(self) -> si.PchipInterpolator:
        """Callable interpolator for the camber line.
        Returns camber line ordinates at ``x``.

        Usage:
            `obj.camber_line_at(x: Union[float, np.ndarray]) -> np.ndarray`

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            float, np.ndarray: camber line y ordinate at x. (interpolator results)
        """
        points = self.camber_line.evaluate_at(
            cosine_spacing(0, 1, num=2000)
        )
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=False)

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber at specified x locations.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolated results: camber value at x.
        """
        return self.camber_line_at(x)

    def thickness_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Calculates the thickness at specified x locations."""
        y_upper = self.upper_surface_at(x)
        y_lower = self.lower_surface_at(x)
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
    def from_points(
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
