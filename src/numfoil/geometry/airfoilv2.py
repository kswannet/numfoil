# Copyright 2020 Kilian Swannet, San Kilkis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Contains all :py:class:`Airfoil` class definitions."""

import re
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Tuple, Union
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import scipy.interpolate as si
from scipy.optimize import minimize

from .geom2d import normalize_2d, rotate_2d_90ccw
from .spline import BSpline2D

from ..util import Container, cosine_spacing

# TODO Add NACA5 series Airfoil as a fun nice-to-have feature

AIRFOIL_REPR_REGEX = re.compile(r"[.]([A-Z])\w+")


class Airfoil(metaclass=ABCMeta):
    """Abstract Base Class definition of an :py:class:`Airfoil`.

    The responsibility of an :py:class:`Airfoil` is to provide methods
    that return points on the airfoil upper and lower surface as well as
    the camber-line when given a normalized location along the
    chord-line ``x``. A single point can be obtained by calling these
    methods with a singular float. Alternatively, multiple points can
    be obtained by passing a :py:class:`numpy.ndarray` with n
    row-vectors resulting in a shape: (n, 2).

    The origin of the coordinate system (x=0, y=0, z=0) is located at
    the leading-edge of an airfoil. The positive x-axis is aligned with
    the chord-line of the airfoil, hence the coordinate (1, 0, 0) would
    represent the trailing-edge. The positive z-axis is aligned with
    the positive thickness direction of the airfoil. The positive
    y-axis is then going into the page, hence clock-wise rotations are
    positive. This definition follows that of Katz and Plotkin.
    """

    @property
    @abstractmethod
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered."""

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

    @staticmethod
    def ensure_1d_vector(x: Union[float, np.ndarray]) -> np.ndarray:
        """Ensures that ``x`` is a 1D vector."""
        x = np.array([x]) if isinstance(x, (float, int)) else x
        if len(x.shape) != 1:
            raise ValueError("Only 1-D np.arrays are supported")
        return x

    def plot(
        self, n_points: int = 200, show: bool = True
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plots the airfoil with ``n_points`` per curve.

        Args:
            n_points: Number of points used per airfoil curve
            show: Determines if the plot window should be launched

        Returns:
            Matplotlib plot objects:

                [0]: Matplotlib Figure instance
                [1]: Matplotlib Axes instance
        """

        # Setting up cosine sampled chord-values
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=n_points)))

        # Retrieving pts on all curves
        pts_lower = self.lower_surface_at(x)
        pts_upper = self.upper_surface_at(x)

        fig, ax = plt.subplots()
        ax.plot(pts_upper[:, 0], pts_upper[:, 1], label="Upper Surface")
        ax.plot(pts_lower[:, 0], pts_lower[:, 1], label="Lower Surface")
        if self.cambered:
            pts_camber = self.camberline_at(x)
            ax.plot(pts_camber[:, 0], pts_camber[:, 1], label="Camber Line")
        ax.legend(loc="best")
        ax.set_xlabel("Normalized Location Along Chordline (x/c)")
        ax.set_ylabel("Normalized Thickness (t/c)")
        plt.axis("equal")
        plt.show() if show else ()  # Rendering plot window if show is true

        return fig, ax


class PointsAirfoil(Airfoil):
    """Class definition of an airfoil from given coordinate points.

    Note:
        Airfoils coordinates are ordered in Selig format, such that the first
        point is the trailing edge, and the points go from the upper surface to
        the lower surface and back to the trailing edge (Counter Clockwise).

    Args:
        points: array of airfoil surface coordinates in Selig format.
    """

    def __init__(self, points: np.ndarray):
        self.points = self.remove_consecutive_duplicates(points)

    @cached_property
    def property_plotter(self):
        return AirfoilPlot(self)

    @cached_property
    def _cache(self):
        return Container(
            surface_spline_c=self.surface.spline
        )

    @cached_property
    def surface(self) -> BSpline2D:
        """create a spline through all points,
        convering the entire airfoil surface"""
        return BSpline2D(self.points, degree=3)

    @cached_property
    def le_idx(self) -> np.ndarray:
        """Returns the leading edge index within :py:attr:`points`."""
        return np.argmin(self.points[:,0])

    @cached_property
    def le_u(self) -> np.ndarray:
        """Returns the leading edge location (u) on the surface spline as the
        point furthest away from the trailing edge."""
        return minimize(
            lambda u: -np.linalg.norm(
                self.surface.evaluate_at(u[0]) - self.trailing_edge, axis=0),
                0.5,
                bounds=[(0, 1)]
            ).x[0]

    @cached_property
    def leading_edge_point(self) -> np.ndarray:
        """Returns the leading edge point coordinate as the
        point furthest away from the trailing edge."""
        # assert self.surface.evaluate_at(self.le_u) == minimize(lambda x: self.surface.evaluate_at(x)[0, 0], 0.5, bounds=[(0, 1)]).x[0]
        return self.surface.evaluate_at(self.le_u)

    @cached_property
    def trailing_edge(self) -> np.ndarray:
        """Returns the [x,y] coordinate of the trailing edge.
        Trailing edge is taken as the midpoint between surface spline ends."""
        return 0.5*(self.surface.evaluate_at(0) + self.surface.evaluate_at(1))

    @cached_property
    def upper_surface(self) -> BSpline2D:
        """Returns a spline of the upper airfoil surface."""
        return BSpline2D(
            self.surface.evaluate_at(
                cosine_spacing(0, self.le_u, 200)
            )[::-1]
        )

    @cached_property
    def lower_surface(self) -> BSpline2D:
        """Returns a spline of the lower airfoil surface."""
        return BSpline2D(
            self.surface.evaluate_at(
                cosine_spacing(self.le_u, 1, 200)
            )
        )

    # TODO consider using geom2d views here
    @cached_property
    def mean_camber_line(self) -> BSpline2D:
        """Returns a spline of the mean-camber line.

        Note:
            The mean camber line is defined as the average distance
            between the top and bottom surfaces when measured normal
            to the chord-line.
        """
        u = np.linspace(0, 1, num=200)
        upper_pts = self.upper_surface.evaluate_at(u)
        lower_pts = self.lower_surface.evaluate_at(u)
        return BSpline2D(0.5 * (upper_pts + lower_pts))

    @property
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered.

        The algorithm works by flipping the lower surface points and
        checking if the distance between the top and bottom points are
        within a set tolerance.
        """
        u = np.linspace(0, 1, num=10)
        upper_pts = self.upper_surface.evaluate_at(u)
        lower_pts = self.lower_surface.evaluate_at(u)
        lower_pts[:, 1] *= -1  # Flipping the lower surface across the chord
        return not np.allclose(upper_pts, lower_pts, atol=1e-6)

    def remove_consecutive_duplicates(self, arr: np.ndarray) -> np.ndarray:
        """Removes consecutive duplicate coordinates.
            > If the file contains consecutive duplicates, splprep will throw an
              error when trying to create the surface spline.
        """
        diff = np.diff(arr, axis=0)                         # Compute the difference between consecutive rows
        idx = np.where(np.any(diff != 0, axis=1))[0] + 1    # Find the indices of the rows that are different from the preceding row
        coor = np.vstack((arr[0], arr[idx]))                # Append the first row and the rows that are different from the preceding row
        return coor

    @cached_property
    def upper_surface_at(self) -> si.PchipInterpolator:
        """Interpolator for upper surface spline.
        Returns upper airfoil ordinates at the supplied ``x``.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: upper surface y ordinate at x.
        """
        # ! ugly fix, for some reason interpolator doesn't always go as far as 1
        # ! so instead force the inteprolator beyond 1
        u = cosine_spacing(0, 1.01, num=200)
        x, y = self.upper_surface.evaluate_at(u).T
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
        # ! ugly fix, for some reason interpolator doesn't always go as far as 1
        # ! so instead force the inteprolator beyond 1
        u = cosine_spacing(0, 1.01, num=200)
        x, y = self.lower_surface.evaluate_at(u).T
        return si.PchipInterpolator(x, y, extrapolate=False)

    @cached_property
    def camberline_at(self) -> si.PchipInterpolator:
        """Interpolator for camber line spline.
        Returns camberline ordinates at the supplied ``x``.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            interpolator results: camber(line) y (ordinate) at x.
        """
        u = cosine_spacing(0, 1, num=200)
        x, y = self.mean_camber_line.evaluate_at(u).T
        return si.PchipInterpolator(x, y, extrapolate=False)

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        # return self.mean_camber_line.evaluate_at(u)[1]
        return self.camberline_at(x)

    def thickness_at(
        self, x: Union[float, np.ndarray]) -> np.ndarray:
        """find the thickness at location x.

        Args:
            x (Union[float, np.ndarray]): chordwise location(s)

        Returns:
            Tuple[np.ndarray, np.ndarray]: thickness at location x
        """
        # np.atleast_1d(u)
        return self.upper_surface_at(x) - self.lower_surface_at(x)

    @cached_property
    def max_thickness_spline(self) -> Tuple[np.ndarray, np.ndarray]:
        # result = minimize(lambda x: -self.thickness_at(x[0]), 0.5, bounds=[(0, 1)])
        result = minimize(
            lambda u: -abs(self.surface.evaluate_at(u[0])[1] - self.surface.evaluate_at(u[1])[1]),
            [0.25, 0.75],
            constraints={
                'type': 'eq',
                'fun': lambda u: (self.surface.evaluate_at(u[0])[0] - self.surface.evaluate_at(u[1])[0])**2,
            },
            method="SLSQP"
        )
        if not result.success:
            raise Exception("Finding max thickness failed: " + result.message)
        return result.x, -result.fun if result.success else float('nan')

    @cached_property
    def max_thickness(self) -> Tuple[np.ndarray, np.ndarray]:
        result = minimize(lambda x: -abs(self.thickness_at(x)**2), 0.5,)
        if not result.success:
            raise Exception("Finding max thickness failed: " + result.message)
        return result.x[0], np.sqrt(-result.fun) if result.success else float('nan')

    @cached_property
    def le_radius(self) -> float:
        return self.surface.radius_at(self.le_u)

    # @cached_property
    @property
    def trailing_edge_gap(self) -> float:
        # return np.linalg.norm(
        #     self.upper_surface_at(1) - self.lower_surface_at(1)
        #     )
        return np.linalg.norm(
            self.surface.evaluate_at(0) - self.surface.evaluate_at(1)
            )

    @cached_property
    def trailing_edge_wedge_angle(self) -> float:
        upper_tangent = self.upper_surface.tangent_at(1)[0]
        lower_tangent = self.lower_surface.tangent_at(1)[0]
        return np.arccos(np.dot(upper_tangent, lower_tangent))

    @cached_property
    def max_camber(self) -> Tuple[np.ndarray, np.ndarray]:
        """finds maximum camber location and value

        Raises:
            Exception: optimization failed

        Returns:
            Tuple[np.ndarray, np.ndarray]: [u_max_camber, max_camber]
        """
        result = minimize(lambda u: -self.mean_camber_line.evaluate_at(u[0])[1], 0.5, bounds=[(0, 1)])
        # result = minimize(lambda x: -self.camber_at(x[0]), 0.5, bounds=[(0, 1)])
        if not result.success:
            raise Exception("Finding max camber failed: " + result.message)
        return result.x[0], -result.fun if result.success else float('nan')

    def set_trailing_edge_gap(self, gap, rf: int = 4):
        """Set trailing edge gap.
        Modifies the surface spline by moving the control points in the
        direction of the normal.

        Args:
            gap (float): desired trailing edge gap / thickness
            rf (int, optional) DEFAULT=4: reduction factor for the surface
                displacement effect. The larger this number, the smaller the
                part of the airfoil affected by the change in trailing edge
                thickness.
        """
        delattr(self, "surface")                                # reset to original surface by forcing recalculation
        dgap = (self.trailing_edge_gap - gap)/2                 # upper and lower surface offset at trailing edge
        ui = self.surface.spline[1]                             # locations of control points on the spline
        normals = self.surface.normal_at(ui)                    # normals at the control points
        scaling_factors = abs(2*(ui-0.5))**rf                   # scaling factor for displacement effect
        self.surface.spline[0][1] += (                          # modify spline control points
            normals * dgap * scaling_factors[:, np.newaxis]
        ).T

        delattr(self, "upper_surface")      # reset spline to new spline
        delattr(self, "lower_surface")      # reset spline to new spline
        delattr(self, "upper_surface_at")   # reset interpolators to new spline
        delattr(self, "lower_surface_at")   # reset interpolators to new spline
        delattr(self, "trailing_edge_gap")  # reset interpolators to new spline

        # self.surface.spline = self._cache.surface_spline          # reset to original surface
        # dgap = (np.linalg.norm(self.surface.evaluate_at(1) - self.surface.evaluate_at(0)) - gap)/2
        # ui = self.surface.spline[1]                                 # spline locations of control points
        # normals = self.surface.normal_at(ui)                        # normals at the control points
        # surface_points = self.surface.evaluate_at(ui)
        # distances = np.linalg.norm(surface_points - self.trailing_edge, axis=1)
        # scaling_factors = (1 - distances)**4                      # alternate way to determine scaling factor based on distance from trailin edge



class ProcessedPointsAirfoil(PointsAirfoil):
    """Provided points are stored in the `unprocessed_points' attr. Points are
    processed by creating an airfoil surface spline through all points. The
    surface spline is then used to resample the airfoil coordinates for
    consistency.

    A minimization then determines the translation, rotation, and scaling
    matrices which ensure the airfoil has a camberline going from (x,y) = (0,0)
    to (1,0).

    The minimzation is needed since the rotation can change which point is the
    leading edge (foremost), requiring translation again, or vice versa.
    """
    def __init__(self, points: np.ndarray):
        self.unprocessed_points = self.remove_consecutive_duplicates(points)

    @cached_property
    def surface(self) -> BSpline2D:
        """Return surface spline of the processed airfoil.

        Create a spline through the provided/original point coordinates.
        The spline (through its control points) is then translated, rotated,
        and scaled such that the airfoil chordline lies on the x-axis, going
        from (x,y) = (0,0) -> (1,0).

        The entire process goes as follows:
            1. Create surface spline through provided point coordinates.
            2. Determine trailing edge point as the midpoint between the
               spline end-points.
            3. Determine leading edge as the point on the spline furthest away
               from the trialing edge point.
            4. Calculate the chord vector.
            5. Find scaling factor based on chord vector length.
            6. Find translation vector based on leading edge coordinate.
            7. Determine rotation angle from the chord vector.
            8. apply translation, scaling, and  rotation to the surface spline
               control points.
        """
        surface_spline = BSpline2D(self.unprocessed_points, degree=3)
        self._cache.unprocessed_surface = surface_spline
        trailing_edge = 0.5*(surface_spline.evaluate_at(0) + surface_spline.evaluate_at(1))
        LE_u = minimize(
            lambda u: -np.linalg.norm(
                surface_spline.evaluate_at(u[0]) - trailing_edge, axis=0
                ),
            0.5,
            bounds=[(0, 1)]
        ).x[0]
        leading_edge = surface_spline.evaluate_at(LE_u)
        chord_vector = trailing_edge - leading_edge

        scaling = 1/np.linalg.norm(chord_vector)
        translation = -leading_edge[:, np.newaxis]
        angle = -np.arctan2(chord_vector[1], chord_vector[0])

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        surface_spline.spline[0][1] = (
            rotation_matrix @ (
                (surface_spline.spline[0][1] + translation) * scaling
            )
        )
        return surface_spline

    @cached_property
    def points(self) -> np.ndarray:
        """Return processed airfoil surface coordinates
        Resample the points form the processed spline, using cosine spacing with
        a total of 199 points in Selig format: 100 points on upper and lower
        surface, without the duplicate LE point

        Returns:
            coordinates (array): 199 airfoil coordinates in Selig format
        """
        u = np.append(
            cosine_spacing(0, self.le_u, 100),      # Upper surface LE to TE
            cosine_spacing(self.le_u, 1, 100)[1:])  # Lower surface TE to LE
        return self.surface.evaluate_at(u)


class FileAirfoil(PointsAirfoil):
    """Base class definition of an airfoil read from a file.

    Note:
        Airfoils coordinates are ordered such that the first point is
        the trailing edge, and the points go from the upper surface
        to the lower surface and back to the trailing edge (Counter
        Clockwse).

    Args:
        filepath: Absolute filepath to the airfoil file including
            file extension.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def parse_file(self) -> np.ndarray:
        """Reads :py:attr:`self.filename` and returns a points array."""

    @cached_property
    def points(self) -> np.ndarray:
        """Returns airfoil ordinate points x, y as row-vectors."""
        return self.remove_consecutive_duplicates(self.parse_file())


class ProcessedFileAirfoil(ProcessedPointsAirfoil):
    def __init__(self, filepath: str):
        self.filepath = filepath

    @cached_property
    def unprocessed_points(self) -> np.ndarray:
        """Returns airfoil ordinate points x, y as row-vectors."""
        return self.remove_consecutive_duplicates(self.parse_file())


class UIUCAirfoil(ProcessedFileAirfoil):#FileAirfoil):#
    """Creates :py:class:`Airfoil` from a UIUC airfoil file."""

    def parse_file(self) -> np.ndarray:
        """UIUC files are ordered correctly and have 1 header line."""
        return np.genfromtxt(self.filepath, skip_header=1)

    @cached_property
    def fullname(self) -> str:
        """Returns the name of the airfoil from header line."""
        with open(self.filepath, 'r') as file:
            first_line = file.readline().strip()
        return first_line

    @cached_property
    def name(self) -> str:
        """Returns the name of the airfoil from filename."""
        return os.path.splitext(os.path.basename(self.filepath))[0]

    def plot(self, *args, show: bool = True, **kwargs):
        """Specializes the :py:class:`Airfoil` plot with a title."""
        # Turning off plot display to be able to display after the
        # title is added to the plot
        fig, ax = super().plot(*args, **kwargs, show=False)
        ax.set_title(
            "UIUC {name} Airfoil".format(
                name=self.fullname
            )
        )
        plt.show() if show else ()  # Rendering plot window if show is true
        return fig, ax

    def __repr__(self) -> str:
        """Overwrites string repr. to include airfoil name."""
        # return f"{super().__repr__()}.{os.path.splitext(os.path.basename(self.filepath))[0]}"
        return re.sub(
            AIRFOIL_REPR_REGEX, f".UIUCAirfoil.{os.path.splitext(os.path.basename(self.filepath))[0]}", super().__repr__()
        )


class AirfoilPlot:
    """ Create a plotter object with ready functions to plot different
    properties of the airfoil. calling the respective methods more than once
    will toggle the property plot.

    plot methods toggle properties shown. Methods are decorated as properties
    just save the extra 2 characters to type.
    """
    def __init__(self, airfoil: Airfoil, n_points: int = 200) -> None:
        self.airfoil = airfoil
        self.n_points = n_points
        self.elements = Container(
            surface_line=None,
            upper_surface_line=None,
            lower_surface_line=None,
            camber_line=None,
            LE_radius=None,
            points=None,
            max_camber=None,
            max_thickness=None,
            max_curvature=None,
            surface_curvature=[],
            upper_curvature=[],
            lower_curvature=[],
            camber_curvature=[],
        )

        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.draw()

        # self.fig, self.ax = plt.subplots()
        # self.ax.set_xlabel("Normalized Location Along Chordline (x/c)")
        # self.ax.set_ylabel("Normalized Thickness (t/c)")
        # self.ax.set_title(airfoil.fullname)
        # plt.axis("equal")
        # self.upper_surface, self.lower_surface, self.camber_line
        # self.ax.legend(loc="best")
        # plt.show()

    def draw(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Normalized Location Along Chordline (x/c)")
        self.ax.set_ylabel("Normalized Thickness (t/c)")
        self.ax.set_title(self.airfoil.fullname)
        plt.axis("equal")
        self.upper_surface, self.lower_surface, self.camber_line
        self.ax.legend(loc="best")
        plt.show()

    # @cached_property
    @property
    def x_cos(self) -> np.ndarray:
        """ X-locations for the airfoil coordinates with cosine spacing
        Returns:
            list:  100 pts cosine spacing
        """
        # return 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=self.n_points)))
        return cosine_spacing(0, 1, self.n_points)

    # @cached_property
    @property
    def x_coor(self) -> np.ndarray:
        """ X-coordinates from trailing edge to leading edge over the upper
        surface, then back to trailing edge along the lower surface.

        Returns:
            list:  199 pts along entire airfoil surface with cosine spacing
        """
        return np.append(self.x_cos[::-1], self.x_cos[1:])

    @property
    def upper_surface(self):
        if self.elements.upper_surface_line:
            self.elements.upper_surface_line.remove()
            self.elements.upper_surface_line = None
        else:
            self.elements.upper_surface_line = self.ax.plot(
                self.x_cos, self.airfoil.upper_surface_at(self.x_cos),
                label="Upper Surface",
                color=self.colors[0]
                )[0]
            # self.ax.add_line(self.elements.upper_surface_line)
        self.ax.legend()

    @property
    def lower_surface(self):
        if self.elements.lower_surface_line:
            self.elements.lower_surface_line.remove()
            self.elements.lower_surface_line = None
        else:
            self.elements.lower_surface_line = plt.plot(
                self.x_cos, self.airfoil.lower_surface_at(self.x_cos),
                label="Lower Surface",
                color=self.colors[1]
                )[0]
            # self.ax.add_patch(self.elements.lower_surface_line)
        self.ax.legend()

    @property
    def camber_line(self):
        if self.elements.camber_line:
            self.elements.camber_line.remove()
            self.elements.camber_line = None
        else:
            self.elements.camber_line = plt.plot(
                self.x_cos, self.airfoil.camberline_at(self.x_cos),
                label="Camber Line",
                color=self.colors[2]
                )[0]
            # self.ax.add_line(self.elements.camber_line)
        self.ax.legend()

    @property
    def surface(self):
        if self.elements.surface_line:
            self.elements.surface_line.remove()
            self.elements.surface_line = None
        else:
            u1 = cosine_spacing(0, self.airfoil.le_u, self.n_points)
            u2 = cosine_spacing(self.airfoil.le_u, 1, self.n_points)
            u = np.append(u1[:-1], u2)
            x, y = self.airfoil.surface.evaluate_at(u).T
            self.elements.surface_line = self.ax.plot(
                x, y, label="Airfoil Surface",
                color=self.colors[3]
                )[0]
            # self.ax.add_line(self.elements.surface_line)
        self.ax.legend()

    @property
    def points(self):
        if self.elements.points:
            self.elements.points.remove()
            self.elements.points = None
        else:
            pts = self.airfoil.points
            self.elements.points = plt.plot(
                pts[:, 0], pts[:, 1], 'x', label="Airfoil Coordinate Points",
                color=self.colors[4]
                )[0]
            # self.ax.add_line(self.elements.points)
        self.ax.legend()

    @property
    def LE_radius(self):
        """Leading edge radius plot consists of several elements:
            - the leading edge itself
            - the leading edge radius circle
            - leading edge circle centroid
            - leading edge radius itself connecting leading edge with centroid
        """
        if self.elements.LE_radius:
            # self.elements.LE_point.remove()
            self.elements.LE_radius_circle.remove()
            # self.elements.LE_radius_c.remove()
            self.elements.LE_radius.remove()
            self.elements.LE_radius = None
        else:
            le = self.airfoil.surface.evaluate_at(self.airfoil.le_u)
            radius = self.airfoil.surface.radius_at(self.airfoil.le_u)
            center = le + radius * self.airfoil.surface.normal_at(self.airfoil.le_u)[0]
            self.elements.LE_radius_circle = plt.Circle(center, radius,
                color="r", linestyle="--", linewidth=1.5, fill=False,
                label=f"Leading Edge Radius {radius:.2e}",
                )
            self.ax.add_patch(self.elements.LE_radius_circle)
            xs, ys = np.column_stack((le, center))
            self.elements.LE_radius = self.ax.plot(xs, ys, 'r--o')[0]#, label="Leading Edge Radius")[0]
            # self.elements.LE_radius_c = self.ax.plot(center[0], center[1], 'go')[0]#, label="Leading Edge Radius")[0]
            # self.elements.LE_radius = self.ax.plot(np.column_stack((le, center))[0], np.column_stack((le, center))[1], 'g--', label=f"Leading Edge Radius {radius:.2e}")[0]
            # self.elements.LE_point = self.ax.plot(le[0], le[1], 'ro', label="Leading Edge")[0]
        self.ax.legend()

    @property
    def max_curvature(self):
        """Leading edge radius plot consists of several elements:
            - the leading edge itself
            - the leading edge radius circle
            - leading edge circle centroid
            - leading edge radius itself connecting leading edge with centroid
        """
        if self.elements.max_curvature:
            self.elements.max_curvature_circle.remove()
            self.elements.max_curvature.remove()
            self.elements.max_curvature = None
        else:
            u, c = self.airfoil.surface.max_curvature
            radius = 1/c
            point = self.airfoil.surface.evaluate_at(u)
            center = point + radius * self.airfoil.surface.normal_at(u)[0]
            self.elements.max_curvature_circle = plt.Circle(center, radius,
                color="grey", linestyle="--", linewidth=1.5, fill=False,
                # label="Maximum Surface Curvature",
                )
            self.ax.add_patch(self.elements.max_curvature_circle)
            xs, ys = np.column_stack((point, center))
            self.elements.max_curvature = self.ax.plot(
                xs, ys,
                color='grey', marker='o',
                label=f"Maximum Curvature - R={radius:.2e}"
            )[0]
        self.ax.legend()

    @property
    def max_camber(self):
        if self.elements.max_camber:
            self.elements.max_camber.remove()
            self.elements.max_camber = None
        else:
            u, c = self.airfoil.max_camber
            x, y = self.airfoil.mean_camber_line.evaluate_at(u)
            self.elements.max_camber = plt.plot(
                [x, x], [0,y], '-*', label=f"Maximum Camber {c:.2e}",
                color=self.colors[5]
                )[0]
            # self.ax.add_line(self.elements.max_camber)
        self.ax.legend()

    @property
    def max_thickness(self):
        if self.elements.max_thickness:
            self.elements.max_thickness.remove()
            self.elements.max_thickness = None
        else:
            # u, t = self.airfoil.max_thickness_spline
            # x, y = self.airfoil.surface.evaluate_at(u).T
            x, t = self.airfoil.max_thickness
            y = self.airfoil.lower_surface_at(x)
            assert np.allclose(y+t, self.airfoil.upper_surface_at(x))
            self.elements.max_thickness = plt.plot(
                # x, y, '-*', label=f"Maximum Thickness {t:.2e}",
                [x, x], [y, y+t], '-*', label=f"Maximum Thickness {t:.2e}",
                color=self.colors[6]
                )[0]
            # self.ax.add_line(self.elements.max_thickness)
        self.ax.legend()

    @property
    def upper_curvature(self):
        if self.elements.upper_curvature:
            for line in self.elements.upper_curvature:
                line.remove()
            self.elements.upper_curvature = []
        else:
            for u in np.linspace(0, 1, num=100):
                point = self.airfoil.upper_surface.evaluate_at(u)
                c = self.airfoil.upper_surface.curvature_at(u)
                radius = c/100#1/c
                center = point + np.sqrt(radius) * self.airfoil.upper_surface.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.upper_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Upper Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    @property
    def lower_curvature(self):
        if self.elements.lower_curvature:
            for line in self.elements.lower_curvature:
                line.remove()
            self.elements.lower_curvature = []
        else:
            for u in np.linspace(0, 1, num=100):
                point = self.airfoil.lower_surface.evaluate_at(u)
                c = self.airfoil.lower_surface.curvature_at(u)
                radius = c/100#1/c
                center = point + np.sqrt(radius) * self.airfoil.lower_surface.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.lower_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Lower Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    @property
    def surface_curvature(self):
        if self.elements.surface_curvature:
            for line in self.elements.surface_curvature:
                line.remove()
            self.elements.surface_curvature = []
        else:
            for u in np.linspace(0, 1, num=200):
                point = self.airfoil.surface.evaluate_at(u)
                c = self.airfoil.surface.curvature_at(u)
                radius = c/100#1/c
                center = point + np.sqrt(radius) * self.airfoil.surface.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.surface_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Camber Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    @property
    def camber_curvature(self):
        if self.elements.camber_curvature:
            for line in self.elements.camber_curvature:
                line.remove()
            self.elements.camber_curvature = []
        else:
            for u in np.linspace(0, 1, num=100):
                point = self.airfoil.mean_camber_line.evaluate_at(u)
                c = self.airfoil.mean_camber_line.curvature_at(u)
                radius = c/100#1/c
                center = point + np.sqrt(radius) * self.airfoil.mean_camber_line.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.camber_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Camber Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    def reset(self):
        delattr(self.airfoil, "property_plotter")



# from numfoil.geometry.airfoilv2 import UIUCAirfoil
# from numfoil.util import cosine_spacing
# airfoil = UIUCAirfoil("src/data/UIUC_airfoils/fx72150a.dat")
# airfoil.property_plotter
