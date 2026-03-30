import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.interpolate as si
import scipy.optimize as opt
import scipy.integrate as spi

from functools import cached_property
from typing import Union, Tuple
from abc import ABCMeta, abstractmethod, ABC
from  warnings import warn as warning

from ..data.datafile import AirfoilDataFile
from ..data.normalization import AirfoilNormalizer
from ..util import cosine_spacing, chebyshev_nodes, ensure_1d_vector, selig
from .spline import *
from .geom2d import Point2D, normalize_2d, rotate_2d_90ccw


AIRFOIL_REPR_REGEX = re.compile(r"[.]([A-Z])\w+")
EPS = 1e-12

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

    @property
    @abstractmethod
    def thickness_distribution(self):
        """Returns the thickness distribution curve"""

    @property
    @abstractmethod
    def camber_line(self):
        """Returns the lower surface curve"""

    @property
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered."""
        raise NotImplementedError

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
            cosine_spacing(0, 1, num=200)
        )
        # filter out all points with non-increasing x-coordinates
        # this prevents a lot of headaches
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=True)

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
            cosine_spacing(0, 1, num=200)
        )
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=True)

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
            cosine_spacing(0, 1, num=100)
        )
        x, y = points[points[:, 0] == np.maximum.accumulate(points[:, 0])].T
        assert np.all(np.diff(x) > 0)
        return si.PchipInterpolator(x, y, extrapolate=True)

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Simply a pointer to the camber_line_at.
        Returns the camber at specified x locations.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            float, np.ndarray: camber value at x.
        """
        return self.camber_line_at(x)

    def thickness_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Calculates the thickness at specified x locations.

        Args:
            x (float, np.ndarray): Chord-line fraction (0 = LE, 1 = TE)

        Returns:
            float, np.ndarray: thickness value at x.
        """
        y_upper = self.upper_surface_at(x)
        y_lower = self.lower_surface_at(x)
        return y_upper - y_lower

    @cached_property
    def max_thickness(self) -> Tuple[float, float]:
        """Finds and returns the location and value of maximum thickness.

        Returns:
            Tuple[float, float]: (x location, thickness value)
        """
        # result = opt.minimize(
        #     lambda x: -self.thickness_at(x), 0.5, bounds=[(0, 1)]
        # )
        # if not result.success:
        #     raise RuntimeError("Failed to find maximum thickness.")
        # return result.x[0], -result.fun
        result = opt.minimize(
            lambda u:
                -self.thickness_distribution.evaluate_at(u)[0][1],
                0.5,
                bounds=[(0, 1)]
        )
        if not result.success:
            print(result)
            raise RuntimeError("Failed to find upper crest.")
        return self.thickness_distribution.evaluate_at(result.x[0])

    @cached_property
    def max_camber(self) -> Tuple[float, float]:
        """Finds the location and value of maximum camber."""
        # result = opt.minimize(lambda x: -self.camber_at(x), 0.5, bounds=[(0, 1)])
        # if result.success:
        #     return result.x[0], -result.fun
        # else:
        #     raise RuntimeError("Failed to find maximum camber.")
        result = opt.minimize(
            lambda u: -abs(self.camber_line.evaluate_at(u)[0][1]), 0.5, bounds=[(0, 1)]
        )
        if not result.success:
            print(result)
            # raise RuntimeError("Failed to find max camber.")
            warning("Failed to find max camber.")
        return self.camber_line.evaluate_at(result.x[0])

    @cached_property
    def area(self) -> float:
        """Calculates the area of the airfoil."""
        x = cosine_spacing(0, 1, num=1000)
        t = self.thickness_at(x)
        return np.trapezoid(t, x)
        # return spi.simpson(t, x)

    @cached_property
    def leading_edge_radius(self) -> np.ndarray:
        """Returns the leading edge radius of the airfoil."""
        return self.surface.radius_at(0.5)

    @cached_property
    def upper_crest(self) -> np.ndarray:
        """Returns the upper crest of the airfoil."""
        result = opt.minimize(
            lambda u: -self.upper_surface.evaluate_at(u)[0][1], 0.5, bounds=[(0, 1)]
        )
        if not result.success:
            print(result)
            raise RuntimeError("Failed to find upper crest.")
        return self.surface.evaluate_at(result.x[0])

    @cached_property
    def lower_crest(self) -> np.ndarray:
        """Returns the upper crest of the airfoil."""
        result = opt.minimize(
            lambda u: self.lower_surface.evaluate_at(u)[0][1], 0.5, bounds=[(0, 1)]
        )
        if not result.success:
            print(result)
            raise RuntimeError("Failed to find upper crest.")
        return self.surface.evaluate_at(result.x[0])

    @cached_property
    def upper_crest_curvature(self) -> float:
        """Returns the curvature of the upper crest."""
        return self.upper_surface.curvature_at(self.upper_crest[0])

    @cached_property
    def lower_crest_curvature(self) -> float:
        """Returns the curvature of the lower crest."""
        return self.lower_surface.curvature_at(self.lower_crest[0])

    @cached_property
    def trailing_edge(self) -> float:
        """Returns the trailing edge ordinate of the airfoil."""
        return 0.5 * (
            self.upper_surface.evaluate_at(1.0) + self.lower_surface.evaluate_at(1.0)
        )

    @cached_property
    def trailing_edge_gap(self) -> float:
        """Returns the gap between the upper and lower surfaces at the trailing edge."""
        return np.abs(
            self.upper_surface_at(1) - self.lower_surface_at(1)
        )

    @cached_property
    def leading_edge(self) -> float:
        """Returns the leading edge ordinate of the airfoil."""
        raise NotImplementedError(
            "Leading edge undefined."
        )

    @cached_property
    def trailing_edge_upper_vector(self) -> np.ndarray:
        """Upper surface gradient at the trailing edge."""
        return self.upper_surface.first_deriv_at(1 - EPS)

    @cached_property
    def trailing_edge_lower_vector(self) -> np.ndarray:
        """Lower surface gradient at the trailing edge."""
        return self.lower_surface.first_deriv_at(1 - EPS)

    @cached_property
    def trailing_edge_vector(self) -> np.ndarray:
        """Vector between the upper and lower surface at the trailing edge."""
        return self.camber_line.tangent_at(1 - EPS)[0]

    @cached_property
    def leading_edge_vector(self) -> np.ndarray:
        """Vector between the upper and lower surface at the leading edge."""
        return self.camber_line.first_deriv_at(0 + EPS)

    @cached_property
    def trailing_edge_wedge_angle(self) -> float:
        """Angle between the upper and lower surface gradients at the trailing edge."""
        return np.arctan2(
            self.trailing_edge_upper_vector[1] - self.trailing_edge_lower_vector[1],
            self.trailing_edge_upper_vector[0] - self.trailing_edge_lower_vector[0],
        )

    @cached_property
    def trailing_edge_angle(self) -> float:
        """Angle between the upper and lower surface gradients at the trailing edge."""
        return np.arctan2(
            *self.trailing_edge_vector
        )

    @cached_property
    def leading_edge_angle(self) -> float:
        """Angle between the upper and lower surface gradients at the leading edge."""
        return np.arctan2(
            *self.leading_edge_vector
        )

    @property
    def points(self) -> np.ndarray:
        """Returns sampled points of the airfoil surface.
        Conventional format of 199 points in Selig format: 100 cosine-spaced
        points for upper and lower surface each, minus the duplicate leading
        edge point.
        """
        x = cosine_spacing(0, 1, num=100)
        return np.vstack([
            np.column_stack([x, self.upper_surface_at(x)])[::-1],
            np.column_stack([x, self.lower_surface_at(x)])[1:],
        ]).view(Point2D)

    def plot(self, n_points=1000, **pltkwargs):
        """Plots the airfoil geometry."""
        x = cosine_spacing(0, 1, num=n_points)
        fig, ax = plt.subplots()
        ax.plot(x, self.upper_surface_at(x), label="Upper Surface", **pltkwargs)
        ax.plot(x, self.lower_surface_at(x), label="Lower Surface", **pltkwargs)
        ax.plot(x, self.camber_at(x), label="Camber Line", **pltkwargs)
        ax.set_title(
            self.description.replace("#", "") \
            or self.name \
            or "Airfoil"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        ax.set_ylim(-0.4, 0.4)
        return fig, ax


class BsplineAirfoil(AirfoilBase):
    """Unified airfoil class handling points-based airfoils with inconsistent or
    missing data."""

    def __init__(
        self,
        surface_curve: ParametricCurve,
        name: str = "",
        description: str = "",
    ):
        # save the surface spline object
        self.surface_curve = surface_curve

        # # # optional curve definitions
        # self._upper_surface = upper_surface_curve
        # self._lower_surface = lower_surface_curve
        # self._thickness_distribution = thickness_distribution
        # self._camber_line = camber_line

        # the shortened name of the airfoil, usually the filename
        self.name = name
        # the full name of the airfoil, usually from the file header
        self.description = name if description is None else description.replace(' AIRFOIL', '')

        # If the spline is not a normalized one, these properties still need to
        # be added. There should be a better way to do this, but problems for later
        if not hasattr(self.surface_curve, 'u_leading_edge'):
            self.surface_curve.leading_edge, self.surface_curve.u_leading_edge = AirfoilNormalizer._find_leading_edge(
                self.surface_curve,
                0.5 * (self.surface_curve.evaluate_at(0) + self.surface_curve.evaluate_at(1))
            )

    @classmethod
    def from_coordinate_array(
        cls,
        points: np.ndarray,
        name: str = "",
        description: str = "",
        normalize: bool = True,
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
                Defaults to True. Keep it True, please.
                (if set to False, please don't, but also you better be damn sure
                the coordinates are properly formatted with a proper leading and
                trailing edge)

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

        # surface_spline = (
        #     AirfoilNormalizer.normalize(points) if normalize
        #     else BSpline2D(points)
        # )

        return cls(
            AirfoilNormalizer.normalized_bspline(points) if normalize else BSpline2D(points),
            name=name,
            description=description,
            # data_points=new_points,
        )

    @classmethod
    def from_file(
        cls,
        filepath: str,
        normalize: bool = True,
    ):
        """Returns an Airfoil object from a data file.
        Coordinates are normalized before passing to the Airfoil object.

        Includes setting the name and full name of the airfoil based on the
        filename and header, respectively.

        Args:
            filepath (str): path to the data file.
            normalize (bool): Whether to normalize the points before fitting.
                Defaults to True. Please, don't change it.
            kwargs (dict): Additional arguments for the fitting method.

        Returns:
            Airfoil: Airfoil object initialized with data from file.
        """
        datafile = AirfoilDataFile(filepath)
        return cls.from_coordinate_array(
            datafile.points,
            name=datafile.filename,
            description=datafile.header,
            normalize=normalize,
        )

    @classmethod
    def from_camber_thickness(
        cls,
        thickness_curve: ParametricCurve | np.ndarray,
        camber_curve: ParametricCurve | np.ndarray,
        name: str = "",
        description: str = ""
    ) -> "BsplineAirfoil":
        """
        Construct airfoil from camber and thickness distributions.

        Args:
            camber_curve (ParametricCurve): Bezier curve for camber line.
            thickness_curve (ParametricCurve): Bezier curve for thickness distribution.
            name (str): Optional name of airfoil.
            description (str): Optional long description.

        Returns:
            BezierAirfoil: New instance created from camber + thickness curves.
        """
        if not isinstance(camber_curve, type(thickness_curve)):
            raise TypeError("Camber and thickness curves must be of the same type (curve object or array).")

        # TODO if x values are already the same, no need for spline interpolation
        if isinstance(camber_curve, np.ndarray) and isinstance(thickness_curve, np.ndarray) \
            and (camber_curve[:,0] == thickness_curve[:,0]).all():
            camber_curve = BSpline2D(camber_curve)
            thickness_curve = BSpline2D(thickness_curve)

        # Bsplines are parametric, so interpolation is needed to ensure x-coordinates align
        x_vals = cosine_spacing(0, 1, num=200)
        camber_points = np.columnstack([
            x_vals,
            si.PchipInterpolator(
                *camber_curve.evaluate_at(x_vals).T,
                extrapolate=True
            )(x_vals)
        ])
        half_thickness_points = np.columnstack([
            x_vals,
            si.PchipInterpolator(
                *thickness_curve.evaluate_at(x_vals).T,
                extrapolate=True
            )(x_vals)/2
        ])
        surface_curve = BSpline2D(
            np.vstack([
                (camber_points + half_thickness_points)[::-1],
                (camber_points - half_thickness_points)[1:]
            ])
        )
        obj = cls(
            surface_curve,
            name=name,
            description=description,
        )
        obj.thickness_distribution = thickness_curve
        obj.camber_line = camber_curve
        return obj

    @property
    def surface(self) -> ParametricCurve:
        """Returns a parametric surface curve representing the entire airfoil."""
        return self.surface_curve

    @cached_property
    def upper_surface(self) -> ParametricCurve:
        """Returns the Bspline curve object for the upper surface of the airfoil.

        Returns:
            ParametricCurve: The upper surface Bspline object of the airfoil.
        """
        curve = BSpline2D(
            AirfoilNormalizer._remove_overshoots(
                self.surface.evaluate_at(
                    cosine_spacing(0, self.surface.u_leading_edge, num=100)[::-1]
                )
            )
        )
        # endpoint snapping to avoid precision errors
        adjusted_control_points = curve.control_points
        # force leading edge point to be at x=0
        adjusted_control_points[0][0] = 0.0
        adjusted_control_points[0][1] = 0.0
        # force final control point to bet at x=1
        direction = adjusted_control_points[-1] - adjusted_control_points[-2]
        magnitude = (1.0 - adjusted_control_points[-1][0]) / direction[0]
        adjusted_control_points[-1] += magnitude * direction
        curve.control_points = adjusted_control_points
        return curve

    @cached_property
    def lower_surface(self) -> ParametricCurve:
        """Returns the Bspline curve object for the lower surface of the airfoil.

        Returns:
            ParametricCurve: The lower surface Bspline object of the airfoil.
        """
        curve = BSpline2D(
            AirfoilNormalizer._remove_overshoots(
                self.surface.evaluate_at(
                    cosine_spacing(self.surface.u_leading_edge, 1, num=100)
                )
            )
        )
        # endpoint snapping to avoid precision errors
        adjusted_control_points = curve.control_points
        # force leading edge point to be at x=0
        adjusted_control_points[0][0] = 0.0
        adjusted_control_points[0][1] = 0.0
        # force final control point to bet at x=1
        direction = adjusted_control_points[-1] - adjusted_control_points[-2]
        magnitude = (1.0 - adjusted_control_points[-1][0]) / direction[0]
        adjusted_control_points[-1] += magnitude * direction
        curve.control_points = adjusted_control_points
        return curve

    @cached_property
    def camber_line(self) -> ParametricCurve:
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
        x = cosine_spacing(0, 1, 200)
        y_upper = self.upper_surface_at(x)
        y_lower = self.lower_surface_at(x)
        y_camber = 0.5 * (y_upper + y_lower)
        return BSpline2D(
            np.column_stack([x, y_camber])
        )

    @cached_property
    def thickness_distribution(self) -> ParametricCurve:
        """Returns the thickness distribution of the airfoil."""
        x = cosine_spacing(0, 1, 200)
        y_upper = self.upper_surface_at(x)
        y_lower = self.lower_surface_at(x)
        thickness = (y_upper - y_lower)
        return BSpline2D(
            np.column_stack([x, thickness])
        )


class BezierAirfoil(AirfoilBase):
    """Unified airfoil class handling points-based airfoils with inconsistent or
    missing data."""

    def __init__(
        self,
        surface_curve: ParametricCurve,
        name: str = "",
        description: str = "",
    ):
        # save the surface spline object
        self.surface_curve = surface_curve

        # the shortened name of the airfoil, usually the filename
        self.name = name
        # the full name of the airfoil, usually from the file header
        # self.description = description.replace(' AIRFOIL', '') if description is not None else name
        self.description = name if description is None else description.replace(' AIRFOIL', '')

        self.u_leading_edge = 0.5

    @classmethod
    def from_control_points(
        cls,
        control_points: np.ndarray,
        name: str = "",
        description: str = "",
        # normalize: bool = True,
        **kwargs
    ):
        """Creates an Airfoil object from control points.
        Creates the surface bezier curve and passes it to the Airfoil object.

        Args:
            control_points (np.ndarray):
                Array of airfoil control points.
            name (str):
                Name of the airfoil.
            description (str):
                Description of the airfoil. (Any additional text)
            normalize (bool):
                    Whether to normalize the points before fitting.
                    Defaults to True.
        Returns:
            Airfoil: Airfoil object initialized with input
        """
        return cls(
            SplevCBezier.from_control_points(control_points),
            name=name,
            description=description,
        )

    @classmethod
    def from_coordinate_array(
        cls,
        points: np.ndarray,
        name: str = "",
        description: str = "",
        normalize: bool = True,
        fit_method: str = "split_u_l",
        trailing_edge_thickness: float | None = None,
        find_trailing_edge: bool = True,
        curvefit_kwargs: dict = {},
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
                Whether to normalize the points before fitting. Defaults to
                True. Keep it True, please. (if set to False, please don't, but
                also you better be damn sure the coordinates are properly
                formatted with a proper leading and trailing edge)

            fit_method (str):
                Method to fit the airfoil. Options are:
                - "full": Fit the entire curve at once. Slow and risks swapping
                    upper and lower surfaces if the fit is poor, especially for
                    thin airfoils. Not recommended.
                - "split_u_l": Split the points into upper and lower and fit
                    those separately. Preferred method as it is much faster.
                - "split_t_c": First determine camber and thickness values, then
                    fit bezier curves to those.
                Defaults to "split_u_l". The main reason for this is to allow
                for different constraints, and fitting two curves separately is
                much faster than fitting one big one.

            n_control_points (int):
                Number of control points PER SIDE to use for the fitting.
                E.g. if 13, total number of control points will be 27: 13 for
                upper surface, 13 for lower surface, and 1 shared for the
                leading edge clamp. If None, it will be set to
                the number of points in the input array.
                Defaults to 13.

            control_point_spacing (np.ndarray):
                Spacing of the control points. Defaults to cosine spacing from 0
                to 1 with 13 points.

            end_clamp (str | np.ndarray):
                Clamping method for the end control points. Options are: - None:
                No clamping, the end control points will be fit. - "data": Use
                the last point of the data as the end control point. - "origin":
                Clamp to the origin (0, 0). - np.ndarray: Custom end control
                points as a 2-element array. Defaults to "data".

            trailing_edge_thickness (float | None):
                Thickness of the trailing edge. If specified, the end control
                points will be adjusted to ensure the trailing edge has the
                specified thickness. If None, no adjustment is made. Defaults to
                None.

            damping_type (str):
                Type of damping to apply during fitting. Options are: - "deriv":
                Damping based on the derivative of the curve. - "dist": Damping
                based on the distance from curve. - "none": No damping. Defaults
                to "deriv".

            w_damping (float):
                Weighting factor for the damping. Defaults to 1e-1.

            find_trailing_edge (bool):
                Whether to find the trailing edge of the airfoil. If True, the
                curve will be slightly processed to get a more accurate location
                of the trailing edge. If False, the trailing edge will be
                assumed to be the midpoint of the curve ends. Defaults to True.

        Raises:
            TypeError: Input must be a numpy array. ValueError: Invalid shape
            for input array. ValueError: Unknown fit method. ValueError: If the
            input array does not have the correct shape

        Returns:
            Airfoil: Airfoil object initialized with input

        TODO:
            - remove the start_clamp and end_clamp arguments, replace them with
              leading_edge and trailing_edge arguments, which are substituted
              accordingly by the clamp arguments for the curve fits.
                leading_edge=origin always trailing_edge:
                    - None | "free": no clamping, the end control points will be
                      fit
                    - data: use the last point of the data as the end control
                      point
                    - closed: clamp to the origin (0, 0)
                    - float: defines the trailing edge thickness, 0 would be the
                      same as "closed"
            - add an option for the trailing edge point to either be appended to
              the control points, or replace the final point (like it does now,
              which is why the spacing should be linspace(0,1,n)[:-1], as the
              last point is fixed and we dont want double points at x=1)

        """
        if not isinstance(points, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if points.shape[1] != 2:
            raise ValueError("Input array must have shape (n, 2).")

        if trailing_edge_thickness is not None and curvefit_kwargs.get('end_clamp', None) is not None:
            raise ValueError(
                "Cannot specify both trailing_edge_thickness and end_clamp."
                "Well, you could, and thickness would take priority, ignoring "
                "the end_clamp argument. So best to set end_clamp to None "
                "explicitly to avoid confusion."
                "This will (should)(probably)(maybe) be fixed in the future."
            )

        if normalize:
            normalized_bspline = AirfoilNormalizer.normalized_bspline(points)
            # If the bspline fits poorly and causes overlap, us the original points
            if np.any((normalized_bspline.evaluate_at(np.linspace(0, normalized_bspline.u_leading_edge, 1000)).y - normalized_bspline.evaluate_at(np.linspace(normalized_bspline.u_leading_edge, 1, 1000)).y).round(8) < 0):
                points = AirfoilNormalizer.normalize_points(points).round(8)
                # make sure the leading edge is included in the normalized
                # points. this should not alter the curve in any way, as the
                # origin should part of it after normalization. This simply adds
                # a point on the curve, or moves one along it.
                #if len(points) % 2 == 1:
                #    points[len(points) // 2] = np.array([0.0, 0.0])
                #elif len(points) % 2 == 0:
                #    np.insert(points, len(points) // 2, [0.0, 0.0], axis=0)
                    # If origin is already present, do nothing
                if np.any(np.all(np.isclose(points, [0.0, 0.0], atol=1e-8), axis=1)):
                    return points
                # Find index of point with smallest x (leading edge region)
                i_le = np.argmin(points[:, 0])
                # Ensure it's between upper and lower surfaces
                # Insert [0, 0] *before* i_le if it's the first lower surface point
                points = np.insert(points, i_le if points[i_le, 1] < 0 else i_le + 1, np.array([0.0, 0.0]), axis=0)

                if fit_method == "split_t_c":
                    warning(
                        "Fitting BSpline for normalizatoin causes surface overlap, likely due to poor data quality. "
                        "Using original points for fitting and switching to 'split_u_l' method."
                    )
                    fit_method = "split_u_l"
            else:
                # To improve the fitting of the new spline, points are
                # resampled from the normalized spline.
                points = AirfoilNormalizer._remove_overshoots(
                    normalized_bspline.evaluate_at(
                        np.hstack([
                            cosine_spacing(0, normalized_bspline.u_leading_edge, num=100),
                            cosine_spacing(normalized_bspline.u_leading_edge, 1, num=100)[1:],
                        ])
                    )
                )

        match fit_method:
            case "split_u_l":
                # split the points in upper and lower and fit those seperatly
                # This is the prefered method as it is much faster that 'full'

                # first check trailing edge thickness if specified
                if trailing_edge_thickness is not None:
                    upper_end_clamp = (np.array([1.0, trailing_edge_thickness / 2]), 'replace')
                    lower_end_clamp = (np.array([1.0, -trailing_edge_thickness / 2]), 'replace')

                curvefit_kwargs['start_clamp'] = curvefit_kwargs.get('start_clamp', ('origin', 'append'))
                curvefit_kwargs['end_clamp'] = curvefit_kwargs.get('end_clamp', ('data', 'replace')) if trailing_edge_thickness is None else upper_end_clamp
                upper = SplevBezier.fit(
                    points[len(points)//2::-1],
                    **curvefit_kwargs,
                    constraints=[
                        {   # force y>x for first control point after the LE
                            "type": "ineq",
                            "fun": lambda y: y[0] - 0.002
                        },
                        {   # ensure rounded leading edge (see GOE440)
                            "type": "ineq",
                            "fun": lambda y: y[1] - y[0] * 0.5
                        },
                    ] if curvefit_kwargs.get("control_point_spacing", None) is not None else [
                        {   # ensure increasing x values when spacing is free
                            "type": "ineq",
                            "fun": lambda y: np.diff(y.reshape(-1, 2)[:, 0]) - 1e-3
                        },
                        {   # ensure x values > 0 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: y.reshape(-1, 2)[:, 0]
                        },
                        {   # ensure x values < 1 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: 1 - y.reshape(-1, 2)[:, 0]
                        },
                        {   # ensure first x value = 0 when spacing is free
                            "type": "eq",
                            "fun": lambda y: y.reshape(-1, 2)[0, 0]
                        },
                        {   # ensure first y value > 0.002 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: y.reshape(-1, 2)[0, 1] - 0.002
                        },
                        {   # ensure second y value > y1 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: y.reshape(-1, 2)[1, 1] - y.reshape(-1, 2)[0, 1]
                        },
                    ]
                )
                curvefit_kwargs['end_clamp'] = curvefit_kwargs.get('end_clamp', ('data', 'replace')) if trailing_edge_thickness is None else lower_end_clamp
                lower = SplevBezier.fit(
                    points[len(points)//2:],
                    **curvefit_kwargs,
                    constraints=[
                        {   # force y>x for first control point after the LE
                            "type": "ineq",
                            "fun": lambda y: -y[0] - 0.002
                        },
                        {   # ensure rounded leading edge (see GOE440)
                            "type": "ineq",
                            "fun": lambda y: -y[1] + y[0]*0.5
                        },
                    ] if curvefit_kwargs.get("control_point_spacing", None) is not None else [
                        {   # ensure increasing x values when spacing is free
                            "type": "ineq",
                            "fun": lambda y: np.diff(y.reshape(-1, 2)[:, 0]) - 1e-3
                        },
                        {   # ensure x values > 0 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: y.reshape(-1, 2)[:, 0]
                        },
                        {   # ensure x values < 1 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: 1 - y.reshape(-1, 2)[:, 0]
                        },
                        {   # ensure first x value = 0 when spacing is free
                            "type": "eq",
                            "fun": lambda y: y.reshape(-1, 2)[0, 0]
                        },
                        {   # ensure first y value > 0.002 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: -y.reshape(-1, 2)[0, 1] - 0.002
                        },
                        {   # ensure second y value > y1 when spacing is free
                            "type": "ineq",
                            "fun": lambda y: -y.reshape(-1, 2)[1, 1] + y.reshape(-1, 2)[0, 1]
                        },
                    ]
                )
                surfacespline = SplevCBezier.from_control_points(
                    np.vstack([
                        upper.control_points[::-1],
                        lower.control_points[1:]
                    ])
                )
                surfacespline.points = points
                airfoil = cls(
                    surfacespline,
                    name=name,
                    description=description,
                )
                airfoil.upper_surface = upper
                airfoil.lower_surface = lower
                return airfoil

            case "split_t_c":
                bspline_airfoil = BsplineAirfoil(normalized_bspline)

                # ---- adjust endpoint kwargs ----
                if trailing_edge_thickness is not None:
                    t_end_clamp = (np.array([1.0, trailing_edge_thickness]), 'replace')
                    curvefit_kwargs['end_clamp'] = t_end_clamp
                curvefit_kwargs['start_clamp'] = curvefit_kwargs.get('start_clamp', ('origin', 'append'))
                curvefit_kwargs['end_clamp'] = curvefit_kwargs.get('end_clamp', None)

                # ---- fit thickness curve ----
                thickness_distribution = SplevBezier.fit(
                    bspline_airfoil.thickness_distribution.evaluate_at(
                        cosine_spacing(0, 1, num=200)
                    ),
                    # n_control_points=n_control_points,
                    # spacing=control_point_spacing,
                    # start_clamp='origin',
                    # end_clamp=end_clamp,
                    # damping_type=damping_type,
                    # w_damping=w_damping,
                    **curvefit_kwargs,
                    constraints=[
                        {   # ensure t > 0
                            "type": "ineq",
                            "fun": lambda y: SplevBezier.from_control_points(
                                np.column_stack([
                                    np.linspace(0, 1, len(y)),
                                    y
                                ]).evaluate_at(np.linspace(1e-8, 1, 100)).y
                            )
                        },
                        # {   # ensure rounded leading edge (see GOE440)
                        #     "type": "ineq",
                        #     "fun": lambda y: y[0] - 0.005
                        # },
                        # {   # ensure rounded leading edge (see GOE440)
                        #     "type": "ineq",
                        #     "fun": lambda y: y[1] - y[0]*0.5
                        # },
                    ] if curvefit_kwargs.get("control_point_spacing", None) is None else [
                        {   # ensure increasing x values when spacing is free
                            "type": "ineq",
                            "fun": lambda y: np.diff(y.reshape(-1, 2)[:, 0])
                        },
                        {   # ensure x values > 0when spacing is free
                            "type": "ineq",
                            "fun": lambda y: y.reshape(-1, 2)[:, 0]
                        }
                    ]
                )
                # ---- fit camber curve ----
                camber_curve = SplevBezier.fit(
                    bspline_airfoil.camber_line.evaluate_at(
                        cosine_spacing(0, 1, num=200)
                    ),
                    # n_control_points=n_control_points,
                    # spacing=control_point_spacing,
                    # start_clamp='origin',
                    # end_clamp=end_clamp,
                    # damping_type=damping_type,
                    # w_damping=w_damping,
                    **curvefit_kwargs,
                    constraints=[
                        # {
                        #     "type": "ineq",
                        #     "fun": lambda y: y[0] + 0.002
                        # },
                        # {
                        #     "type": "ineq",
                        #     "fun": lambda y: 0.002 - y[0]
                        # },
                    ] if curvefit_kwargs.get("control_point_spacing", None) is None else [
                        {   # ensure increasing x values when spacing is free
                            "type": "ineq",
                            "fun": lambda y: np.diff(y.reshape(-1, 2)[:, 0])
                        },
                        {   # ensure x values > 0when spacing is free
                            "type": "ineq",
                            "fun": lambda y: y.reshape(-1, 2)[:, 0]
                        }
                    ]
                )
                return cls.from_camber_thickness(
                    thickness_distribution,
                    camber_curve,
                    name=name,
                    description=description,
                    points=points,
                )
            case "full":  # ! this one is a bad idea, as it is very slow
                # TODO : change this so it combines upper and lower fit in a
                # TODO | single optimization? (in the splevcbezier class)
                warning(
                    "Using 'full' fit method is not recommended, as it is very slow. "
                )
                # Use the composite bezier class to fit the entire curve at once
                surfacespline = SplevCBezier.fit(
                    points,
                    curvefit_kwargs.get('n_control_points'),
                    spacing=curvefit_kwargs.get('control_point_spacing'),
                    w_damping=curvefit_kwargs.get('w_damping', 1e-1),
                )
                return cls(
                    surfacespline,
                    name=name,
                    description=description,
                )
            case _:
                raise ValueError(f"Unknown fit method: {fit_method}")

    @classmethod
    def from_file(
        cls,
        filepath: str,
        normalize: bool = True,
        data_type: str = "coordinates",
        curvefit_kwargs: dict = {'n_control_points': 13},
        **kwargs: dict
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
        match data_type:
            case "coordinates":
                return cls.from_coordinate_array(
                    datafile.points,
                    name=datafile.filename,
                    description=datafile.header,
                    normalize=normalize,
                    curvefit_kwargs=curvefit_kwargs,
                    **kwargs
                )
            case "control_points":
                return cls.from_control_points(
                    datafile.points,
                    name=datafile.filename,
                    description=datafile.header,
                )
            case _:
                raise ValueError(f"Unknown data type: {data_type}")

    @classmethod
    def from_camber_thickness(
        cls,
        thickness_curve: ParametricCurve | np.ndarray,
        camber_curve: ParametricCurve | np.ndarray,
        name: str = "",
        description: str = "",
        points: np.ndarray = None,
    ) -> "BezierAirfoil":
        """
        Construct airfoil from camber and thickness distributions.

        Args:
            camber_curve (ParametricCurve): Bezier curve for camber line.
            thickness_curve (ParametricCurve): Bezier curve for thickness distribution.
            name (str): Optional name of airfoil.
            description (str): Optional long description.

        Returns:
            BezierAirfoil: New instance created from camber + thickness curves.
        """
        if not isinstance(camber_curve, type(thickness_curve)):
            raise TypeError("Camber and thickness curves must be of the same type (curve object or array).")

        if isinstance(camber_curve, ParametricCurve) and isinstance(thickness_curve, ParametricCurve):
            # Ensure x-coordinates align
            x_camber = camber_curve.control_points[:, 0]
            y_camber = camber_curve.control_points[:, 1]
            x_thickness = thickness_curve.control_points[:, 0]
            y_thickness = thickness_curve.control_points[:, 1]
        else:
            # Assume they are numpy arrays
            x_camber = camber_curve[:, 0]
            y_camber = camber_curve[:, 1]
            x_thickness = thickness_curve[:, 0]
            y_thickness = thickness_curve[:, 1]

        if not np.all(x_camber == x_thickness):
            raise ValueError("Camber and thickness curves must have the same x-coordinates.")
        else:
            x = x_camber

        assert np.all(x_camber == x_thickness), \
            "Camber and thickness curves must have the same x-coordinates."

        # Reconstruct upper/lower surface control points
        y_upper = y_camber + 0.5 * y_thickness
        y_lower = y_camber - 0.5 * y_thickness

        upper_cp = np.column_stack([x[::-1], y_upper[::-1]])  # Reverse for trailing → leading
        lower_cp = np.column_stack([x[1:], y_lower[1:]])      # Skip duplicate LE point

        # Combine to single surface curve (just like your other logic)
        control_points = np.vstack([upper_cp, lower_cp])

        surface_curve = SplevCBezier.from_control_points(control_points)
        surface_curve.points = points

        # Attach original components for reference if needed
        # surface_curve.camber_curve = camber_curve
        # surface_curve.thickness_curve = thickness_curve

        airfoil = cls(
            surface_curve,
            name=name,
            description=description,
        )
        airfoil.thickness_distribution = thickness_curve
        airfoil.camber_line = camber_curve
        return airfoil

        # return cls(
        #     surface_curve,
        #     name=name,
        #     description=description,
        # )

    @property
    def surface(self) -> ParametricCurve:
        """Returns a parametric surface curve representing the entire airfoil."""
        return self.surface_curve

    @cached_property
    def upper_surface(self) -> ParametricCurve:
        return SplevBezier(
            self.surface.control_points[self.surface.n_control_points//2::-1]
        )

    @cached_property
    def lower_surface(self) -> ParametricCurve:
        return SplevBezier(
            self.surface.control_points[self.surface.n_control_points//2:]
        )

    @cached_property
    def camber_line(self) -> ParametricCurve:
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
    def thickness_distribution(self) -> ParametricCurve:
        """Returns the thickness distribution of the airfoil."""
        assert np.all(
            self.upper_surface.control_points.x == self.lower_surface.control_points.x
        )
        thickness_control_points = np.column_stack([
            self.upper_surface.control_points[:,0],
            (
                self.upper_surface.control_points[:,1] -
                self.lower_surface.control_points[:,1]
            )
        ])
        return SplevBezier(thickness_control_points)


class CSTAirfoil(AirfoilBase):
    def __init__(
        self,
        upper_coeficients: np.ndarray,
        lower_coeficients: np.ndarray,
        leading_edge_weight: float = 0.0,
        trailing_edge_weight: float = 0.0,
        trialing_edge_thickness: float = 0.0,
        data_points: np.ndarray = None,
        name: str = None,
        full_name: str = None,
        normalize: bool = True,
    ):
        # the original input points, mainly for reference
        self.data_points = data_points
        # the data points after processing, used for fitting

        # the shortened name of the airfoil, usually the filename
        self.name = name
        # the full name of the airfoil, usually from the file header
        self.full_name = full_name or name

    @classmethod
    def from_coordinate_array(
        cls,
        points: np.ndarray,
        normalize: bool = True,
        name: str = "",
        description: str = "",
    ):
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

        if normalize:
            points = AirfoilNormalizer.normalize_points(points).T#, find_trailing_edge=True).T
            # make sure the leading edge is included in the normalized
            # points. this should not alter the curve in any way, as the
            # origin should part of it after normalization. This simply adds
            # a point on the curve, or moves one along it.
            if len(points) % 2 == 1:
                points[len(points) // 2] = np.array([0.0, 0.0])
            elif len(points) % 2 == 0:
                np.insert(points, len(points) // 2, [0.0, 0.0], axis=0)

        return cls(
            data_points=points,
            name=name
        )

    @classmethod
    def from_file(
        cls,
        filepath: str,
        normalize: bool = True,
        data_type: str = "coordinates",
        **kwargs: dict
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
        match data_type:
            case "coordinates":
                return cls.from_coordinate_array(
                    datafile.points,
                    name=datafile.filename,
                    description=datafile.header,
                    normalize=normalize,
                    **kwargs
                )
            case "control_points":
                return cls.from_control_points(
                    datafile.points,
                    name=datafile.filename,
                    description=datafile.header,
                    **kwargs,
                )
            case _:
                raise ValueError(f"Unknown data type: {data_type}")

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




class NACA4Airfoil(AirfoilBase):
    """Creates a NACA 4 series :py:class:`Airfoil` from digit input.
    NACA code format "NACAcxtt":
        - c: maximum camber value * 100
        - x: chordwise location of maximum camber * 10
        - tt: maximum thickness * 100

    Args:
        naca_code: 4-digit NACA airfoil code, i.e. "naca0012" or "0012"

    Keyword Arguments:
        te_closed: Sets if the trailing-edge of the airfoil is closed.
            Defaults to False.

    Attributes:
        max_camber: Maximum camber as a percentage of the chord. Valid
            inputs range from 0-9 % maximum camber. Defaults to 0.
        camber_location: Location of maximum camber in tenths of the
            chord length. A value of 1 would mean 10% of the chord.
            Defaults to 0.
        max_thickness: Maximum thickness as a percentage of the chord.
    """

    def __init__(
        self, naca_code: str, *, te_closed: bool = False,
    ):
        self.max_camber, self.camber_location, self.max_thickness = self.parse_naca_code(
            naca_code
        )
        if (self.max_camber == 0) ^ (self.camber_location == 0):
            raise ValueError(
                "Non-zero camber value cannot have 0 as chordwise location."
            )

        self.te_closed = te_closed

    @property
    def cambered(self) -> bool:
        """Returns if the current :py:class:`Airfoil` is cambered."""
        return self.max_camber != 0 and self.camber_location != 0

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line points at the supplied ``x``."""
        # Setting up chord-line and camber-line point arrays
        x = self.ensure_1d_vector(x)
        pts_c = np.zeros((x.size, 2))
        pts_c[:, 0] = x

        # Localizing inputs for speed and clarity
        m = self.max_camber
        p = self.camber_location

        if self.cambered:
            fwd, aft = x <= p, x > p  # Indices before and after max ordinate
            pts_c[fwd, 1] = (m / (p ** 2)) * (2 * p * x[fwd] - x[fwd] ** 2)
            pts_c[aft, 1] = (m / (1 - p) ** 2) * (
                (1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2
            )
        return pts_c

    def thickness_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the thickness value at ``x``."""
        return 2 * self.half_thickness_at(x)

    def camber_tangent_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line tangent vector at supplied ``x``."""
        # Setting up chord-line and camber-line tangent arrays
        x = self.ensure_1d_vector(x)
        t_c = np.repeat(
            np.array([[1, 0]], dtype=np.float64), repeats=x.size, axis=0
        )

        # Localizing inputs for speed and clarity
        m = self.max_camber
        p = self.camber_location

        if self.cambered:
            fwd, aft = x <= p, x > p  # Indices before and after max ordinate
            t_c[fwd, 1] = (2 * m / p ** 2) * (p - x[fwd])
            t_c[aft, 1] = (2 * m / (1 - p) ** 2) * (p - x[aft])

        return normalize_2d(t_c, inplace=True)

    def camber_normal_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the camber-line normal vector at supplied ``x``.

        Note:
            This method implements a fast 2D Affine Transform.
        """
        return rotate_2d_90ccw(self.camber_tangent_at(x))

    def upper_surface_at(self, x: np.ndarray) -> np.ndarray:
        """Returns upper surface points at the supplied ``x``."""
        return self.camber_at(x) + self.offset_vectors_at(x)

    def lower_surface_at(self, x: np.ndarray) -> np.ndarray:
        """Returns lower surface points at the supplied ``x``."""
        return self.camber_at(x) - self.offset_vectors_at(x)

    def offset_vectors_at(self, x: np.ndarray) -> np.ndarray:
        """Returns half-thickness magnitude vectors at ``x``."""
        n_c = self.camber_normal_at(x)  # Camber normal-vectors
        y_t = self.half_thickness_at(x)  # Half thicknesses
        return np.multiply(n_c, y_t.reshape(x.size, 1), out=n_c)

    def half_thickness_at(self, x: np.ndarray) -> np.ndarray:
        """Calculates the NACA-4 series 'Half-Thickness' y_t at ``x``.

        Args:
            x: Chord-line fraction (0 = LE, 1 = TE)
        """
        x = self.ensure_1d_vector(x)
        return (self.max_thickness / 0.2) * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * (x ** 2)
            + 0.2843 * (x ** 3)
            - (0.1036 if self.te_closed else 0.1015) * (x ** 4)
        )

    def plot(self, *args, show: bool = True, **kwargs):
        """Specializes the :py:class:`Airfoil` plot with a title."""
        # Turning off plot display to be able to display after the
        # title is added to the plot
        fig, ax = super().plot(*args, **kwargs, show=False)
        ax.set_title(
            "{name} {te_shape} Trailing-Edge Airfoil".format(
                name=self.name, te_shape="Closed" if self.te_closed else "Open"
            )
        )
        plt.show() if show else ()  # Rendering plot window if show is true
        return fig, ax

    @property
    def name(self) -> str:
        """Returns the name of the airfoil from current attributes."""
        return "NACA{m:.0f}{p:.0f}{t:02.0f}".format(
            m=self.max_camber * 100,
            p=self.camber_location * 10,
            t=self.max_thickness * 100,
        )

    def __repr__(self) -> str:
        """Overwrites string repr. to include airfoil name."""
        return re.sub(
            AIRFOIL_REPR_REGEX, f".{self.name}Airfoil", super().__repr__()
        )

    @staticmethod
    def parse_naca_code(naca_code: str) -> map:
        """Parses a ``naca_code`` into the 3 (scaled) values needed:
        max camber, max camber location, and maximum thickness.

        Note:
            ``naca_code`` can include the prefix "naca" or "NACA".

        Raise:
            ValueError: If a``naca_code`` is supplied with
                missing digits or invalid characters.

        Returns:
            Tuple(float, float, float):
                max camber, max camber location, max thickness.
        """
        digits = naca_code.upper().strip("NACA")
        if len(digits) == 4 and all(d.isdigit() for d in digits):
            max_camber, camber_location, max_t1, max_t2 = map(int, digits)
            return (
                max_camber / 100,
                camber_location / 10,
                float(f".{max_t1}{max_t2}")  # 0."t1""t2"
            )
        else:
            raise ValueError("NACA code must contain 4 numbers")

    # For compatibility, individual curves are also represented as splines
    # @cached_property
    def upper_surface(self) -> ParametricCurve:
        """Returns the upper surface curve of the airfoil as Bspline."""
        return BSpline2D(
            np.column_stack([
                x := cosine_spacing(0, 1, num=200),
                self.upper_surface_at(x)
            ])
        )

    # @cached_property
    def lower_surface(self) -> ParametricCurve:
        """Returns the lower surface curve of the airfoil as Bspline."""
        return BSpline2D(
            np.column_stack([
                x := cosine_spacing(0, 1, num=200),
                self.lower_surface_at(x)
            ])
        )

    # @cached_property
    def surface(self)-> ParametricCurve:
        x = cosine_spacing(0, 1, num=200)
        return BSpline2D(
            np.vstack([
                np.column_stack([ x, self.upper_surface_at(x)])[::-1],
                np.column_stack([ x, self.lower_surface_at(x)])[1:]
            ])
        )

    # @cached_property
    def camber_line(self) -> ParametricCurve:
        """Returns the camber line curve of the airfoil as Bspline."""
        return BSpline2D(
            np.column_stack([
                x := cosine_spacing(0, 1, num=200),
                self.camber_at(x)[:, 1]
            ])
        )

    # @cached_property
    def thickness_distribution(self) -> ParametricCurve:
        """Returns the thickness distribution curve of the airfoil as Bspline."""
        return BSpline2D(
            np.column_stack([
                x := cosine_spacing(0, 1, num=200),
                self.thickness_at(x)
            ])
        )



