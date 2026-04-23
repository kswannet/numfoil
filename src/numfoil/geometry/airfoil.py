import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.interpolate as si
import scipy.optimize as opt
import scipy.integrate as spi

from functools import cached_property
from typing import Union, Tuple, Optional, Literal
from abc import ABCMeta, abstractmethod, ABC
from  warnings import warn as warning

from numfoil.data.datafile import AirfoilDataFile
from numfoil.data.normalization import AirfoilNormalizer
from numfoil.util import cosine_spacing, chebyshev_nodes, ensure_1d_vector, selig
from numfoil.geometry.spline import (
    Curve,
    ParametricCurve,
    BSpline2D,
    Bezier, SplevCBezier, SplevBezier,
    CSTCurve, KulfanModifiedCST, CSTAirfoilSurface,
)
from .geom2d import (
    GeometricMoments2D,
    Point2D,
    Vector2D,
    normalize_2d,
    rotate_2d_90ccw,
)


from scipy.special import comb

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
        if not np.all(np.diff(points) > 0):
            warning(
                "Upper surface x-coordinates are not strictly increasing. "
                "Interpolator may be inaccurate."
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
        if not np.all(np.diff(points) > 0):
            warning(
                "Upper surface x-coordinates are not strictly increasing. "
                "Interpolator may be inaccurate."
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
        Uses optimization for exact solution, stored as a cached property so it
        only needs to run once.

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
        """Calculates the approximate area of the airfoil.
        For a more accurate solution, call the area property of `geometric_moments`.
        """
        x = cosine_spacing(0, 1, num=1000)
        t = self.thickness_at(x)
        return np.trapezoid(t, x)
        # return spi.simpson(t, x)

    @cached_property
    def geometric_moments(self) -> GeometricMoments2D:
        """Cached geometric moments computed from the closed airfoil contour.

        Returns:
            GeometricMoments2D: Reusable moment container for area descriptors.
        """
        return GeometricMoments2D.from_polygon(self.points, max_total_order=4)

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
        return self.upper_surface.evaluate_at(result.x[0])

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
    def trailing_edge_thickness(self) -> float:
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
        return self.upper_surface.first_deriv_at(1 - EPS).view(Vector2D)

    @cached_property
    def trailing_edge_lower_vector(self) -> np.ndarray:
        """Lower surface gradient at the trailing edge."""
        return self.lower_surface.first_deriv_at(1 - EPS).view(Vector2D)

    @cached_property
    def trailing_edge_vector(self) -> np.ndarray:
        """Vector between the upper and lower surface at the trailing edge."""
        return self.camber_line.tangent_at(1 - EPS)[0].view(Vector2D)

    @cached_property
    def leading_edge_vector(self) -> np.ndarray:
        """Vector between the upper and lower surface at the leading edge."""
        return self.camber_line.first_deriv_at(0 + EPS).view(Vector2D)

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

    def plot(self, n_points=1000, show=False, **pltkwargs):
        """Plots the airfoil geometry."""
        x = cosine_spacing(0, 1, num=n_points)
        fig, ax = plt.subplots()
        ax.plot(x, self.upper_surface_at(x), label="Upper Surface", **pltkwargs)
        ax.plot(x, self.lower_surface_at(x), label="Lower Surface", **pltkwargs)
        ax.plot(x, self.camber_at(x), label="Camber Line", **pltkwargs)
        ax.set_title(
            self.description.replace("#", "")
            or self.name
            or "Airfoil"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        ax.set_ylim(-0.4, 0.4)
        if show:
            plt.show()
        return fig, ax

    ###############################
    ### Aliases for convenience ###
    ###############################

    @property
    def t_max(self):
        return self.max_thickness

    @property
    def c_max(self):
        return self.max_camber

    @property
    def r_le(self):
        return self.leading_edge_radius

    @property
    def z_u(self):
        return self.upper_crest

    @property
    def z_l(self):
        return self.lower_crest

    @property
    def k_z_u(self):
        return self.upper_crest_curvature

    @property
    def k_z_l(self):
        return self.lower_crest_curvature

    @property
    def t_te(self):
        return self.trailing_edge_thickness

    @property
    def trailing_edge_gap(self):
        return self.trailing_edge_thickness


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
            self.surface_curve.leading_edge, self.surface_curve.u_leading_edge = AirfoilNormalizer.find_leading_edge(
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
            AirfoilNormalizer.remove_overshoots(
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
            AirfoilNormalizer.remove_overshoots(
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
                points = AirfoilNormalizer.remove_overshoots(
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


class KulfanAirfoil(AirfoilBase):
    """Represent one Kulfan/CST airfoil with NumPy-only curve models.

    This class is the single-airfoil (non-batched) counterpart to the torch
    implementation and stores one upper and one lower `KulfanModifiedCST`
    surface that share the LE/TE modifiers.

    Math:
        y_u(x) = y_cst,u(x) + w_le * x * (1-x)^{n+0.5} + t_te * x / 2
        y_l(x) = y_cst,l(x) + w_le * x * (1-x)^{n+0.5} - t_te * x / 2

    Parameter layout:
        [upper_coeffs | lower_coeffs | le_weight | te_thickness]

    Args:
        upper_surface (KulfanModifiedCST): Upper Kulfan-modified CST curve.
        lower_surface (KulfanModifiedCST): Lower Kulfan-modified CST curve.
        name (str): Short airfoil label.
        description (str): Optional long description.

    Returns:
        None: Class is initialized in place.

    Attributes:
        upper_surface (KulfanModifiedCST): Upper surface curve object.
        lower_surface (KulfanModifiedCST): Lower surface curve object.
        name (str): Short airfoil label.
        description (str): Optional long description.
        full_name (str): Full airfoil name, typically from description or name.
        data_points (np.ndarray | None):
            Optional array of raw data points used for fitting.

    Methods:
        from_tensor: Class method to build from flat parameter vector.
        from_kulfan_params: Class method to build from Kulfan parameters.
        ...
    """

    def __init__(
        self,
        upper_surface: KulfanModifiedCST,
        lower_surface: KulfanModifiedCST,
        name: str = "",
        description: str = "",
    ):
        """Initialize a single Kulfan airfoil object.

        The constructor only accepts fully formed upper/lower surface objects
        and validates consistency of shared parameters.

        Math:
            Shared constraints: n1_u = n1_l, n2_u = n2_l,
            w_le,u = w_le,l, t_te,u = t_te,l

        Args:
            upper_surface (KulfanModifiedCST): Upper surface model.
            lower_surface (KulfanModifiedCST): Lower surface model.
            name (str): Airfoil short name.
            description (str): Optional textual description.

        Returns:
            None: Attributes are stored on the instance.
        """
        self._upper_surface = upper_surface
        self._lower_surface = lower_surface
        self.name = name
        self.description = name if description is None else description
        self.full_name = self.description or name
        self.data_points = None

        self.eps = max(self._upper_surface.eps, self._lower_surface.eps)
        self._validate_curves()

    def _validate_curves(self) -> None:
        """Validate upper/lower surface compatibility for one airfoil.

        This guard ensures both surfaces are structurally consistent and use the
        expected orientation convention (upper positive, lower negative near LE).

        Math:
            K_u = K_l,
            n1_u = n1_l,
            n2_u = n2_l,
            w_le,u = w_le,l,
            t_te,u = t_te,l

        Args:
            None.

        Returns:
            None: Raises on incompatibility.
        """
        if self._upper_surface.n_coefficients != self._lower_surface.n_coefficients:
            # this is probably not strictly necessary, but good to be consistent
            raise ValueError(
                "Upper and lower surfaces must use the same number of coefficients"
            )
        if self._upper_surface.surface_type != "upper":
            raise ValueError("upper_surface.surface_type must be 'upper'")
        if self._lower_surface.surface_type != "lower":
            raise ValueError("lower_surface.surface_type must be 'lower'")

        if not self._upper_surface.n1 == self._lower_surface.n1:
            raise ValueError("Upper and lower surfaces must share n1 exponent")
        if not self._upper_surface.n2 == self._lower_surface.n2:
            raise ValueError("Upper and lower surfaces must share n2 exponent")

        if not np.isclose(
            self._upper_surface.leading_edge_weight,
            self._lower_surface.leading_edge_weight,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("Upper/lower surfaces must share leading-edge weight")

        if not np.isclose(
            self._upper_surface.trailing_edge_thickness,
            self._lower_surface.trailing_edge_thickness,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("Upper/lower surfaces must share trailing-edge thickness")

    @classmethod
    def from_tensor(
        cls,
        parameters: np.ndarray,
        name: str = "",
        description: str = "",
        n1: float = 0.5,
        n2: float = 1.0,
    ) -> "KulfanAirfoil":
        """Build an airfoil from a flattened Kulfan parameter vector.

        Math:
            len(p) = 2K + 2
            p = [a_u(0..K-1), a_l(0..K-1), w_le, t_te]

        Args:
            parameters (np.ndarray): Flat vector of shape [2K + 2].
            name (str): Airfoil short name.
            description (str): Optional textual description.
            n1 (float): Leading-edge class exponent.
            n2 (float): Trailing-edge class exponent.

        Returns:
            KulfanAirfoil: Instantiated single-airfoil object.
        """
        params = np.asarray(parameters, dtype=float).reshape(-1)
        if params.size < 4:
            raise ValueError("Need at least 4 parameters: [u, l, w_le, t_te]")
        if params.size % 2 != 0:
            raise ValueError("Parameter length must be even: 2*n_coeff + 2")

        n_coeffs = (params.size - 2) // 2
        upper_coeffs = params[:n_coeffs]
        lower_coeffs = params[n_coeffs:-2]
        w_le = float(params[-2])
        t_te = float(params[-1])

        return cls.from_kulfan_params(
            upper_coeffs=upper_coeffs,
            lower_coeffs=lower_coeffs,
            w_le=w_le,
            t_te=t_te,
            n1=n1,
            n2=n2,
            name=name,
            description=description,
        )

    @classmethod
    def from_kulfan_params(
        cls,
        upper_coeffs: np.ndarray,
        lower_coeffs: np.ndarray,
        w_le: float = 0.0,
        t_te: float = 0.0,
        n1: float = 0.5,
        n2: float = 1.0,
        name: str = "",
        description: str = "",
    ) -> "KulfanAirfoil":
        """Build an airfoil from explicit upper/lower Kulfan parameters.
        This is the more verbose, keyword-argument version of `from_tensor`.

        Math:
            y_u uses +t_te*x/2,
            y_l uses -t_te*x/2,
            both share the same w_le and t_te.

        Args:
            upper_coeffs (np.ndarray): Upper CST coefficients [K].
            lower_coeffs (np.ndarray): Lower CST coefficients [K].
            w_le (float): Leading-edge modifier weight.
            t_te (float): Trailing-edge thickness magnitude.
            n1 (float): Leading-edge class exponent.
            n2 (float): Trailing-edge class exponent.
            name (str): Airfoil short name.
            description (str): Optional textual description.

        Returns:
            KulfanAirfoil: Instantiated airfoil object.
        """
        upper_curve = KulfanModifiedCST(
            coefficients=np.asarray(upper_coeffs, dtype=float).reshape(-1),
            leading_edge_weight=float(w_le),
            trailing_edge_thickness=float(abs(t_te)),
            surface_type="upper",
            n1=n1,
            n2=n2,
        )
        lower_curve = KulfanModifiedCST(
            coefficients=np.asarray(lower_coeffs, dtype=float).reshape(-1),
            leading_edge_weight=float(w_le),
            trailing_edge_thickness=float(abs(t_te)),
            surface_type="lower",
            n1=n1,
            n2=n2,
        )
        return cls(
            upper_surface=upper_curve,
            lower_surface=lower_curve,
            name=name,
            description=description,
        )

    @classmethod
    def from_points(
        cls,
        upper_points: np.ndarray,
        lower_points: np.ndarray,
        n_coefficients: int = 8,
        *,
        trailing_edge_solution: Literal["fit", "data"] = "data",
        n1: float = 0.5,
        n2: float = 1.0,
        name: str = "",
        description: str = "",
    ) -> "KulfanAirfoil":
        """Fit and construct an airfoil from separate upper/lower coordinates.

        This convenience constructor forwards to `fit` with identical
        parameters.

        Math:
            p* = argmin_p ||y_u(x) - y_u,data||_2 + ||y_l(x) - y_l,data||_2

        Args:
            upper_points (np.ndarray): Upper surface points [N_u, 2].
            lower_points (np.ndarray): Lower surface points [N_l, 2].
            n_coefficients (int): Coefficients per side.
            trailing_edge_solution (Literal["fit", "data"]): TE strategy.
            n1 (float): Leading-edge class exponent.
            n2 (float): Trailing-edge class exponent.
            name (str): Airfoil short name.
            description (str): Optional textual description.

        Returns:
            KulfanAirfoil: Fitted airfoil object.
        """
        return cls.fit(
            upper_points=upper_points,
            lower_points=lower_points,
            n_coefficients=n_coefficients,
            trailing_edge_solution=trailing_edge_solution,
            n1=n1,
            n2=n2,
            name=name,
            description=description,
        )

    @classmethod
    def fit(
        cls,
        upper_points: np.ndarray,
        lower_points: np.ndarray,
        n_coefficients: int = 8,
        *,
        use_kulfan_modifiers: bool = True,
        trailing_edge_solution: Literal["fit", "data"] = "data",
        n1: float = 0.5,
        n2: float = 1.0,
        name: str = "",
        description: str = "",
        rcond: Optional[float] = None,
    ) -> "KulfanAirfoil":
        """Jointly fit upper/lower Kulfan-CST surfaces to point clouds.

        The solve is linear in unknown coefficients and optional Kulfan
        modifiers. With shared modifiers enabled, a block system is assembled
        for both surfaces.

        Math:
            M theta = y
            theta = [a_u, a_l, w_le, t_te]  (fit mode)
            theta = [a_u, a_l, w_le]         (data TE mode)

        Args:
            upper_points (np.ndarray): Upper points [N_u, 2].
            lower_points (np.ndarray): Lower points [N_l, 2].
            n_coefficients (int): Number of coefficients per side.
            use_kulfan_modifiers (bool): If True, fit LE/TE modifiers.
            trailing_edge_solution (Literal["fit", "data"]): TE strategy.
            n1 (float): Leading-edge class exponent.
            n2 (float): Trailing-edge class exponent.
            name (str): Airfoil short name.
            description (str): Optional textual description.
            rcond (Optional[float]): Least-squares cutoff.

        Returns:
            KulfanAirfoil: Fitted airfoil object.
        """
        upper_points = np.asarray(upper_points, dtype=float)
        lower_points = np.asarray(lower_points, dtype=float)

        if upper_points.ndim != 2 or upper_points.shape[1] != 2:
            raise ValueError("upper_points must have shape [N, 2]")
        if lower_points.ndim != 2 or lower_points.shape[1] != 2:
            raise ValueError("lower_points must have shape [N, 2]")
        if trailing_edge_solution not in ("fit", "data"):
            raise ValueError("trailing_edge_solution must be 'fit' or 'data'")

        min_samples = n_coefficients + (2 if use_kulfan_modifiers else 0)
        if upper_points.shape[0] < min_samples or lower_points.shape[0] < min_samples:
            raise ValueError(
                f"Need at least {min_samples} points per side for this fit"
            )

        x_upper = upper_points[:, 0]
        y_upper = upper_points[:, 1]
        x_lower = lower_points[:, 0]
        y_lower = lower_points[:, 1]

        if np.any((x_upper < 0.0) | (x_upper > 1.0)):
            raise ValueError("upper_points x coordinates must be in [0, 1]")
        if np.any((x_lower < 0.0) | (x_lower > 1.0)):
            raise ValueError("lower_points x coordinates must be in [0, 1]")

        eps = np.finfo(float).eps
        x_upper = np.clip(x_upper, eps, 1.0 - eps)
        x_lower = np.clip(x_lower, eps, 1.0 - eps)

        k = np.arange(n_coefficients, dtype=float)
        n = n_coefficients - 1
        binom = comb(n, k)

        # Matrix formulation of the CST basis functions for upper surfaces
        Cx_upper = np.power(x_upper, n1) * np.power(1.0 - x_upper, n2)
        Bx_upper = np.power(x_upper[:, None], k) * np.power(
            1.0 - x_upper[:, None],
            n - k,
        )
        Mx_upper = Cx_upper[:, None] * Bx_upper * binom

        # Matrix formulation of the CST basis functions for lower surfaces
        Cx_lower = np.power(x_lower, n1) * np.power(1.0 - x_lower, n2)
        Bx_lower = np.power(x_lower[:, None], k) * np.power(
            1.0 - x_lower[:, None],
            n - k,
        )
        Mx_lower = Cx_lower[:, None] * Bx_lower * binom

        # Add columns for Kulfan modifiers to solve the combined system.
        # Upper and lower curve must be fit simultaneously to solve for shared
        # modifier parameters, which must be identical for both sides.
        if use_kulfan_modifiers:
            le_mod_upper = x_upper * np.power(1.0 - x_upper, n + 0.5)
            le_mod_lower = x_lower * np.power(1.0 - x_lower, n + 0.5)

            te_mod_upper = x_upper / 2.0
            te_mod_lower = -x_lower / 2.0

            if trailing_edge_solution == "data":
                te_thickness = max(float(y_upper[-1] - y_lower[-1]), 0.0)
                y_upper_target = y_upper - te_mod_upper * te_thickness
                y_lower_target = y_lower - te_mod_lower * te_thickness
            else:
                te_thickness = None
                y_upper_target = y_upper
                y_lower_target = y_lower

            zeros_upper = np.zeros((x_upper.size, n_coefficients))
            zeros_lower = np.zeros((x_lower.size, n_coefficients))

            M_upper = np.hstack(
                [
                    Mx_upper,
                    zeros_upper,
                    le_mod_upper[:, None],
                ]
            )
            M_lower = np.hstack(
                [
                    zeros_lower,
                    Mx_lower,
                    le_mod_lower[:, None],
                ]
            )

            if trailing_edge_solution == "fit":
                M_upper = np.hstack([M_upper, te_mod_upper[:, None]])
                M_lower = np.hstack([M_lower, te_mod_lower[:, None]])

            M = np.vstack([M_upper, M_lower])
            y = np.concatenate([y_upper_target, y_lower_target])
            solution, *_ = np.linalg.lstsq(M, y, rcond=rcond)

            upper_coeffs = solution[:n_coefficients]
            lower_coeffs = solution[n_coefficients : 2 * n_coefficients]

            if trailing_edge_solution == "fit":
                le_weight = float(solution[-2])
                te_thickness = max(float(solution[-1]), 0.0)
            else:
                le_weight = float(solution[-1])
                te_thickness = float(te_thickness)

            upper_coeffs[0] = max(upper_coeffs[0], eps)
            lower_coeffs[0] = min(lower_coeffs[0], -eps)

            upper_curve = KulfanModifiedCST(
                coefficients=upper_coeffs,
                leading_edge_weight=le_weight,
                trailing_edge_thickness=te_thickness,
                surface_type="upper",
                n1=n1,
                n2=n2,
            )
            lower_curve = KulfanModifiedCST(
                coefficients=lower_coeffs,
                leading_edge_weight=le_weight,
                trailing_edge_thickness=te_thickness,
                surface_type="lower",
                n1=n1,
                n2=n2,
            )
        # else, if the vanilla version is used without modifiers, just use the
        # standard CST fit for upper and lower surfaces separately. These are
        # then turned into modified curves with the modifiers set to 0. There is
        # no real reason for this conversion apart from potential compatibility
        # issues, but the curve itself remains the same.
        else:
            upper_base = CSTCurve.fit(
                upper_points,
                num_coefficients=n_coefficients,
                n1=n1,
                n2=n2,
                rcond=rcond,
            )
            lower_base = CSTCurve.fit(
                lower_points,
                num_coefficients=n_coefficients,
                n1=n1,
                n2=n2,
                rcond=rcond,
            )
            upper_base.coefficients[0] = max(upper_base.coefficients[0], eps)
            lower_base.coefficients[0] = min(lower_base.coefficients[0], -eps)

            upper_curve = KulfanModifiedCST(
                coefficients=upper_base.coefficients,
                leading_edge_weight=0.0,
                trailing_edge_thickness=0.0,
                surface_type="upper",
                n1=n1,
                n2=n2,
            )
            lower_curve = KulfanModifiedCST(
                coefficients=lower_base.coefficients,
                leading_edge_weight=0.0,
                trailing_edge_thickness=0.0,
                surface_type="lower",
                n1=n1,
                n2=n2,
            )

        airfoil = cls(
            upper_surface=upper_curve,
            lower_surface=lower_curve,
            name=name,
            description=description,
        )
        airfoil.data_points = np.vstack([upper_points[::-1], lower_points[1:]])
        return airfoil

    @classmethod
    def from_coordinate_array(
        cls,
        points: np.ndarray,
        normalize: bool = True,
        name: str = "",
        description: str = "",
        n_coefficients: int = 8,
        trailing_edge_solution: Literal["fit", "data"] = "data",
        n1: float = 0.5,
        n2: float = 1.0,
    ) -> "KulfanAirfoil":
        """Fit a Kulfan airfoil from full airfoil coordinates.

        The method optionally normalizes the airfoil (Default: True), splits at
        the leading edge, and then performs upper/lower fitting.

        Math:
            full points -> (upper, lower) -> fit upper/lower Kulfan surfaces

        Args:
            points (np.ndarray): Full airfoil points [N, 2]. normalize (bool):
            If True, normalize and re-sample before fitting. name (str): Airfoil
            short name. description (str): Optional textual description.
            n_coefficients (int): Number of coefficients per side.
            trailing_edge_solution (Literal["fit", "data"]): TE strategy. n1
            (float): Leading-edge class exponent. n2 (float): Trailing-edge
            class exponent.

        Returns:
            KulfanAirfoil: Fitted airfoil object.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape [N, 2]")

        if normalize:
            spline = AirfoilNormalizer.normalized_bspline(pts)
            u_le = spline.u_leading_edge
            upper_points = spline.evaluate_at(np.linspace(0.0, u_le, 200))[::-1]
            lower_points = spline.evaluate_at(np.linspace(u_le, 1.0, 200))
        else:
            le_idx = int(np.argmin(pts[:, 0]))
            if le_idx <= 0 or le_idx >= len(pts) - 1:
                raise ValueError("Could not split points into upper/lower surfaces")
            upper_points = pts[: le_idx + 1][::-1]
            lower_points = pts[le_idx:]

        airfoil = cls.fit(
            upper_points=upper_points.round(6),
            lower_points=lower_points.round(6),
            n_coefficients=n_coefficients,
            trailing_edge_solution=trailing_edge_solution,
            n1=n1,
            n2=n2,
            name=name,
            description=description,
        )
        airfoil.data_points = pts
        return airfoil

    @classmethod
    def from_file(
        cls,
        filepath: str,
        normalize: bool = True,
        data_type: str = "coordinates",
        **kwargs: dict,
    ) -> "KulfanAirfoil":
        """Create a Kulfan airfoil from an airfoil data file.

        This method reads coordinates with `AirfoilDataFile` and delegates to
        `from_coordinate_array`.

        Math:
            file -> points -> optional normalization -> Kulfan fit

        Args:
            filepath (str): Path to input airfoil file.
            normalize (bool): Whether to normalize before fitting.
            data_type (str): Only "coordinates" is supported.
            **kwargs (dict): Extra keyword arguments forwarded to fitting.

        Returns:
            KulfanAirfoil: Fitted airfoil object from file data.
        """
        if data_type != "coordinates":
            raise ValueError("KulfanAirfoil only supports data_type='coordinates'")

        data = AirfoilDataFile(filepath)
        return cls.from_coordinate_array(
            data.points,
            normalize=normalize,
            name=data.filename,
            description=data.header,
            **kwargs,
        )

    @property
    def upper_surface(self) -> KulfanModifiedCST:
        """Return the upper Kulfan-modified CST surface.

        Math:
            y_u(x) = upper_surface(x)

        Args:
            None.

        Returns:
            KulfanModifiedCST: Upper surface model.
        """
        return self._upper_surface

    @property
    def lower_surface(self) -> KulfanModifiedCST:
        """Return the lower Kulfan-modified CST surface.

        Math:
            y_l(x) = lower_surface(x)

        Args:
            None.

        Returns:
            KulfanModifiedCST: Lower surface model.
        """
        return self._lower_surface

    @property
    def parameters(self) -> np.ndarray:
        """Return flattened Kulfan parameter vector for this airfoil.

        Math:
            p = [a_u, a_l, w_le, t_te]

        Args:
            None.

        Returns:
            np.ndarray: Flat parameter vector with shape [2K + 2].
        """
        return np.concatenate(
            [
                self.upper_surface.coefficients,
                self.lower_surface.coefficients,
                np.array(
                    [
                        self.upper_surface.leading_edge_weight,
                        self.upper_surface.trailing_edge_thickness,
                    ]
                ),
            ]
        )

    params = parameters

    @property
    def batch_size(self) -> int:
        """Return pseudo-batch size for API compatibility.

        This NumPy class represents one airfoil only.

        Math:
            B = 1

        Args:
            None.

        Returns:
            int: Always 1.
        """
        return 1

    @property
    def is_batched(self) -> bool:
        """Indicate whether multiple airfoils are represented.

        Math:
            is_batched = (B > 1)

        Args:
            None.

        Returns:
            bool: Always False for this class.
        """
        return False

    def __len__(self) -> int:
        """Return number of represented airfoils.

        Math:
            len(self) = 1

        Args:
            None.

        Returns:
            int: Always 1.
        """
        return 1

    def evaluate_at(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate upper and lower ordinates at chord locations x.

        This follows the geometry-module convention of exposing an explicit
        `evaluate_at` method rather than a torch-style `forward` entry point.

        Math:
            y_u = upper_surface.evaluate_at(x)[:, 1]
            y_l = lower_surface.evaluate_at(x)[:, 1]

        Args:
            x (np.ndarray): Chordwise locations in [0, 1], shape [N].

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_upper, y_lower), each shape [N].
        """
        x = ensure_1d_vector(x)
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("x must be in [0, 1]")
        y_upper = self.upper_surface.evaluate_at(x)[:, 1]
        y_lower = self.lower_surface.evaluate_at(x)[:, 1]
        return y_upper, y_lower

    def coordinates_at(self, x: np.ndarray, dtype=np.float64) -> np.ndarray:
        """Return Selig-format coordinates sampled at x locations.

        The order is TE->LE on the upper side and LE->TE on the lower side,
        excluding duplicate leading-edge point.

        Math:
            P = [ (x, y_u)[::-1], (x, y_l)[1:] ]

        Args:
            x (np.ndarray): Chordwise locations in [0, 1], shape [N].
            dtype (np.dtype): Output floating dtype.

        Returns:
            np.ndarray: Coordinates with shape [2N-1, 2].
        """
        x = ensure_1d_vector(x)
        y_upper, y_lower = self.evaluate_at(x)
        return np.vstack(
            [
                np.column_stack([x, y_upper])[::-1],
                np.column_stack([x, y_lower])[1:],
            ]
        ).astype(dtype).view(Point2D)

    @property
    def points(self) -> np.ndarray:
        """Return default cosine-sampled Selig-format coordinates.

        Uses 100 points per side (199 total after LE de-duplication).

        Math:
            x_i = cosine_spacing(0, 1, 100)
            points = coordinates_at(x_i)

        Args:
            None.

        Returns:
            np.ndarray: Coordinate array with shape [199, 2].
        """
        return self.coordinates_at(cosine_spacing(0, 1, num=100), dtype=np.float64)

    def upper_surface_at(self, x: np.ndarray) -> np.ndarray:
        """Evaluate upper surface ordinate values at x.

        Math:
            y_u(x) = upper_surface(x)

        Args:
            x (np.ndarray): Chordwise locations in [0, 1].

        Returns:
            np.ndarray: Upper ordinates with shape [N].
        """
        x = ensure_1d_vector(x)
        return self.upper_surface.evaluate_at(x)[:, 1]

    def lower_surface_at(self, x: np.ndarray) -> np.ndarray:
        """Evaluate lower surface ordinate values at x.

        Math:
            y_l(x) = lower_surface(x)

        Args:
            x (np.ndarray): Chordwise locations in [0, 1].

        Returns:
            np.ndarray: Lower ordinates with shape [N].
        """
        x = ensure_1d_vector(x)
        return self.lower_surface.evaluate_at(x)[:, 1]

    @property
    def surface(self) -> CSTAirfoilSurface:
        """Return compatibility perimeter wrapper around both surfaces.

        Math:
            surface(u) switches between y_u and y_l via x(u)=|1-2u|

        Args:
            None.

        Returns:
            CSTAirfoilSurface: Parametric wrapper object.
        """
        return CSTAirfoilSurface(self.upper_surface, self.lower_surface)

    def thickness_at(self, x: np.ndarray) -> np.ndarray:
        """Compute thickness distribution at x.

        Math:
            t(x) = y_u(x) - y_l(x)

        Args:
            x (np.ndarray): Chordwise locations in [0, 1].

        Returns:
            np.ndarray: Thickness values with shape [N].
        """
        y_upper, y_lower = self.evaluate_at(x)
        return y_upper - y_lower

    def camber_at(self, x: np.ndarray) -> np.ndarray:
        """Compute camber line ordinate at x.

        Math:
            c(x) = (y_u(x) + y_l(x)) / 2

        Args:
            x (np.ndarray): Chordwise locations in [0, 1].

        Returns:
            np.ndarray: Camber values with shape [N].
        """
        y_upper, y_lower = self.evaluate_at(x)
        return 0.5 * (y_upper + y_lower)

    @property
    def thickness_distribution(self) -> KulfanModifiedCST:
        """Fit a single-surface Kulfan curve to thickness data.

        Thickness is always non-negative in nominal airfoils, so the fitted
        helper curve is treated as an "upper" surface convention.

        Math:
            t(x) = y_u(x) - y_l(x)
            fit KulfanModifiedCST to [x, t(x)]

        Args:
            None.

        Returns:
            KulfanModifiedCST: Fitted thickness curve model.
        """
        x = cosine_spacing(0, 1, num=200)
        thickness_points = np.column_stack([x, self.thickness_at(x)])
        return KulfanModifiedCST.fit(
            thickness_points,
            n_coefficients=self.upper_surface.n_coefficients,
            trailing_edge_solution="data",
            surface_type="upper",
            n1=self.upper_surface.n1,
            n2=self.upper_surface.n2,
        )

    @property
    def camber_line(self) -> CSTCurve:
        """Fit a CST curve to sampled camber data.

        A neutral class function (n1=n2=1) is used for smooth camber fitting.

        Math:
            c(x) = (y_u + y_l)/2
            fit CSTCurve to [x, c(x)]

        Args:
            None.

        Returns:
            CSTCurve: Fitted camber-line curve.
        """
        x = cosine_spacing(0, 1, num=200)
        camber_points = np.column_stack([x, self.camber_at(x)])
        return CSTCurve.fit(
            camber_points,
            num_coefficients=self.upper_surface.n_coefficients,
            n1=1.0,
            n2=1.0,
        )

    @property
    def max_thickness(self) -> Tuple[float, float]:
        """Return maximum thickness and its x-location.

        The maximum is found by dense sampling and argmax.

        Math:
            (t_max, x_t) = max_x t(x)

        Args:
            None.

        Returns:
            Tuple[float, float]: (t_max, x_at_t_max).
        """
        x = np.linspace(0.0, 1.0, 2000)
        t = self.thickness_at(x)
        idx = int(np.argmax(t))
        return float(t[idx]), float(x[idx])

    @property
    def max_camber(self) -> Tuple[float, float]:
        """Return signed maximum camber magnitude and its x-location.

        The extremum is located by maximizing absolute camber and preserving
        the sampled sign at that location.

        Math:
            x_c = argmax_x |c(x)|,
            c_max = c(x_c)

        Args:
            None.

        Returns:
            Tuple[float, float]: (c_max_signed, x_at_c_max).
        """
        x = np.linspace(0.0, 1.0, 2000)
        c = self.camber_at(x)
        idx = int(np.argmax(np.abs(c)))
        return float(c[idx]), float(x[idx])

    @property
    def leading_edge_radius(self) -> float:
        """Estimate leading-edge radius from surface curvature.

        The estimate averages upper/lower curvature magnitudes near x=0.

        Math:
            R_le ~ 1 / (0.5*(|kappa_u| + |kappa_l|))

        Args:
            None.

        Returns:
            float: Estimated leading-edge radius (inf for zero curvature).
        """
        x_le = np.array([1e-8])
        kappa_u = abs(float(self.upper_surface.curvature_at(x_le)[0]))
        kappa_l = abs(float(self.lower_surface.curvature_at(x_le)[0]))
        kappa_avg = 0.5 * (kappa_u + kappa_l)
        return np.inf if kappa_avg < self.eps else 1.0 / kappa_avg


    @property
    def trailing_edge_angle(self) -> float:
        """Compute trailing-edge included angle in degrees.

        Math:
            theta_te = |atan(y'_u) - atan(y'_l)| * 180/pi

        Args:
            None.

        Returns:
            float: Trailing-edge angle in degrees.
        """
        x_te = np.array([0.99])
        dy_upper = float(self.upper_surface.first_deriv_at(x_te)[0])
        dy_lower = float(self.lower_surface.first_deriv_at(x_te)[0])
        return abs(np.arctan(dy_upper) - np.arctan(dy_lower)) * 180.0 / np.pi

    @property
    def trailing_edge_thickness(self) -> float:
        """Return trailing-edge thickness at x=1.

        Math:
            t_te = y_u(1) - y_l(1)

        Args:
            None.

        Returns:
            float: Trailing-edge thickness.
        """
        return float(self.thickness_at(np.array([1.0]))[0])

    @property
    def trailing_edge_wedge_angle(self) -> float:
        """Compute wedge angle near the trailing edge in degrees.

        This uses derivatives at x=1-eps to reduce endpoint singular effects.

        Math:
            gamma_te = |atan(y'_u) - atan(y'_l)| * 180/pi

        Args:
            None.

        Returns:
            float: Trailing-edge wedge angle in degrees.
        """
        x_te = np.array([1.0 - 1e-8])
        dy_upper = float(self.upper_surface.first_deriv_at(x_te)[0])
        dy_lower = float(self.lower_surface.first_deriv_at(x_te)[0])
        return abs(np.arctan(dy_upper) - np.arctan(dy_lower)) * 180.0 / np.pi

    @property
    def upper_crest(self) -> Tuple[float, float]:
        """Return upper crest location and ordinate.

        The crest is approximated by dense sampling and argmax on upper y.

        Math:
            x_z_u = argmax_x y_u(x), y_z_u = y_u(x_z_u)

        Args:
            None.

        Returns:
            Tuple[float, float]: (x_crest, y_crest) for upper surface.
        """
        x = np.linspace(0.0, 1.0, 2000)
        y = self.upper_surface_at(x)
        idx = int(np.argmax(y))
        return float(x[idx]), float(y[idx])

    @property
    def lower_crest(self) -> Tuple[float, float]:
        """Return lower crest location and ordinate.

        The crest is approximated by dense sampling and argmin on lower y.

        Math:
            x_z_l = argmin_x y_l(x), y_z_l = y_l(x_z_l)

        Args:
            None.

        Returns:
            Tuple[float, float]: (x_crest, y_crest) for lower surface.
        """
        x = np.linspace(0.0, 1.0, 2000)
        y = self.lower_surface_at(x)
        idx = int(np.argmin(y))
        return float(x[idx]), float(y[idx])

    @property
    def upper_crest_curvature(self) -> float:
        """Return curvature at the upper crest.

        Math:
            kappa_z_u = kappa_u(x_z_u)

        Args:
            None.

        Returns:
            float: Upper crest curvature.
        """
        x_crest, _ = self.upper_crest
        return float(self.upper_surface.curvature_at(np.array([x_crest]))[0])

    @property
    def lower_crest_curvature(self) -> float:
        """Return curvature at the lower crest.

        Math:
            kappa_z_l = kappa_l(x_z_l)

        Args:
            None.

        Returns:
            float: Lower crest curvature.
        """
        x_crest, _ = self.lower_crest
        return float(self.lower_surface.curvature_at(np.array([x_crest]))[0])

    @property
    def area(self) -> float:
        """Return the enclosed airfoil area from base polygon moments.

        This override keeps the Kulfan-specific API location while delegating
        to the shared geometric-moment implementation in :class:`AirfoilBase`.

        Returns:
            float: Positive enclosed area, units length^2.
        """
        return float(super().area)

    def _validate_thickness(self) -> bool:
        """Check sampled thickness non-negativity.

        The check uses cosine spacing to emphasize LE/TE resolution.

        Math:
            valid iff min_x t(x) >= 0 over sampled nodes

        Args:
            None.

        Returns:
            bool: True when sampled thickness is non-negative.
        """
        beta = np.linspace(0.0, np.pi, 512)
        x = 0.5 * (1.0 - np.cos(beta))
        return bool(np.all(self.thickness_at(x) >= 0.0))

    def plot(
        self,
        title: Optional[str] = None,
        num_points: int = 2000,
        save_dir: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot upper and lower Kulfan surfaces with parameter annotations.

        A dense chordwise grid is used for visualization quality. The parameter
        textbox displays fitted Kulfan coefficients and shared modifiers.

        Math:
            plot y_u(x), y_l(x) for x in [0, 1]

        Args:
            title (Optional[str]): Optional figure title.
            num_points (int): Number of x samples for plotting.
            save_dir (Optional[str]): Optional output image path.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects.
        """
        x = np.linspace(0.0, 1.0, num_points)
        y_upper, y_lower = self.evaluate_at(x)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y_upper, "b-", label="Upper Kulfan Surface")
        ax.plot(x, y_lower, "r-", label="Lower Kulfan Surface")
        ax.set_title(title or "Kulfan Airfoil")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend(loc="best")

        upper_coeffs = np.round(self.upper_surface.coefficients, 4)
        lower_coeffs = np.round(self.lower_surface.coefficients, 4)
        w_le = self.upper_surface.leading_edge_weight
        t_te = self.upper_surface.trailing_edge_thickness

        row_fmt = "{:<14} " + " ".join(["{:>8.4f}"] * len(upper_coeffs))
        text = (
            row_fmt.format("Upper coeffs:", *upper_coeffs)
            + "\n"
            + row_fmt.format("Lower coeffs:", *lower_coeffs)
            + "\n"
            + f"{'LE weight:':<15} {w_le:>8.4f}\n"
            + f"{'TE thickness:':<14} {t_te:>8.4f}"
        )

        plt.subplots_adjust(bottom=0.245, top=0.925)
        plt.figtext(
            0.1,
            0.0275,
            text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white"),
            linespacing=1.5,
        )

        if save_dir is not None:
            fig.savefig(save_dir)

        return fig, ax


class CSTAirfoil(KulfanAirfoil):
    """Backward-compatible alias for the NumPy Kulfan airfoil implementation.

    This class intentionally adds no behavior and exists only so legacy code
    importing `CSTAirfoil` continues to function with the updated Kulfan/CST
    representation.

    Math:
        CSTAirfoil == KulfanAirfoil

    Args:
        Same as `KulfanAirfoil`.

    Returns:
        KulfanAirfoil: Behaviorally identical instance.
    """


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

    @cached_property
    def naca_camber_curve(self):
        class Camberline(Curve):
            def __init__(self, m, p):
                self.max_camber = m
                self.camber_location = p

            @property
            def cambered(self) -> bool:
                """Returns if the current :py:class:`Airfoil` is cambered."""
                return self.max_camber != 0 and self.camber_location != 0

            def evaluate_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
                """Returns camber-line points at the supplied ``x``.
                Args:
                    x: Chord-line fractions (0 = LE, 1 = TE), length n
                Returns:
                    np.ndarray: Camber-line points at x, with shape (n, 2)
                """
                # Setting up chord-line and camber-line point arrays
                x = ensure_1d_vector(x)
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

            def first_deriv_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
                """Returns the camber-line tangent vector at supplied ``x``."""
                # Setting up chord-line and camber-line tangent arrays
                x = ensure_1d_vector(x)
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

                return t_c

            def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
                return self.evaluate_at(x)

        return Camberline(self.max_camber, self.camber_location)

    def camber_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Returns camber-line ordinates at the supplied ``x``."""
        return self.camber_line.evaluate_at(x)[:, 1]

    @cached_property
    def naca_thickness_distribution(self):
        class ThicknessDistribution(Curve):
            def __init__(self, t, te):
                self.max_thickness = t
                self.te_closed = te

            def half_thickness_at(self, x: np.ndarray) -> np.ndarray:
                """Calculates the NACA-4 series 'Half-Thickness' y_t at ``x``.

                Args:
                    x: Chord-line fraction (0 = LE, 1 = TE)
                Returns:
                    y_t: Half-thickness at x.
                """
                x = ensure_1d_vector(x)
                return (self.max_thickness / 0.2) * (
                    0.2969 * np.sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * (x ** 2)
                    + 0.2843 * (x ** 3)
                    - (0.1036 if self.te_closed else 0.1015) * (x ** 4)
                )

            def evaluate_at(self, x: Union[float, np.ndarray]) -> np.ndarray:
                """Returns thickness distribution points at the supplied ``x``.
                Args:
                    x: Chord-line fractions (0 = LE, 1 = TE), length n
                Returns:
                    np.ndarray: Thickness distribution points at x, with shape (n, 2)
                """
                x = ensure_1d_vector(x)
                pts_t = np.zeros((x.size, 2))
                pts_t[:, 0] = x
                pts_t[:, 1] = self.half_thickness_at(x) * 2
                return pts_t

            def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
                return self.evaluate_at(x)

        return ThicknessDistribution(self.max_thickness, self.te_closed)

    def offset_vectors_at(self, x: np.ndarray) -> np.ndarray:
        """Returns half-thickness magnitude vectors at ``x``."""
        n_c = self.naca_camber_curve.normal_at(x)  # Camber normal-vectors
        y_t = self.naca_thickness_distribution.half_thickness_at(x)  # Half thicknesses
        return np.multiply(n_c, y_t.reshape(x.size, 1), out=n_c)

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
    def parse_naca_code(naca_code: str) -> Tuple[float, float, float]:
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
    @cached_property
    def upper_surface(self) -> ParametricCurve:
        """Returns the upper surface curve of the airfoil as Bspline."""
        x = cosine_spacing(0, 1, num=200)
        return BSpline2D(
            self.camber_line.evaluate_at(x) + self.offset_vectors_at(x)
        )

    @cached_property
    def lower_surface(self) -> ParametricCurve:
        """Returns the lower surface curve of the airfoil as Bspline."""
        x = cosine_spacing(0, 1, num=200)
        return BSpline2D(
            self.camber_line.evaluate_at(x) - self.offset_vectors_at(x)
        )

    @cached_property
    def surface(self)-> ParametricCurve:
        x = cosine_spacing(0, 1, num=200)
        return BSpline2D(
            np.vstack([
                (self.camber_line.evaluate_at(x) + self.offset_vectors_at(x))[::-1],
                (self.camber_line.evaluate_at(x) - self.offset_vectors_at(x))[1:]
            ])
        )

    @cached_property
    def camber_line(self) -> ParametricCurve:
        """Returns the camber line curve of the airfoil as Bspline."""
        return BSpline2D(
            self.naca_camber_curve.evaluate_at(cosine_spacing(0, 1, num=200))
        )

    @cached_property
    def thickness_distribution(self) -> ParametricCurve:
        """Returns the thickness distribution curve of the airfoil as Bspline."""
        return BSpline2D(
            self.naca_thickness_distribution.evaluate_at(cosine_spacing(0, 1, num=200))
        )



