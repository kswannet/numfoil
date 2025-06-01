import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as si
import scipy.optimize as opt
import scipy.integrate as spi

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
            lambda u: -self.camber_line.evaluate_at(u)[0][1], 0.5, bounds=[(0, 1)]
        )
        if not result.success:
            print(result)
            raise RuntimeError("Failed to find upper crest.")
        return self.camber_line.evaluate_at(result.x[0])

    @cached_property
    def area(self) -> float:
        """Calculates the area of the airfoil."""
        x = cosine_spacing(0, 1, num=1000)
        t = self.thickness_at(x)
        return np.trapz(t, x)
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
    def trailing_edge_gap(self) -> float:
        """Returns the gap between the upper and lower surfaces at the trailing edge."""
        return np.abs(
            self.upper_surface_at(1) - self.lower_surface_at(1)
        )

    @cached_property
    def trailing_edge_upper_vector(self) -> np.ndarray:
        """Upper surface gradient at the trailing edge."""
        return self.upper_surface.first_deriv_at(1)

    @cached_property
    def trailing_edge_lower_vector(self) -> np.ndarray:
        """Lower surface gradient at the trailing edge."""
        return self.lower_surface.first_deriv_at(1)

    @cached_property
    def trailing_edge_vector(self) -> np.ndarray:
        """Vector between the upper and lower surface at the trailing edge."""
        return self.camber_line.tangent_at(1)[0]

    @cached_property
    def leading_edge_vector(self) -> np.ndarray:
        """Vector between the upper and lower surface at the leading edge."""
        return self.camber_line.first_deriv_at(0)

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
        """Returns sampled points of the airfoil surface."""
        x = cosine_spacing(0, 1, num=100)
        return np.vstack([
            np.column_stack([x, self.upper_surface_at(x)])[::-1],
            np.column_stack([x, self.lower_surface_at(x)])[1:],
        ])

    def plot(self, n_points=1000):
        """Plots the airfoil geometry."""
        x = cosine_spacing(0,1, num=n_points)
        fig, ax = plt.subplots()
        ax.plot(x, self.upper_surface_at(x), label="Upper Surface")
        ax.plot(x, self.lower_surface_at(x), label="Lower Surface")
        ax.plot(x, self.camber_at(x), label="Camber Line")
        ax.set_title(self.description or self.name or "Airfoil")
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
        name: str = None,
        description: str = None,
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
        self.description = description.replace(' AIRFOIL', '') or name

        self.u_leading_edge = 0.5

    @classmethod
    def from_coordinate_array(
        cls,
        points: np.ndarray,
        name: str = None,
        description: str = None,
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
            AirfoilNormalizer.normalize(points) if normalize else BSpline2D(points),
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
        name: str = None,
        description: str = None
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

        if isinstance(camber_curve, np.ndarray) and isinstance(thickness_curve, np.ndarray) \
            and camber_curve[:,1] != thickness_curve[:,1]:
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
        # force final control point to bet at x=1
        adjusted_control_points = curve.control_points
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
        # force final control point to bet at x=1
        adjusted_control_points = curve.control_points
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
        name: str = None,
        description: str = None,
    ):
        # save the surface spline object
        self.surface_curve = surface_curve

        # the shortened name of the airfoil, usually the filename
        self.name = name
        # the full name of the airfoil, usually from the file header
        self.description = description.replace(' AIRFOIL', '') if description is not None else name

        self.u_leading_edge = 0.5

    @classmethod
    def from_control_points(
        cls,
        control_points: np.ndarray,
        name: str = None,
        description: str = None,
        # normalize: bool = True,
        **kwargs
    ):
        """Creates an Airfoil object from control points.

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
            # data_points=control_points,
        )

    @classmethod
    def from_coordinate_array(
        cls,
        points: np.ndarray,
        name: str = None,
        description: str = None,
        normalize: bool = True,
        fit_method: str = "split_u_l",
        n_control_points: int = None,
        control_point_spacing=cosine_spacing(0, 1, 13)[:-1],
        end_clamp: str | np.ndarray = "data",
        trailing_edge_thickness: float | None = None,
        damping_type: str = "deriv",
        w_damping: float = 1e-1,
        find_trailing_edge: bool = True,
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

            fit_method (str):
                Method to fit the airfoil. Options are:
                - "full": Fit the entire curve at once.
                - "split_u_l": Split the points into upper and lower and fit
                    those separately. Preferred method as it is much faster.
                - "split_t_c": First determine camber and thickness values, then
                    fit bezier curves to those.
                Defaults to "split_u_l".
                The main reason for this is to allow for different constraints,
                and fitting two curves separately is much faster than fitting
                one big one.

            kwargs (dict):
                Additional arguments for the fitting method.


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

        # To improve the fitting of the new spline, points are resampled from
        # the normalized bspline
        if normalize:
            normalized_bspline = AirfoilNormalizer.normalize(points, find_trailing_edge=find_trailing_edge)
            points = AirfoilNormalizer._remove_overshoots(
                normalized_bspline.evaluate_at(
                    np.hstack([
                        cosine_spacing(0, normalized_bspline.u_leading_edge, num=100),
                        cosine_spacing(normalized_bspline.u_leading_edge, 1, num=100)[1:],
                    ])
                )
            )

        # TODO : clean this up, there must be a better way than this mess...
        match fit_method:
            case "split_u_l":
                # split the points in upper and lower and fit those seperatly
                # prefered method as it is much faster
                upper = SplevBezier.fit(
                    points[len(points)//2::-1],
                    n_control_points=n_control_points,
                    spacing=control_point_spacing,
                    start_clamp='origin',
                    end_clamp=end_clamp if trailing_edge_thickness is None else np.array([1.0, trailing_edge_thickness/2]),
                    damping_type=damping_type,
                    w_damping=w_damping,
                    constraints=[
                        # {   # ensure all control points have positive y-values
                        #     "type": "ineq",
                        #     "fun": lambda y: y
                        # },
                        {   # force y>0.005 for first control point after the LE
                            "type": "ineq",
                            "fun": lambda y: y[0] - 0.002
                        },
                        {   # ensure rounded leading edge (see GOE440)
                            "type": "ineq",
                            "fun": lambda y: y[1] - y[0]*0.5
                        },
                    ]
                )
                lower = SplevBezier.fit(
                    points[len(points)//2:],
                    n_control_points=n_control_points,
                    spacing=control_point_spacing,
                    start_clamp='origin',
                    end_clamp=end_clamp if trailing_edge_thickness is None else np.array([1.0, -trailing_edge_thickness/2]),
                    damping_type=damping_type,
                    w_damping=w_damping,
                    constraints=[
                        # {   # ensure all control points have negative y-values
                        #     "type": "ineq",
                        #     "fun": lambda y: -y
                        # },
                        {   # force y>0.005 for first control point after the LE
                            "type": "ineq",
                            "fun": lambda y: -y[0] + 0.002
                        },
                        {   # ensure rounded leading edge (see GOE440)
                            "type": "ineq",
                            "fun": lambda y: -y[1] + y[0]*0.5
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
                thickness_pts = bspline_airfoil.thickness_distribution.evaluate_at(
                    cosine_spacing(0, 1, num=200)
                )
                thickness_distribution = SplevBezier.fit(
                    thickness_pts,
                    n_control_points=n_control_points,
                    spacing=control_point_spacing,
                    start_clamp='origin',
                    end_clamp=end_clamp,
                    damping_type=damping_type,
                    w_damping=w_damping,
                    constraints=[
                        {   # ensure all control points have positive y-values
                            "type": "ineq",
                            "fun": lambda y: y
                        },
                        # {   # ensure rounded leading edge (see GOE440)
                        #     "type": "ineq",
                        #     "fun": lambda y: y[0] - 0.005
                        # },
                        # {   # ensure rounded leading edge (see GOE440)
                        #     "type": "ineq",
                        #     "fun": lambda y: y[1] - y[0]*0.5
                        # },
                    ]
                )
                camber_pts = bspline_airfoil.camber_line.evaluate_at(
                    cosine_spacing(0, 1, num=200)
                )
                camber_curve = SplevBezier.fit(
                    camber_pts,
                    n_control_points=n_control_points,
                    spacing=control_point_spacing,
                    start_clamp='origin',
                    end_clamp=end_clamp,
                    damping_type=damping_type,
                    w_damping=w_damping,
                    constraints=[
                        # {
                        #     "type": "ineq",
                        #     "fun": lambda y: y[0] + 0.002
                        # },
                        # {
                        #     "type": "ineq",
                        #     "fun": lambda y: 0.002 - y[0]
                        # },
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
                # Use the composite bezier class to fit the entire curve at once
                surfacespline = SplevCBezier.fit(
                    points,
                    n_control_points,
                    spacing=control_point_spacing,
                    w_damping=w_damping,
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

    @classmethod
    def from_camber_thickness(
        cls,
        thickness_curve: ParametricCurve | np.ndarray,
        camber_curve: ParametricCurve | np.ndarray,
        name: str = None,
        description: str = None,
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


class NACA4Airfoil(AirfoilBase):
    def __init__(self):
        pass
