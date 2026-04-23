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

"""Contains definitions for 2D points, vectors, and transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "GeometricMoments2D",
    "Point2D",
    "Vector2D",
    "is_row_vector",
    "magnitude_2d",
    "normalize_2d",
    "rotate_2d_90ccw",
]


class Geom2D(np.ndarray):
    """Defines the primitive row-vector as a geometric array."""

    def __new__(cls, array: Union[Sequence[Tuple[float, float]], np.ndarray]):
        """Creates a :py:class:`Geom2D` instance from ``array``.

        Args:
            array: A 2D Numpy array containing n row-vectors
        """
        array = np.array(array) if not isinstance(array, np.ndarray) else array
        assert is_row_vector(array)
        return np.asarray(array, dtype=np.float64).view(cls)

    @property
    def x(self) -> np.ndarray:
        """Returns the x coordinate(s) of :py:class:`Point2D`."""
        # return self[..., 0].view(np.ndarray)
        x = self[..., 0].view(np.ndarray)
        return x if x.size > 1 else x.item()

    @property
    def y(self) -> np.ndarray:
        """Returns the y coordinate(s) of :py:class:`Point2D`."""
        # return self[..., 1].view(np.ndarray)
        y = self[..., 1].view(np.ndarray)
        return y if y.ndim > 0 else y.item()


class Point2D(Geom2D):
    """Defines a point in 2D space."""

    _default_precision = 6  # Default number of decimals (rounding precision)

    def __new__(cls, array: Union[Sequence[Tuple[float, float]], np.ndarray]):
        """Creates a :py:class:`Point2D` instance from ``array`` with automatic rounding."""
        array = np.array(array) if not isinstance(array, np.ndarray) else array
        assert is_row_vector(array)
        return np.asarray(  # Round to class default precision
            np.round(array, decimals=cls._default_precision), dtype=np.float64
        ).view(cls)

    @classmethod
    def set_precision(cls, decimals: int):
        """Set the default precision for all Point2D instances."""
        cls._default_precision = decimals

    def __sub__(self, other) -> Vector2D:
        """Overloads subtract magic method to allow vector creation.

        When two :py:class:`Point2D` objects are subtracted from
        one another a vector is created as follows::

            >>> a = Point2D([0, 0])
            >>> b = Point2D([1, 1])
            >>> b - a
            Vector2D([1, 1])  # A Vector2D object is created from a to b

            >>> a - b
            Vector2D([-1, -1])  # A Vector2D from b to a

        However, if a simple scalar is added to the :py:class:`Point2D`
        object it will return a `Point2D` object as expected::

            >>> a = Point2D([0, 0])
            >>> a + 2
            Point2D([2, 2])
        """
        if isinstance(other, Point2D):
            return super().__sub__(other).view(Vector2D)
        else:
            return super().__sub__(other)

    def plot(self, fmt='--o', *args, **kwargs):
        """Plots the :py:class:`Point2D` object."""
        kwargs.setdefault('markersize', 3)       # Default marker size
        kwargs.setdefault('linewidth', 1)      # Default line width
        plt.figure()
        plt.plot(self.x, self.y, fmt, *args, **kwargs)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Point2D Array")
        plt.grid(True)
        plt.show()


class Vector2D(Geom2D):
    """Defines a vector in 2D space."""

    @property
    def magnitude(self) -> np.ndarray:
        """Calculates the length (magnitude) of py:class:`Vector2D`."""
        return magnitude_2d(self).view(np.ndarray)

    @property
    def normalized(self) -> np.ndarray:
        """Returns the unit-vector(s) of :py:class:`Vector2D`."""
        return normalize_2d(self)


class Angle(float):
    """Angle scalar with unit metadata, stored internally as radians."""

    __slots__ = ("unit",)

    def __new__(cls, value: float, unit: str = "rad"):
        u = cls._normalize_unit(unit)
        radians_value = float(value) if u == "rad" else float(np.deg2rad(value))
        obj = super().__new__(cls, radians_value)
        obj.unit = u
        return obj

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        u = str(unit).strip().lower()
        if u in ("rad", "radian", "radians"):
            return "rad"
        if u in ("deg", "degree", "degrees"):
            return "deg"
        raise ValueError("unit must be 'rad' or 'deg'")

    def to_rad(self) -> "Angle":
        """Return a new Angle displayed in radians."""
        return Angle(float(self), "rad")

    def to_degree(self) -> "Angle":
        """Return a new Angle displayed in degrees."""
        return Angle(np.rad2deg(float(self)), "deg")

    def __call__(self) -> float:
        """Return value in the current display unit."""
        return float(np.rad2deg(float(self))) if self.unit == "deg" else float(self)

    def unit_vector(self) -> Vector2D:
        """Return [cos(theta), sin(theta)] using internal radians."""
        theta = float(self)
        return Vector2D([np.cos(theta), np.sin(theta)])

    def __repr__(self) -> str:
        return f"Angle({self()}, '{self.unit}')"


R_90CCW_MATRIX = np.array([[0, 1], [-1, 0]])


def rotate_2d_90ccw(array: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Rotates 2D row-vector(s) 90 degrees counter-clockwise.

    Args:
        array: An array of row-vector(s) with shape (n, 2)
        inplace: If the normalization should be performed inplace
            on the ``array`` object.

    Returns:
        Transformed row-vector(s) with dimension (n, 2).
    """
    # should also work: np.column_stack([-tangents[:, 1], tangents[:, 0]])
    assert is_row_vector(array)
    return np.dot(a=array, b=R_90CCW_MATRIX, out=array if inplace else None)


def magnitude_2d(array: np.ndarray) -> np.ndarray:
    """Calculates the magnitude of 2D row-vector(s).

    Args:
        array: An array of row-vector(s) with shape (n, 2)

    Returns:
        A column-vector of size (n, 1) containing n magnitudes.
    """
    assert is_row_vector(array)
    if len(array.shape) < 2:  # array is a 1D vector
        return np.linalg.norm(array).reshape(1, 1)
    else:
        return np.linalg.norm(array, axis=1).reshape((array.shape[0], 1))


def normalize_2d(array: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Normalizes 2D row-vector(s) into unit-vector(s).

    Args:
        array: An array of row-vector(s) with shape (n, 2)
        inplace: If the normalization should be performed inplace
            on the ``array`` object.

    Returns:
        Normalized row-vector(s) with dimension (n, 2).
    """
    assert is_row_vector(array)
    return np.divide(
        array, magnitude_2d(array), out=array if inplace else None,
    )


def is_row_vector(array: np.ndarray) -> bool:
    """Returns ``True`` if ``array`` is contains 2D row-vector(s).

    Raises:
        ValueError: If an ``array`` is 3D or contains column-vector(s)
    """
    if len(array.shape) == 2 and array.shape[1] != 2:
        raise ValueError(
            "The input `array` must contain 2D row-vectors with shape (n, 2)"
        )
    return True



@dataclass
class GeometricMoments2D:
    """Container and calculator for planar geometric moments up to a given order.

    This class represents geometric moments of a 2D enclosed polygonal lamina
    with uniform density. Raw moments are stored directly, and central moments
    plus descriptor matrices are derived lazily.

    Mathematical definitions:
        Raw moment:
            M_pq = integral_A x^p y^q dA

        Central moment:
            mu_pq = integral_A (x - x_bar)^p (y - y_bar)^q dA

        Centroid:
            x_bar = M_10 / M_00,
            y_bar = M_01 / M_00

    Attributes:
        raw_moments (dict[tuple[int, int], float]):
            Mapping (p, q) -> M_pq for p + q up to ``max_total_order``.
        max_total_order (int):
            Maximum total polynomial order included in the stored moments.
    """

    raw_moments: dict[tuple[int, int], float]
    max_total_order: int = 4

    @classmethod
    def from_polygon(
        cls,
        points: np.ndarray,
        *,
        max_total_order: int = 4,
    ) -> "GeometricMoments2D":
        """Build geometric moments from an enclosed polygon contour.

        The contour can be clockwise or counter-clockwise. Orientation is
        handled internally so physical moments remain positive/consistent.
        Monomial integration over the closed polygon represented by
        :attr:`points` using oriented edge-wise triangle decomposition. Signed
        orientation is corrected so returned moments are physical (orientation
        independent).

        Args:
            points (np.ndarray): Polygon vertices with shape (N, 2).
            max_total_order (int): Highest total order p+q to compute.

        Returns:
            GeometricMoments2D: Moment container with raw moments populated.
        """
        polygon = np.asarray(points, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("Polygon points must have shape [N, 2].")
        if polygon.shape[0] < 3:
            raise ValueError("At least three polygon points are required.")
        if max_total_order < 0:
            raise ValueError("max_total_order must be non-negative.")

        x_i = polygon[:, 0]
        y_i = polygon[:, 1]
        x_next = np.roll(x_i, -1)
        y_next = np.roll(y_i, -1)

        edge_cross = x_i * y_next - x_next * y_i
        signed_area = 0.5 * float(np.sum(edge_cross))
        if np.isclose(signed_area, 0.0, atol=np.finfo(float).eps):
            raise ValueError(
                "Polygon encloses near-zero area; geometric moments are undefined."
            )

        orientation_sign = 1.0 if signed_area >= 0.0 else -1.0
        factorials = [math.factorial(i) for i in range(max_total_order + 3)]

        raw_moments: dict[tuple[int, int], float] = {}
        for p in range(max_total_order + 1):
            for q in range(max_total_order + 1 - p):
                denominator = factorials[p + q + 2]
                integral_value = 0.0

                for i in range(p + 1):
                    x_binomial = math.comb(p, i)
                    x_terms = np.power(x_i, i) * np.power(x_next, p - i)
                    for j in range(q + 1):
                        coefficient = x_binomial * math.comb(q, j)
                        simplex_weight = (
                            factorials[i + j]
                            * factorials[p + q - i - j]
                            / denominator
                        )
                        y_terms = np.power(y_i, j) * np.power(y_next, q - j)
                        monomial_sum = float(np.sum(edge_cross * x_terms * y_terms))
                        integral_value += coefficient * simplex_weight * monomial_sum

                raw_moments[(p, q)] = orientation_sign * integral_value

        return cls(raw_moments=raw_moments, max_total_order=max_total_order)

    @property
    def area(self) -> float:
        """Return the enclosed airfoil area (zeroth geometric moment).

        This property treats the airfoil as a planar lamina of uniform areal
        density and returns the true enclosed area of the closed contour used by
        :attr:`points`.

        Mathematical definition:
            M_00 = integral_A 1 dA

        where A is the enclosed 2D airfoil region.

        Computation details:
            The contour is interpreted as a closed polygon and integrated using
            exact polygon-moment formulas (triangle decomposition / Green's
            theorem equivalent). Internally, signed area is used to handle
            clockwise or counter-clockwise orientation, and the returned value
            is always physically positive.

        Geometric interpretation:
            This is the sectional area of the airfoil shape itself, not a
            point-cloud statistic.

        Airfoil interpretation guidance:
            For normalized airfoils (unit chord), this value is dimensionless in
            code but scales physically as length^2.

        Returns:
            float: Positive enclosed area, units of length^2.
        """
        return float(self.raw_moments[(0, 0)])

    @property
    def centroid_x(self) -> float:
        """Return x-coordinate of the area centroid.

        The centroid is the geometric balance point of a uniform lamina over
        the enclosed airfoil area.

        Mathematical definition:
            M_10 = integral_A x dA,
            x_bar = M_10 / M_00

        Computation details:
            Uses polygon-based raw moments of the closed contour and divides the
            first raw x-moment by enclosed area.

        Geometric interpretation:
            x_bar indicates where area is centered along chordwise direction.

        Airfoil interpretation guidance:
            For conventional normalized airfoils, x_bar is typically in [0, 1]
            and indicates forward/aft loading tendency of geometry.

        Returns:
            float: Centroid x-location, units of length.
        """
        return float(self.raw_moments[(1, 0)] / self.area)

    @property
    def centroid_y(self) -> float:
        """Return y-coordinate of the area centroid.

        The centroid is the geometric balance point of a uniform lamina over
        the enclosed airfoil area.

        Mathematical definition:
            M_01 = integral_A y dA,
            y_bar = M_01 / M_00

        Computation details:
            Uses polygon-based raw moments of the closed contour and divides the
            first raw y-moment by enclosed area.

        Geometric interpretation:
            y_bar measures vertical bias of enclosed area relative to the
            reference x-axis.

        Airfoil interpretation guidance:
            Symmetric airfoils about y=0 should yield y_bar approximately 0,
            while cambered airfoils generally produce non-zero y_bar.

        Returns:
            float: Centroid y-location, units of length.
        """
        return float(self.raw_moments[(0, 1)] / self.area)

    @property
    def centroid(self) -> Vector2D:
        """Return centroid as a 2-vector ``[centroid_x, centroid_y]``.

        Mathematical definition:
            c = [x_bar, y_bar]
            with x_bar = M_10/M_00 and y_bar = M_01/M_00.

        Computation details:
            Uses first raw moments and area from the same polygon-integrated
            closed contour.

        Geometric interpretation:
            This is the balance point of the enclosed airfoil area for a
            uniform-density lamina.

        Airfoil interpretation guidance:
            Useful reference origin for central moments, inertia-like metrics,
            and principal spread directions.

        Returns:
            Vector2D: 2D centroid vector with ``x`` and ``y`` attributes.
        """
        return Vector2D([self.centroid_x, self.centroid_y])

    @cached_property
    def central_moments(self) -> dict[tuple[int, int], float]:
        """Compute central moments ``mu_pq`` up to ``max_total_order``.

        Central moments are generated from raw moments using centroid-shift
        binomial expansion.

        Returns:
            dict[tuple[int, int], float]: Mapping ``(p, q) -> mu_pq``.
        """
        x_bar = self.centroid_x
        y_bar = self.centroid_y

        central: dict[tuple[int, int], float] = {}
        for p in range(self.max_total_order + 1):
            for q in range(self.max_total_order + 1 - p):
                if p + q < 2:
                    continue
                central_value = 0.0
                for i in range(p + 1):
                    x_coeff = math.comb(p, i) * ((-x_bar) ** (p - i))
                    for j in range(q + 1):
                        coefficient = x_coeff * math.comb(q, j) * ((-y_bar) ** (q - j))
                        central_value += coefficient * self.raw_moments[(i, j)]
                central[(p, q)] = float(central_value)

        return central

    @property
    def central_second_moment_matrix(self) -> np.ndarray:
        """Return raw second central area-moment matrix about the centroid.

        Mathematical definition:
            mu_pq = integral_A (x - x_bar)^p (y - y_bar)^q dA

            second-order matrix:
                [[mu_20, mu_11],
                 [mu_11, mu_02]]

        Computation details:
            Central moments are formed from polygon-integrated raw moments by
            binomial expansion around centroid coordinates.

        Geometric interpretation:
            This matrix describes area spread around the centroid. Its
            eigenvectors give principal shape directions; eigenvalues quantify
            spread magnitude along those directions.

        Airfoil interpretation guidance:
            Larger mu_20 indicates stronger chordwise spread; larger mu_02
            indicates stronger thickness-direction spread. Non-zero mu_11
            indicates axis coupling (tilt of principal axes).

        Returns:
            np.ndarray: 2x2 matrix of raw central second moments, units length^4.
        """
        return np.array(
            [
                [self.central_moments[(2, 0)], self.central_moments[(1, 1)]],
                [self.central_moments[(1, 1)], self.central_moments[(0, 2)]],
            ],
            dtype=np.float64,
        )

    @property
    def second_central_moments(self) -> np.ndarray:
        """Alias for :attr:`central_second_moment_matrix`.

        Returns:
            np.ndarray: 2x2 matrix of raw second central moments, length^4.
        """
        return self.central_second_moment_matrix

    @property
    def area_covariance_matrix(self) -> np.ndarray:
        """Return area-normalized second central moments (covariance-like).

        Mathematical definition:
            C = (1 / M_00) * [[mu_20, mu_11],
                              [mu_11, mu_02]]

        Computation details:
            Divides the raw second central moment matrix by enclosed area.

        Geometric interpretation:
            This is the covariance matrix of a uniform area distribution over
            the airfoil region.

        Airfoil interpretation guidance:
            Compared to raw moments, this removes area scaling and yields a
            spread metric with units length^2, convenient for shape comparison
            at equal coordinate scaling.

        Returns:
            np.ndarray: 2x2 covariance-like matrix, units length^2.
        """
        return self.central_second_moment_matrix / self.area

    @property
    def planar_area_inertia_matrix(self) -> np.ndarray:
        """Return centroidal planar area-inertia convention matrix.

        This property exposes the classical second moment of area (area inertia)
        convention used in mechanics, built from central moments.

        Mathematical definition:
            I_xx = mu_02,
            I_yy = mu_20,
            I_xy = mu_11,

            I = [[I_xx, -I_xy],
                 [-I_xy, I_yy]]

        Computation details:
            Uses already-computed central second moments about centroid.

        Geometric interpretation:
            Describes resistance of the enclosed area to bending/rotation about
            centroidal in-plane axes.

        Airfoil interpretation guidance:
            Particularly useful when linking geometric descriptors to structural
            section properties or principal inertial axes.

        Returns:
            np.ndarray: 2x2 area-inertia matrix about centroid, units length^4.
        """
        mu20 = self.central_moments[(2, 0)]
        mu11 = self.central_moments[(1, 1)]
        mu02 = self.central_moments[(0, 2)]
        return np.array([[mu02, -mu11], [-mu11, mu20]], dtype=np.float64)

    @property
    def central_third_moments(self) -> dict[str, float]:
        """Return third-order central area moments as asymmetry descriptors.

        Mathematical definition:
            mu_30 = integral_A (x - x_bar)^3 dA
            mu_21 = integral_A (x - x_bar)^2 (y - y_bar) dA
            mu_12 = integral_A (x - x_bar) (y - y_bar)^2 dA
            mu_03 = integral_A (y - y_bar)^3 dA

        Computation details:
            Built from polygon raw moments via exact binomial conversion to
            central moments.

        Geometric interpretation:
            Third-order central moments are signed asymmetry measures of area
            distribution around centroid. They are skewness-like in sign and
            directional meaning, but remain dimensional.

        Airfoil interpretation guidance:
            Signs indicate bias direction of enclosed area. Magnitudes indicate
            strength of asymmetry, but scale with geometry size.

        Returns:
            dict[str, float]:
                Dictionary with keys ``mu30``, ``mu21``, ``mu12``, ``mu03``;
                each has units length^5.
        """
        return {
            "mu30": float(self.central_moments[(3, 0)]),
            "mu21": float(self.central_moments[(2, 1)]),
            "mu12": float(self.central_moments[(1, 2)]),
            "mu03": float(self.central_moments[(0, 3)]),
        }

    @property
    def standardized_third_moments(self) -> dict[str, float]:
        """Return dimensionless skewness-like third central moment descriptors.

        Mathematical definition:
            sigma_x^2 = mu_20 / M_00,
            sigma_y^2 = mu_02 / M_00,
            gamma_pq = mu_pq / (M_00 * sigma_x^p * sigma_y^q),
            for p + q = 3.

        Computation details:
            Uses central moments and area covariance scales to standardize out
            units and first-order size effects.

        Geometric interpretation:
            ``gamma_pq`` values are signed asymmetry indicators analogous to
            skewness along and across axes of the current coordinate frame.

        Airfoil interpretation guidance:
            Useful for comparing asymmetry across differently sized but
            similarly normalized airfoils. Values are not rotation invariant.

        Returns:
            dict[str, float]:
                Dimensionless values with keys ``gamma30``, ``gamma21``,
                ``gamma12``, ``gamma03``.
        """
        sigma_x = float(np.sqrt(max(self.area_covariance_matrix[0, 0], 0.0)))
        sigma_y = float(np.sqrt(max(self.area_covariance_matrix[1, 1], 0.0)))
        denom_eps = np.finfo(float).eps
        raw = self.central_third_moments

        scale_30 = self.area * (sigma_x ** 3)
        scale_21 = self.area * (sigma_x ** 2) * sigma_y
        scale_12 = self.area * sigma_x * (sigma_y ** 2)
        scale_03 = self.area * (sigma_y ** 3)

        # return {
        #     "gamma30": np.nan if abs(scale_30) <= denom_eps else float(raw["mu30"] / scale_30),
        #     "gamma21": np.nan if abs(scale_21) <= denom_eps else float(raw["mu21"] / scale_21),
        #     "gamma12": np.nan if abs(scale_12) <= denom_eps else float(raw["mu12"] / scale_12),
        #     "gamma03": np.nan if abs(scale_03) <= denom_eps else float(raw["mu03"] / scale_03),
        # }
        return {
            "gamma30": float(raw["mu30"] / scale_30) if abs(scale_30) > denom_eps else np.nan,
            "gamma21": float(raw["mu21"] / scale_21) if abs(scale_21) > denom_eps else np.nan,
            "gamma12": float(raw["mu12"] / scale_12) if abs(scale_12) > denom_eps else np.nan,
            "gamma03": float(raw["mu03"] / scale_03) if abs(scale_03) > denom_eps else np.nan,
        }

    @property
    def central_fourth_moments(self) -> dict[str, float]:
        """Return fourth-order central area moments as concentration descriptors.

        Mathematical definition:
            mu_40 = integral_A (x - x_bar)^4 dA
            mu_31 = integral_A (x - x_bar)^3 (y - y_bar) dA
            mu_22 = integral_A (x - x_bar)^2 (y - y_bar)^2 dA
            mu_13 = integral_A (x - x_bar) (y - y_bar)^3 dA
            mu_04 = integral_A (y - y_bar)^4 dA

        Computation details:
            Built from polygon raw moments via exact binomial conversion.

        Geometric interpretation:
            Fourth-order central moments characterize concentration versus
            outboard spread of area about centroid (kurtosis-like behavior), but
            remain dimensional.

        Airfoil interpretation guidance:
            Larger values typically indicate stronger area spread toward more
            distant chord-normal regions and/or sharper concentration contrast.

        Returns:
            dict[str, float]:
                Dictionary with keys ``mu40``, ``mu31``, ``mu22``, ``mu13``,
                ``mu04``; each has units length^6.
        """
        return {
            "mu40": float(self.central_moments[(4, 0)]),
            "mu31": float(self.central_moments[(3, 1)]),
            "mu22": float(self.central_moments[(2, 2)]),
            "mu13": float(self.central_moments[(1, 3)]),
            "mu04": float(self.central_moments[(0, 4)]),
        }

    @property
    def standardized_fourth_moments(self) -> dict[str, float]:
        """Return dimensionless kurtosis-like fourth central moment descriptors.

        Mathematical definition:
            sigma_x^2 = mu_20 / M_00,
            sigma_y^2 = mu_02 / M_00,
            kappa_pq = mu_pq / (M_00 * sigma_x^p * sigma_y^q),
            for p + q = 4.

        Computation details:
            Uses central moments and covariance scales to remove physical units.

        Geometric interpretation:
            ``kappa_pq`` values quantify concentration/tail-heaviness-like
            behavior along coordinate directions for the enclosed area.

        Airfoil interpretation guidance:
            Useful for relative shape concentration comparisons across airfoils
            at comparable normalization. Values are coordinate-frame dependent.

        Returns:
            dict[str, float]:
                Dimensionless values with keys ``kappa40``, ``kappa31``,
                ``kappa22``, ``kappa13``, ``kappa04``.
        """
        sigma_x = float(np.sqrt(max(self.area_covariance_matrix[0, 0], 0.0)))
        sigma_y = float(np.sqrt(max(self.area_covariance_matrix[1, 1], 0.0)))
        denom_eps = np.finfo(float).eps
        raw = self.central_fourth_moments

        scale_40 = self.area * (sigma_x ** 4)
        scale_31 = self.area * (sigma_x ** 3) * sigma_y
        scale_22 = self.area * (sigma_x ** 2) * (sigma_y ** 2)
        scale_13 = self.area * sigma_x * (sigma_y ** 3)
        scale_04 = self.area * (sigma_y ** 4)

        # return {
        #     "kappa40": np.nan if abs(scale_40) <= denom_eps else float(raw["mu40"] / scale_40),
        #     "kappa31": np.nan if abs(scale_31) <= denom_eps else float(raw["mu31"] / scale_31),
        #     "kappa22": np.nan if abs(scale_22) <= denom_eps else float(raw["mu22"] / scale_22),
        #     "kappa13": np.nan if abs(scale_13) <= denom_eps else float(raw["mu13"] / scale_13),
        #     "kappa04": np.nan if abs(scale_04) <= denom_eps else float(raw["mu04"] / scale_04),
        # }
        return {
            "kappa40": float(raw["mu40"] / scale_40) if abs(scale_40) > denom_eps else np.nan,
            "kappa31": float(raw["mu31"] / scale_31) if abs(scale_31) > denom_eps else np.nan,
            "kappa22": float(raw["mu22"] / scale_22) if abs(scale_22) > denom_eps else np.nan,
            "kappa13": float(raw["mu13"] / scale_13) if abs(scale_13) > denom_eps else np.nan,
            "kappa04": float(raw["mu04"] / scale_04) if abs(scale_04) > denom_eps else np.nan,
        }
