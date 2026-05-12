from __future__ import annotations

from functools import cached_property
from typing import Tuple, Union

import numpy as np

from ..util import cosine_spacing, ensure_1d_vector
from .geom2d import Point2D, normalize_2d, rotate_2d_90ccw
from .spline import Curve, BSpline2D
from .airfoil import AirfoilBase, BsplineAirfoil


ArrayLike = Union[float, np.ndarray]


def parse_naca4_code(naca_code: str) -> Tuple[float, float, float]:
    """Parses a ``naca_code`` into the 3 (scaled) values needed:
    max camber, max camber location, and maximum thickness.

    Args:
        naca_code: A string like "NACA2412" or "2412".
    Returns:
        Tuple: (m, p, t) where
            - m is max camber (0-1),
            - p is max camber location (0-1),
            - t is max thickness (0-1).
    """
    digits = naca_code.upper().replace("NACA", "")
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError("NACA4 code must contain exactly 4 digits.")
    m, p, t1, t2 = map(int, digits)
    return m / 100.0, p / 10.0, float(f"0.{t1}{t2}")


def parse_naca5_code(naca_code: str) -> Tuple[float, float, float, bool]:
    """Parses a ``naca_code`` into the 4 values needed:
    design lift coefficient, max camber location, max thickness, and reflex
    flag.

    Args:
        naca_code: A string like "NACA24112" or "24112".
    Returns:
        Tuple: (cl_design, p, t, reflex) where
            - cl_design is the design lift coefficient (0-1),
            - p is max camber location (0-1),
            - t is max thickness (0-1),
            - reflex is True if the airfoil is reflexed.
    """
    digits = naca_code.upper().replace("NACA", "")
    if len(digits) != 5 or not digits.isdigit():
        raise ValueError("NACA5 code must contain exactly 5 digits.")

    d1, d2, d3, d4, d5 = map(int, digits)
    cl_design = 0.15 * d1
    p = (10 * d2 + d3) / 200.0
    t = float(f"0.{d4}{d5}")
    reflex = (d3 != 0)  # common convention: third digit 0 = normal, non-zero = reflex
    return cl_design, p, t, reflex


def naca4_points(naca_code: str, te_closed: bool = False) -> np.ndarray:
    """Returns standard surface points array for a NACA 4-series airfoil defined
    by ``naca_code`` in Selig format.
    Args:
        naca_code: A string like "NACA2412" or "2412".
        te_closed: Whether the trailing edge is closed.
    Returns:
        np.ndarray: Airfoil surface points, shape (199, 2), ordered from trailing edge
            along upper surface to leading edge and back along lower surface to
            trailing edge.
    """
    m, p, t = parse_naca4_code(naca_code)

    x = cosine_spacing(0.0, 1.0, num=100)
    fwd = x <= p
    aft = ~fwd

    camber_vals = np.zeros((x.size, 2))
    camber_vals[:, 0] = x
    camber_tangents = np.zeros((x.size, 2))
    camber_tangents[:, 0] = 1.0

    if m > 0.0:
        camber_vals[fwd, 1] = (m / p**2) * (2.0 * p * x[fwd] - x[fwd] ** 2)
        camber_vals[aft, 1] = (m / (1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * x[aft] - x[aft] ** 2
        )
        camber_tangents[fwd, 1] = (2.0 * m / p**2) * (p - x[fwd])
        camber_tangents[aft, 1] = (2.0 * m / (1.0 - p) ** 2) * (p - x[aft])

    semi_thickness_vals = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * (x ** 2)
        + 0.2843 * (x ** 3)
        - (0.1036 if te_closed else 0.1015) * (x ** 4)
    )

    camber_normals = rotate_2d_90ccw(normalize_2d(camber_tangents))

    return np.vstack([
        (camber_vals + camber_normals * semi_thickness_vals.reshape(-1, 1))[::-1],
        (camber_vals - camber_normals * semi_thickness_vals.reshape(-1, 1))[1:]
    ]).view(Point2D)



class AnalyticYCurve(Curve):
    """
    Single-valued analytic curve y(x), exposed in the same shape conventions
    as other curve classes.
    """

    def __init__(self, y_fn, dy_fn, d2y_fn):
        """Initialize an analytic single-valued curve y(x).

        Args:
            y_fn (Callable[[np.ndarray], np.ndarray]): Function returning y(x).
            dy_fn (Callable[[np.ndarray], np.ndarray]): Function returning dy/dx.
            d2y_fn (Callable[[np.ndarray], np.ndarray]): Function returning d2y/dx2.
        """
        self._y_fn = y_fn
        self._dy_fn = dy_fn
        self._d2y_fn = d2y_fn

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """Evaluate y(x).

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: y values, shape (N,).
        """
        x = ensure_1d_vector(x)
        return np.asarray(self._y_fn(x))

    def evaluate_at(self, x: ArrayLike) -> np.ndarray:
        """Evaluate curve points at x.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Points [x, y], shape (N, 2).
        """
        x = ensure_1d_vector(x)
        y = self.__call__(x)
        return np.column_stack([x, y]).view(Point2D)

    def first_deriv_at(self, x: ArrayLike) -> np.ndarray:
        """Return vector derivative [dx/dx, dy/dx] for Curve compatibility.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Derivative vectors, shape (N, 2).
        """
        x = ensure_1d_vector(x)
        dy = np.asarray(self._dy_fn(x))
        return np.column_stack([np.ones_like(x), dy])

    def second_deriv_at(self, x: ArrayLike) -> np.ndarray:
        """Return vector second derivative [0, d2y/dx2] for compatibility."""
        """
        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Second-derivative vectors, shape (N, 2).
        """
        x = ensure_1d_vector(x)
        d2y = np.asarray(self._d2y_fn(x))
        return np.column_stack([np.zeros_like(x), d2y])


class NACAHalfThicknessCurve(AnalyticYCurve):
    """
    Shared NACA half-thickness curve for both 4- and 5-series.
    """

    def __init__(self, thickness: float, te_closed: bool = False):
        """Initialize the NACA half-thickness curve.

        Args:
            thickness (float): Maximum thickness ratio (0-1).
            te_closed (bool): If True, use closed trailing edge coefficient.
        """
        self.t = thickness
        self.te_closed = te_closed

        a4 = 0.1036 if self.te_closed else 0.1015

        def y_fn(x):
            """Evaluate the semithickness at ``x``

            Args:
                x (ArrayLike): Chordwise locations

            Returns:
                np.ndarray: semithickness values at provided ``x``
            """
            return (self.t / 0.2) * (
                0.2969 * np.sqrt(x)
                - 0.1260 * x
                - 0.3516 * (x ** 2)
                + 0.2843 * (x ** 3)
                - a4 * (x ** 4)
            )

        def dy_fn(x):
            """Evaluate the first derivative of semithickness curve at ``x``.

            Args:
                x (ArrayLike): Chordwise locations

            Returns:
                np.ndarray: Derivative values at provided ``x``
            """
            # d/dx sqrt(x) term is singular at x=0, clamp for numeric stability.
            xs = np.clip(x, 1e-14, None)
            return (self.t / 0.2) * (
                0.2969 * 0.5 / np.sqrt(xs)
                - 0.1260
                - 2.0 * 0.3516 * x
                + 3.0 * 0.2843 * x**2
                - 4.0 * a4 * x**3
            )

        def d2y_fn(x):
            """Evaluate the second derivative of semithickness curve at ``x``.

            Args:
                x (ArrayLike): Chordwise locations

            Returns:
                np.ndarray: Second derivative values at provided ``x``
            """
            xs = np.clip(x, 1e-14, None)
            return (self.t / 0.2) * (
                -0.2969 * 0.25 / (xs ** 1.5)
                - 2.0 * 0.3516
                + 6.0 * 0.2843 * x
                - 12.0 * a4 * x**2
            )

        super().__init__(y_fn, dy_fn, d2y_fn)


class NACA4CamberCurve(AnalyticYCurve):
    """Analytic NACA 4-series camber line y(x)."""

    def __init__(self, m: float, p: float):
        """Initialize a NACA 4-series camber line.

        Args:
            m (float): Maximum camber (0-1).
            p (float): Location of maximum camber (0-1).
        """
        self.m = m
        self.p = p
        if (m == 0.0) ^ (p == 0.0):
            raise ValueError("Non-zero camber requires non-zero camber location.")

        def y_fn(x):
            """Evaluate the camber line at ``x``.

            Args:
                x (ArrayLike): Chordwise locations

            Returns:
                np.ndarray: Camber values at provided ``x``
            """
            y = np.zeros_like(x)
            if self.m == 0.0:
                return y
            fwd = x <= self.p
            aft = ~fwd
            y[fwd] = (self.m / self.p**2) * (2.0 * self.p * x[fwd] - x[fwd] ** 2)
            y[aft] = (self.m / (1.0 - self.p) ** 2) * (
                (1.0 - 2.0 * self.p) + 2.0 * self.p * x[aft] - x[aft] ** 2
            )
            return y

        def dy_fn(x):
            """Evaluate the first derivative of camber line at ``x``.

            Args:
                x (ArrayLike): Chordwise locations

            Returns:
                np.ndarray: Derivative values at provided ``x``
            """
            dy = np.zeros_like(x)
            if self.m == 0.0:
                return dy
            fwd = x <= self.p
            aft = ~fwd
            dy[fwd] = (2.0 * self.m / self.p**2) * (self.p - x[fwd])
            dy[aft] = (2.0 * self.m / (1.0 - self.p) ** 2) * (self.p - x[aft])
            return dy

        def d2y_fn(x):
            """Evaluate the second derivative of camber line at ``x``.

            Args:
                x (ArrayLike): Chordwise locations

            Returns:
                np.ndarray: Second derivative values at provided ``x``
            """
            d2 = np.zeros_like(x)
            if self.m == 0.0:
                return d2
            fwd = x <= self.p
            aft = ~fwd
            d2[fwd] = -2.0 * self.m / self.p**2
            d2[aft] = -2.0 * self.m / (1.0 - self.p) ** 2
            return d2

        super().__init__(y_fn, dy_fn, d2y_fn)


class NACA5CamberCurve(AnalyticYCurve):
    """
    Standard 5-series camber line using tabulated (m, k1[, k2/k1]) constants.
    """

    _NORMAL = {
        0.05: (0.0580, 361.4),
        0.10: (0.1260, 51.64),
        0.15: (0.2025, 15.957),
        0.20: (0.2900, 6.643),
        0.25: (0.3910, 3.230),
    }
    _REFLEX = {
        0.10: (0.1300, 51.990, 0.000764),
        0.15: (0.2170, 15.793, 0.00677),
        0.20: (0.3180, 6.520, 0.0303),
        0.25: (0.4410, 3.191, 0.1355),
    }

    def __init__(self, cl_design: float, p: float, reflex: bool):
        """Initialize a NACA 5-series camber line.

        Args:
            cl_design (float): Design lift coefficient.
            p (float): Location of maximum camber (0-1).
            reflex (bool): Whether the camber line is reflexed.
        """
        p_key = round(p, 2)
        self.reflex = reflex

        if reflex:
            if p_key not in self._REFLEX:
                raise ValueError(f"Unsupported reflex 5-series p={p}.")
            m, k1, k2k1 = self._REFLEX[p_key]
            k2 = k1 * k2k1
        else:
            if p_key not in self._NORMAL:
                raise ValueError(f"Unsupported normal 5-series p={p}.")
            m, k1 = self._NORMAL[p_key]
            k2 = 0.0

        # Scale k1 by target design Cl.
        # Reference tables are usually for Cl=0.3.
        scale = cl_design / 0.3 if cl_design > 0 else 1.0
        k1 *= scale
        k2 *= scale

        self.m = m
        self.k1 = k1
        self.k2 = k2

        def y_fn(x):
            y = np.zeros_like(x)
            fwd = x <= self.m
            aft = ~fwd
            if not self.reflex:
                y[fwd] = (self.k1 / 6.0) * (
                    x[fwd] ** 3
                    - 3.0 * self.m * x[fwd] ** 2
                    + self.m**2 * (3.0 - self.m) * x[fwd]
                )
                y[aft] = (self.k1 / 6.0) * self.m**3 * (1.0 - x[aft])
            else:
                y[fwd] = (self.k1 / 6.0) * (
                    (x[fwd] - self.m) ** 3
                    - self.k2 * (1.0 - self.m) ** 3 * x[fwd]
                    + self.m**3 * x[fwd]
                    - self.m**3
                )
                y[aft] = (self.k1 / 6.0) * (
                    self.k2 * (x[aft] - self.m) ** 3
                    - self.k2 * (1.0 - self.m) ** 3 * x[aft]
                    + self.m**3 * x[aft]
                    - self.m**3
                )
            return y

        def dy_fn(x):
            dy = np.zeros_like(x)
            fwd = x <= self.m
            aft = ~fwd
            if not self.reflex:
                dy[fwd] = (self.k1 / 6.0) * (
                    3.0 * x[fwd] ** 2
                    - 6.0 * self.m * x[fwd]
                    + self.m**2 * (3.0 - self.m)
                )
                dy[aft] = -(self.k1 / 6.0) * self.m**3
            else:
                dy[fwd] = (self.k1 / 6.0) * (
                    3.0 * (x[fwd] - self.m) ** 2
                    - self.k2 * (1.0 - self.m) ** 3
                    + self.m**3
                )
                dy[aft] = (self.k1 / 6.0) * (
                    3.0 * self.k2 * (x[aft] - self.m) ** 2
                    - self.k2 * (1.0 - self.m) ** 3
                    + self.m**3
                )
            return dy

        def d2y_fn(x):
            d2 = np.zeros_like(x)
            fwd = x <= self.m
            aft = ~fwd
            if not self.reflex:
                d2[fwd] = (self.k1 / 6.0) * (6.0 * x[fwd] - 6.0 * self.m)
                d2[aft] = 0.0
            else:
                d2[fwd] = (self.k1 / 6.0) * (6.0 * (x[fwd] - self.m))
                d2[aft] = (self.k1 / 6.0) * (6.0 * self.k2 * (x[aft] - self.m))
            return d2

        super().__init__(y_fn, dy_fn, d2y_fn)


class NACAAnalyticAirfoil(AirfoilBase):
    """Analytic NACA airfoil using camber + thickness with normal offset."""

    def __init__(
        self,
        camber_curve: AnalyticYCurve,
        thickness_curve: NACAHalfThicknessCurve,
        n_points_per_side: int = 100,
        name: str = "",
        description: str = "",
    ):
        """Initialize analytic NACA airfoil.

        Args:
            camber_curve (AnalyticYCurve): Camber line curve.
            thickness_curve (NACAHalfThicknessCurve): Half-thickness curve.
            n_points_per_side (int): Sampling resolution per side.
            name (str): Airfoil short name.
            description (str): Optional description or long name.
        """
        self.camber_curve = camber_curve
        self.thickness_curve = thickness_curve
        self.n_points_per_side = n_points_per_side
        self.name = name
        self.description = description or name

    def camber_points_at(self, x: ArrayLike) -> np.ndarray:
        """Return camber line points at x.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Camber points, shape (N, 2).
        """
        x = ensure_1d_vector(x)
        y = self.camber_curve(x)
        return np.column_stack([x, y]).view(Point2D)

    def camber_at(self, x: ArrayLike) -> np.ndarray:
        """Return camber y-values at x.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Camber values, shape (N,).
        """
        return self.camber_curve(x)

    def camber_tangent_at(self, x: ArrayLike) -> np.ndarray:
        """Return unit tangent vectors along the camber line.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Tangent vectors, shape (N, 2).
        """
        x = ensure_1d_vector(x)
        t = self.camber_curve.first_deriv_at(x)
        return normalize_2d(t, inplace=True)

    def camber_normal_at(self, x: ArrayLike) -> np.ndarray:
        """Return unit normal vectors along the camber line.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Normal vectors, shape (N, 2).
        """
        return rotate_2d_90ccw(self.camber_tangent_at(x))

    def half_thickness_at(self, x: ArrayLike) -> np.ndarray:
        """Return half-thickness y_t at x.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Half-thickness values, shape (N,).
        """
        return self.thickness_curve(x)

    def thickness_at(self, x: ArrayLike) -> np.ndarray:
        """Return thickness t(x) at x.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Thickness values, shape (N,).
        """
        return 2.0 * self.half_thickness_at(x)

    def upper_surface_points_at(self, x: ArrayLike) -> np.ndarray:
        """Return upper surface points at x.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Upper surface points, shape (N, 2).
        """
        n_c = self.camber_normal_at(x)
        y_t = self.half_thickness_at(x)
        return self.camber_points_at(x) + n_c * y_t.reshape(-1, 1)

    def lower_surface_points_at(self, x: ArrayLike) -> np.ndarray:
        """Return lower surface points at x.

        Args:
            x (float | np.ndarray): Chordwise location(s).

        Returns:
            np.ndarray: Lower surface points, shape (N, 2).
        """
        n_c = self.camber_normal_at(x)
        y_t = self.half_thickness_at(x)
        return self.camber_points_at(x) - n_c * y_t.reshape(-1, 1)

    # def upper_surface_at(self, x: ArrayLike) -> np.ndarray:
    #     """Return upper surface y-values at x.

    #     Args:
    #         x (float | np.ndarray): Chordwise location(s).

    #     Returns:
    #         np.ndarray: Upper y-values, shape (N,).
    #     """
    #     return self.upper_surface_points_at(x)[:, 1]

    # def lower_surface_at(self, x: ArrayLike) -> np.ndarray:
    #     """Return lower surface y-values at x.

    #     Args:
    #         x (float | np.ndarray): Chordwise location(s).

    #     Returns:
    #         np.ndarray: Lower y-values, shape (N,).
    #     """
    #     return self.lower_surface_points_at(x)[:, 1]

    def to_bspline_airfoil(self, normalize: bool = True) -> BsplineAirfoil:
        """Create a BsplineAirfoil from sampled analytic points.

        To normalize the naca airfoil, meaning the geometry will not exceed
        x=[0,1], a new Bspline airfoil object is created, as the normalized
        geometry will no longer strictly adhere to the naca equations.
        The will change the definition of the leading edge and camber line
        compared to the original, but the outside surface will remain unchanged,
        only undergoing a slight transformation

        Args:
            normalize (bool): Normalize the resulting airfoil.

        Returns:
            BsplineAirfoil: Wrapped B-spline airfoil instance.
        """
        x = cosine_spacing(0.0, 1.0, num=self.n_points_per_side)
        pts = np.vstack(
            [
                self.upper_surface_points_at(x)[::-1],
                self.lower_surface_points_at(x)[1:],
            ]
        )
        return BsplineAirfoil.from_coordinate_array(
            points=pts,
            name=self.name,
            description=self.description,
            normalize=normalize,
        )

    @cached_property
    def surface(self):
        """Return parametric surface curve representing the full airfoil.

        Returns:
            ParametricCurve: Surface curve for the airfoil.
        """
        x = cosine_spacing(0.0, 1.0, num=self.n_points_per_side)
        return BSpline2D(
            np.vstack(
                [
                    self.upper_surface_points_at(x)[::-1],
                    self.lower_surface_points_at(x)[1:],
                ]
            )
        )

    @cached_property
    def upper_surface(self):
        """Return upper surface curve.

        Returns:
            ParametricCurve: Upper surface curve.
        """
        x = cosine_spacing(0.0, 1.0, num=self.n_points_per_side)
        return BSpline2D(
            self.upper_surface_points_at(x),
        )

    @cached_property
    def lower_surface(self):
        """Return lower surface curve.

        Returns:
            ParametricCurve: Lower surface curve.
        """
        x = cosine_spacing(0.0, 1.0, num=self.n_points_per_side)
        return BSpline2D(
            self.lower_surface_points_at(x),
        )

    @cached_property
    def camber_line(self):
        """Return camber line curve.

        Returns:
            ParametricCurve: Camber line curve.
        """
        x = cosine_spacing(0.0, 1.0, num=self.n_points_per_side)
        return BSpline2D(
            self.camber_points_at(x)
        )

    @cached_property
    def thickness_distribution(self):
        """Return thickness distribution curve.

        Returns:
            ParametricCurve: Thickness distribution curve.
        """
        x = cosine_spacing(0.0, 1.0, num=self.n_points_per_side)
        thickness_pts = np.column_stack([x, self.thickness_at(x)])
        return BSpline2D(
            thickness_pts,
        )


class NACA4Airfoil(NACAAnalyticAirfoil):
    """NACA 4-series analytic airfoil."""
    def __init__(
        self,
        naca_code: str,
        te_closed: bool = False,
        n_points_per_side: int = 100,
    ):
        """Initialize a NACA 4-series analytic airfoil.
        NACA 4 digits:
            - 1st digit: max camber in % of chord (m*100)
            - 2nd digit: location of max camber in tenths of chord (p*10)
            - 3rd & 4th digits: max thickness in % of chord (t*100)

        Args:
            naca_code (str): NACA 4-digit code.
            te_closed (bool): If True, close the trailing edge.
            n_points_per_side (int): Sampling resolution per side.
        """
        m, p, t = parse_naca4_code(naca_code)
        camber = NACA4CamberCurve(m, p)
        half_t = NACAHalfThicknessCurve(t, te_closed=te_closed)
        name = f"NACA{naca_code.upper().replace('NACA', '')}"
        super().__init__(
            camber_curve=camber,
            thickness_curve=half_t,
            n_points_per_side=n_points_per_side,
            name=name,
            description=f"{name} analytic",
        )


class NACA5Airfoil(NACAAnalyticAirfoil):
    """NACA 5-series analytic airfoil."""

    def __init__(
        self,
        naca_code: str,
        te_closed: bool = False,
        n_points_per_side: int = 100,
    ):
        """Initialize a NACA 5-series analytic airfoil.

        Args:
            naca_code (str): NACA 5-digit code.
            te_closed (bool): If True, close the trailing edge.
            n_points_per_side (int): Sampling resolution per side.
        """
        cl, p, t, reflex = parse_naca5_code(naca_code)
        camber = NACA5CamberCurve(cl_design=cl, p=p, reflex=reflex)
        half_t = NACAHalfThicknessCurve(t, te_closed=te_closed)
        name = f"NACA{naca_code.upper().replace('NACA', '')}"
        super().__init__(
            camber_curve=camber,
            thickness_curve=half_t,
            n_points_per_side=n_points_per_side,
            name=name,
            description=f"{name} analytic",
        )
