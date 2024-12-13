import numpy as np
import scipy.interpolate as si
from scipy.optimize import minimize
from functools import cached_property
from typing import Union
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from typing import Tuple


class AirfoilBase(metaclass=ABCMeta):
    """Abstract Base Class definition of an :py:class:`Airfoil`.
    ...
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


class Airfoil:
    """Unified airfoil class handling points-based airfoils with inconsistent or missing data."""

    def __init__(self, points: Union[np.ndarray, str]):
        self.unprocessed_points = points

    # @cached_property
    # def unprocessed_points(self) -> np.ndarray:
    #     """Loads airfoil points from a numpy array or a file."""
    #     if isinstance(self.raw_data, str):
    #         return np.genfromtxt(self.raw_data, skip_header=1)
    #     elif isinstance(self.raw_data, np.ndarray):
    #         return self.raw_data
    #     else:
    #         raise TypeError("Input must be a numpy array or a file path.")

    @classmethod
    def from_array(cls, points: np.ndarray):
        return cls(points)

    @classmethod
    def from_file(cls, filepath: str):
        points = np.genfromtxt(filepath, skip_header=1)
        return cls(points)

    



    @cached_property
    def preliminary_spline(self) -> si.PchipInterpolator:
        """Fits an initial spline to the raw points to help identify edge points."""
        points = self.remove_consecutive_duplicates(self.unprocessed_points)
        x, y = points.T
        return si.PchipInterpolator(x, y, extrapolate=False)

    @cached_property
    def trailing_edge(self) -> np.ndarray:
        """Calculates the trailing edge location as the midpoint of spline ends."""
        start_point = self.preliminary_spline(0)
        end_point = self.preliminary_spline(1)
        return 0.5 * (start_point + end_point)

    @cached_property
    def leading_edge(self) -> np.ndarray:
        """Determines the leading edge as the point on the spline furthest from the trailing edge."""
        trailing_edge = self.trailing_edge
        u_leading_edge = minimize(
            lambda u: -np.linalg.norm(
                self.preliminary_spline(u) - trailing_edge
            ),
            0.5,
            bounds=[(0, 1)],
        ).x[0]
        return self.preliminary_spline(u_leading_edge)

    @cached_property
    def processed_points(self) -> np.ndarray:
        """Processes points by normalizing, rotating, and translating based on edge points."""
        trailing_edge = self.trailing_edge
        leading_edge = self.leading_edge

        # Calculate chord vector and normalize points
        chord_vector = trailing_edge - leading_edge
        scale = 1 / np.linalg.norm(chord_vector)
        translation = -leading_edge
        rotation_angle = -np.arctan2(chord_vector[1], chord_vector[0])
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )

        # Apply transformation
        normalized_points = (self.unprocessed_points + translation) * scale
        return (rotation_matrix @ normalized_points.T).T

    @staticmethod
    def remove_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
        """Removes consecutive duplicate points."""
        diff = np.diff(points, axis=0)
        idx = np.where(np.any(diff != 0, axis=1))[0] + 1
        return np.vstack([points[0], points[idx]])

    @cached_property
    def upper_surface(self) -> si.PchipInterpolator:
        """Interpolator for the upper surface of the airfoil."""
        upper_points = self.processed_points[
            self.processed_points[:, 0] <= 0.5
        ]
        x, y = upper_points.T
        return si.PchipInterpolator(x, y, extrapolate=False)

    @cached_property
    def lower_surface(self) -> si.PchipInterpolator:
        """Interpolator for the lower surface of the airfoil."""
        lower_points = self.processed_points[
            self.processed_points[:, 0] >= 0.5
        ]
        x, y = lower_points.T
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
        result = minimize(
            lambda x: -self.thickness_at(x), 0.5, bounds=[(0, 1)]
        )
        if result.success:
            return result.x[0], -result.fun
        else:
            raise RuntimeError("Failed to find maximum thickness.")

    @property
    def max_camber(self) -> Tuple[float, float]:
        """Finds the location and value of maximum camber."""
        result = minimize(lambda x: -self.camber_at(x), 0.5, bounds=[(0, 1)])
        if result.success:
            return result.x[0], -result.fun
        else:
            raise RuntimeError("Failed to find maximum camber.")
