import numpy as np
from typing import Tuple
import scipy.interpolate as si
from scipy.optimize import minimize
from .spline import BSpline2D, ClampedBezierCurve, CompositeBezierBspline


class AirfoilDataLoader:
    """Handles loading airfoil data from files or arrays."""

    @staticmethod
    def load_from_file(filepath: str) -> np.ndarray:
        """Loads airfoil points from a file."""
        try:
            return np.genfromtxt(filepath, skip_header=1)
        except Exception as e:
            raise ValueError(f"Failed to load file: {e}")

    @staticmethod
    def load_from_array(data: np.ndarray) -> np.ndarray:
        """Loads airfoil points from a numpy array."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        return data


class AirfoilProcessor:
    """Processes raw airfoil points for alignment and normalization."""

    @staticmethod
    def remove_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
        """Removes consecutive duplicate points."""
        diff = np.diff(points, axis=0)
        idx = np.where(np.any(diff != 0, axis=1))[0] + 1
        return np.vstack([points[0], points[idx]])

    @staticmethod
    def fit_preliminary_spline(points: np.ndarray) -> si.PchipInterpolator:
        """Fits a preliminary spline to the raw points."""
        points = AirfoilProcessor.remove_consecutive_duplicates(points)
        x, y = points.T
        return si.PchipInterpolator(x, y, extrapolate=False)

    @staticmethod
    def calculate_edge_points(
        preliminary_spline: si.PchipInterpolator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates initial estimates for the leading and trailing edge."""
        # Trailing edge: midpoint of the spline ends
        start_point = preliminary_spline(0)
        end_point = preliminary_spline(1)
        trailing_edge = 0.5 * (start_point + end_point)

        # Leading edge: point on the spline furthest from the trailing edge
        u_leading_edge = minimize(
            lambda u: -np.linalg.norm(preliminary_spline(u) - trailing_edge),
            0.5,
            bounds=[(0, 1)],
        ).x[0]
        leading_edge = preliminary_spline(u_leading_edge)

        return leading_edge, trailing_edge

    @staticmethod
    def normalize_points(
        points: np.ndarray, leading_edge: np.ndarray, trailing_edge: np.ndarray
    ) -> np.ndarray:
        """Normalizes points by applying translation, scaling, and rotation."""
        scale = AirfoilProcessor.calculate_scale(leading_edge, trailing_edge)
        translation = AirfoilProcessor.calculate_translation(leading_edge)
        rotation_matrix = AirfoilProcessor.calculate_rotation_matrix(
            leading_edge, trailing_edge
        )

        # Apply translation, scaling, and rotation
        normalized_points = (points + translation) * scale
        return (rotation_matrix @ normalized_points.T).T

    @staticmethod
    def calculate_scale(
        leading_edge: np.ndarray, trailing_edge: np.ndarray
    ) -> float:
        """Calculates the scale factor based on the chord length."""
        chord_vector = trailing_edge - leading_edge
        return 1 / np.linalg.norm(chord_vector)

    @staticmethod
    def calculate_translation(leading_edge: np.ndarray) -> np.ndarray:
        """Calculates the translation vector to move the leading edge to the origin."""
        return -leading_edge

    @staticmethod
    def calculate_rotation_matrix(
        leading_edge: np.ndarray, trailing_edge: np.ndarray
    ) -> np.ndarray:
        """Calculates the rotation matrix to align the chord line with the x-axis."""
        chord_vector = trailing_edge - leading_edge
        rotation_angle = -np.arctan2(chord_vector[1], chord_vector[0])
        return np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
