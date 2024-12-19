import numpy as np
from typing import Tuple
import scipy.interpolate as si
import scipy.optimize as opt
from .spline_v2 import BSpline2D, SplevCBezier
import os

class AirfoilDataFile:
    """Handles loading airfoil data from files or arrays."""

    @staticmethod
    def has_header(filepath: str) -> bool:
        """
        Determine if a file has a header by inspecting the first line.

        Args:
            filepath (str): Path to the file.

        Returns:
            bool: True if the file has a header, False otherwise.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()

        # Check if the first line contains non-numeric content
        try:
            # Attempt to convert each word in the first line to a float
            _ = [float(value) for value in first_line.split()]
            return False  # If successful, it's numeric, so no header
        except ValueError:
            return True  # If conversion fails, it's likely a header

    @staticmethod
    def load_from_file(filepath: str, header: int) -> np.ndarray:
        """Loads airfoil points from a file."""
        try:
            return np.genfromtxt(filepath, skip_header=header)
        except Exception as e:
            raise ValueError(f"Failed to load file: {e}") from e

    @staticmethod
    def header(filepath: str) -> str:
        """Returns the name of the airfoil from header line."""
        with open(filepath, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
        return first_line

    @staticmethod
    def filename(filepath) -> str:
        """Returns the name of the airfoil from filename."""
        return os.path.splitext(os.path.basename(filepath))[0]


class AirfoilProcessor:
    """Processes raw airfoil points for alignment and normalization."""

    @staticmethod
    def remove_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
        """Removes consecutive duplicate points."""
        diff = np.diff(points, axis=0)
        idx = np.where(np.any(diff != 0, axis=1))[0] + 1
        return np.vstack([points[0], points[idx]])

    @staticmethod
    def fit_preliminary_spline(points: np.ndarray) -> BSpline2D:
        """Fits a preliminary spline to the raw points."""
        points = AirfoilProcessor.remove_consecutive_duplicates(points)
        return BSpline2D(points)

    @staticmethod
    def get_trailing_edge(preliminary_spline: BSpline2D) -> np.ndarray:
        """Calculates the trailing edge point."""
        start_point = preliminary_spline.evaluate_at(0)
        end_point = preliminary_spline.evaluate_at(1)
        # if endpoints are both at same x-coordinate, return midpoint
        if abs(start_point[0] - end_point[0]) < 1e-4:
            return 0.5 * (start_point + end_point)
        # else, return the point with the largest x-coordinate
        else:
            return start_point if start_point[0] > end_point[0] else end_point

    @staticmethod
    def get_leading_edge(
        preliminary_spline: BSpline2D,
        trailing_edge: np.ndarray = None,
        method: str = "L2_norm",
    ) -> np.ndarray:
        """Finds the leading edge point by maximizing the distance from the
        trailing edge."""
        if trailing_edge is None:
            trailing_edge = AirfoilProcessor.get_trailing_edge(preliminary_spline)

        # initial guess is midway the surface curve/spline
        init_guess = 0.5

        def objective(u):
            residuals = trailing_edge - preliminary_spline.evaluate_at(u)
            match method:
                case "least_squares":
                    return -residuals.ravel()

                case "L2_norm":
                    return -np.linalg.norm(residuals)

                case _:
                    raise ValueError(
                        "Invalid method. Use 'least_squares' or 'L2_norm'."
                    )

        match method:
            case "least_squares":
                result = opt.least_squares(
                    objective, init_guess,
                    bounds=(0, 1),
                    )
            case "L2_norm":
                result = opt.minimize(
                    objective,
                    init_guess,
                    bounds=[(0, 1)],
                    # method="SLSQP",
                    )
            case _:
                raise ValueError(
                    "Invalid method. Use 'least_squares' or 'L2_norm'."
                )
        return preliminary_spline.evaluate_at(result.x[0])

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

    @staticmethod
    def normalize(
        points: np.ndarray) -> np.ndarray:
        """Normalizes points by applying translation, scaling, and rotation."""
        preliminary_spline = AirfoilProcessor.fit_preliminary_spline(points)
        trailing_edge = AirfoilProcessor.get_trailing_edge(preliminary_spline)
        leading_edge = AirfoilProcessor.get_leading_edge(
            preliminary_spline,
            trailing_edge
        )
        scale = AirfoilProcessor.calculate_scale(leading_edge, trailing_edge)
        translation = AirfoilProcessor.calculate_translation(leading_edge)
        rotation_matrix = AirfoilProcessor.calculate_rotation_matrix(
            leading_edge, trailing_edge
        )

        # Apply translation, scaling, and rotation
        normalized_points = (points + translation) * scale
        return (rotation_matrix @ normalized_points.T).T
