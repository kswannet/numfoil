import numpy as np
from typing import Tuple
import scipy.interpolate as si
import scipy.optimize as opt
from .spline_v2 import BSpline2D, SplevCBezier
from .geom2d import Point2D, Geom2D
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
    def count_header_lines(filepath: str) -> int:
        """
        Count the number of header lines before the coordinate data starts.

        Args:
            filepath (str): Path to the input file.

        Returns:
            int: The number of header lines.
        """
        def is_valid_coordinates_line(line: str) -> bool:
            try:
                # Try to parse the line as numeric data
                array = np.array([float(value) for value in line.split()])
                # Ensure it has either 2 elements (a single coordinate) or
                # matches one of the valid shapes [2, n] or [n, 2] when multiple lines are stacked.
                return len(array) == 2
            except ValueError:
                return False

        header_count = 0
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith(("#", "//")) or not is_valid_coordinates_line(line):
                    # Increment header count for empty lines, comments, or invalid coordinate lines
                    header_count += 1
                else:
                    # Stop counting once valid data is found
                    break

        return header_count

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

# TODO: I feel like this could be merged with the airfoil class, but I have yet
# TODO: to see the light on how to do this in a nice way.
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
    def get_trailing_edge(surface_spline: BSpline2D) -> np.ndarray:
        """Calculates the trailing edge point."""
        start_point = surface_spline.evaluate_at(0)
        end_point = surface_spline.evaluate_at(1)

        res1 = opt.minimize(lambda u: -np.linalg.norm(np.array([0,0])-surface_spline.evaluate_at(u)[0]), 0, bounds=[(0, 1)])
        res2 = opt.minimize(lambda u: -np.linalg.norm(np.array([0,0])-surface_spline.evaluate_at(u)[0]), 1, bounds=[(0, 1)])
        # if the maximum x value found is not the same at both ends of the
        # spline, the trailing edge is not properly defined and doubles back on
        # itself or the coordinates are missing one of the endpoints
        # ! this is still not ideal. If a trailing edge point is missing somehow
        # ! extrapolating might lead to a better result than just taking the
        # ! maximum x value. This is a quick fix for now.
        if abs(res1.fun - res2.fun) > 1e-5:
            # take location u with maximum x value, most likely to be trailing edge
            u_TE = res1.x[0] if -res1.fun>-res2.fun else res2.x[0]
            return surface_spline.evaluate_at(u_TE)


        # if endpoints are both at same x-coordinate, return midpoint
        elif abs(start_point[0] - end_point[0]) < 1e-5:
            # todo: fix x value to 1 here (if close already)?
            return 0.5 * (start_point + end_point)
        else:
            raise ValueError("Trailing edge not properly defined, possible unaccounted edge case")
            # return start_point if start_point[0] > end_point[0] else end_point

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
    def chord_vector(
        leading_edge: np.ndarray, trailing_edge: np.ndarray
    ) -> np.ndarray:
        """Calculates the chord vector from the leading to trailing edge."""
        return trailing_edge - leading_edge

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
    def calculate_rotation_angle(chord_vector: np.ndarray) -> float:
        """Calculates the rotation angle to align the chord line with the x-axis."""
        return -np.arctan2(chord_vector[1], chord_vector[0])

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
    def transformation(
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
        return (scale, translation, rotation_matrix)

    @staticmethod
    def normalize(points: np.ndarray) -> np.ndarray:
        """Normalizes points by applying translation, scaling, and rotation."""
        scale, translation, rotation_matrix = AirfoilProcessor.transformation(points)
        normalized_points = (points + translation) * scale
        # return (rotation_matrix @ normalized_points.T).T.view(Point2D)
        # # ! trying something else:
        normalized_points = (rotation_matrix @ normalized_points.T).T
        normalized_points = normalized_points.view(NormalizedAirfoilCoordinates)
        normalized_points._scale = scale
        normalized_points._translation = translation
        normalized_points._rotation_matrix = rotation_matrix
        normalized_points._rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return normalized_points


class NormalizedAirfoilCoordinates(Point2D):
    """Defines an array with normalized coordinates in 2D space.
    Probably not needed, but never know when one might need to check the
    transformation values used during normalization.

    TODO: makes this a more general?
    """
    def __new__(cls, array: Tuple[float, float] | np.ndarray):
        """Creates a :py:class:`NormalizedCoordinates` instance from ``array``."""
        obj = AirfoilProcessor.normalize(array).view(cls)
        obj._scale, obj._translation, obj._rotation_matrix = AirfoilProcessor.transformation(array)
        obj._rotation = np.arctan2(obj._rotation_matrix[1, 0], obj._rotation_matrix[0, 0])
        return obj

    @property
    def scale(self):
        """Returns the scaling factor which was applied."""
        return self._scale

    @property
    def translation(self):
        """Returns the applied translation vector."""
        return self._translation

    @property
    def rotation(self) -> float:
        """Returns the applied rotation angle in radians."""
        return self._rotation

    @property
    def rotation_matrix(self) -> float:
        """Returns the rotation angle in radians."""
        return self._rotation_matrix

    @property
    def transformation(self) -> Tuple[float, np.ndarray, float]:
        """Returns the transformation values used for normalization."""
        return (self._scale, self._translation, self._rotation)
