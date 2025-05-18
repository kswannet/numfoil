from functools import cached_property
import numpy as np
from typing import Tuple
import scipy.interpolate as si
import scipy.optimize as opt
from .spline import BSpline2D, SplevCBezier
from .geom2d import Point2D, Geom2D
import os

class AirfoilDataFile:
    """
    Class that loads airfoil coordinate data and any header lines from a file.

    This version handles the possibility of multiple header lines, stopping only
    once it encounters the first valid coordinate line.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    @cached_property
    def filename(self) -> str:
        """Returns the name of the airfoil from filename."""
        return os.path.splitext(os.path.basename(self.filepath))[0]

    @cached_property
    def _content(self) -> list[str]:
        """Returns the content of the file as a list of lines."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File {self.filepath} not found.")
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.readlines()

    def __repr__(self):
        return f"AirfoilDataFile('{self.filename}')"

    def __str__(self):
        return str(" ".join(self._content))

    @cached_property
    def num_header_lines(self) -> int:
        """The number of header lines found.

        Returns:
            int: The number of header lines.
        """
        count = 0
        for line in self._content:
            if line.strip() and self._is_valid_coordinate(line.strip()):
                # Found the first valid coordinate => stop counting
                break
            count += 1
        return count

    @property
    def header(self) -> str:
        """Returns file header.

        Returns:
            str | None: The header lines as a single multi‐line string.
        """
        if self.num_header_lines == 0:
            return None
        return "\n".join([
                line.strip()
                for line in self._content[:self.num_header_lines]
        ])

    @cached_property
    def points(self) -> np.ndarray:
        """Loads airfoil points from a file."""
        coord_lines = self._content[self.num_header_lines:]
        if not coord_lines:
            return None
        try:
            return np.genfromtxt(coord_lines, comments=None).view(Point2D)
        except Exception as e:
            raise ValueError(
                f"Failed to parse coordinates in {self.filepath}: {e}"
            ) from e

    @staticmethod
    def _is_valid_coordinate(line: str) -> bool:
        """Check if a line contains valid coordinate data.

        Logic:
            If the line can be converted to a float array, it's valid, but only
            if it has exactly 2 elements (a single [x,y] coordinate) or matches
            one of the valid shapes [2, n] or [n, 2] when multiple lines are
            stacked.

        Args:
            line (str): A line from the file.

        Returns:
            bool: True if the line contains valid coordinate data,
                False otherwise.
        """
        try:
            array = np.array([float(value) for value in line.split()])
            return len(array) == 2
        except ValueError:
            return False


class AirfoilNormalizer:
    """
    Normalizes airfoil data — either raw points or a BSpline2D — and returns a
    spline with transformed control points containing transformation metadata.

    Airfoil data, especially from the UIUC database, sucks. It is messy,
    inconsistent (varying number of points, trailing edges from closed to
    gigantic gaps), contains errors (misplaced points), missing leading and
    trailing edge points, and is a pain to work with. Instead of filtering out
    any airfoil which causes issues (which wouldn't leave many usable ones),
    this class tries to normalize the data as best as possible. Like the data,
    this process is messy, trying to deal with all the different edge cases, and
    is far from flawless. But it kinda works. There are probably better ways to
    deal with this, but those are outside my capabilities.
    """

    @classmethod
    def normalize(cls, data: np.ndarray | BSpline2D, output="spline") -> BSpline2D:
        if isinstance(data, BSpline2D):
            return cls._normalize_spline(data)
        elif isinstance(data, np.ndarray):
            points = cls._remove_consecutive_duplicates(data)
            spline = BSpline2D(points)
            match output:
                case "spline":
                    return cls._normalize_spline(spline)
                case "points":
                    leading_edge, trailing_edge, _ = cls._find_leading_trailing_edges(spline)
                    scale, translation, rotation = cls._compute_transformation(leading_edge, trailing_edge)
                    return cls._apply_transformation(points, scale, translation, rotation)
                case _:
                    raise ValueError(f"Unknown output type: {output}")
        else:
            raise TypeError("Input must be a numpy array or BSpline2D.")

    @classmethod
    def _normalize_spline(cls, spline: BSpline2D,  find_trailing_edge: bool = False) -> BSpline2D:
        leading_edge, trailing_edge, u_leading_edge = cls._find_leading_trailing_edges(spline,  find_trailing_edge=find_trailing_edge)
        scale, translation, rotation = cls._compute_transformation(leading_edge, trailing_edge)

        # Apply transformation to control points
        # ctrl_transformed = rotation @ ((spline.spline[1] + translation) * scale)
        # ctrl_transformed = TransformedArray(ctrl_transformed, scale=scale, translation=translation, rotation=rotation)

        # Replace spline control points
        # spline.spline[1] = ctrl_transformed

        spline.spline[1] = cls._apply_transformation(
            spline.spline[1], scale, translation, rotation
        )

        # recompute
        leading_edge, trailing_edge, u_leading_edge = cls._find_leading_trailing_edges(spline,  find_trailing_edge=find_trailing_edge)

        # append additional information
        spline.leading_edge = leading_edge
        spline.trailing_edge = trailing_edge
        spline.u_leading_edge = u_leading_edge
        return spline

    @staticmethod
    def _apply_transformation(points: np.ndarray, scale: float, translation: np.ndarray, rotation: np.ndarray | float) -> np.ndarray:
        """Applies the transformation to an array of points.
        Args:
            points (np.ndarray): The points to be transformed.
            scale (float): The scaling factor.
            translation (np.ndarray): The translation vector.
            rotation (np.ndarray | float): The rotation matrix or angle in radians.
        Returns:
            np.ndarray(2, N): The transformed points.
        """
        points = np.asarray(points)
        # return (rotation @ ((points + translation) * scale).T).T.view(Point2D)
        if isinstance(rotation, float):
            rotation = np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation),  np.cos(rotation)]
            ])
        elif rotation.shape != (2, 2):
            raise ValueError("Rotation must be a 2x2 matrix or a rotation angle in radians.")

        if points.ndim == 2 and points.shape[0] != 2:
            points = points.T

        if translation.ndim == 1:
            translation = translation[:, np.newaxis]

        return TransformedArray(
            rotation @ ((points + translation) * scale),
            (scale, translation, rotation)
        )

    @staticmethod
    def _remove_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
        """Removes consecutive duplicate points from the array."""
        diff = np.diff(points, axis=0)
        idx = np.where(np.any(diff != 0, axis=1))[0] + 1
        return np.vstack([points[0], points[idx]])

    @staticmethod
    def _remove_overshoots(points: np.ndarray) -> np.ndarray:
        """Removes points that are outside the range x=[0, 1]."""
        return points[(points[:, 0] >= 0) & (points[:, 0] <= 1)]

    @staticmethod
    def _find_trailing_edge(spline: BSpline2D, find_trailing_edge=False, verbose=False) -> tuple[np.ndarray, np.ndarray]:
        """Finds the trailing edge of the airfoil. To account for cases where
        the trailing edge is ill-defined, or missing, there are two methods.
        First, the trailing edge is found by maximizing the distance from the
        leading edge. This is done by finding the point with the minimum
        negative L2 norm of the x-coordinate at the start and end of the spline
        separately.

        TODO : using the norm is redundant? just maximumize x-value?

        If the maximum x-values found for both sides of the spline are close
        enough together (wihtin some error tolerance), the trailing edge is
        likely well enough defined. In that case, the trailing edge is set to
        the midpoint of the start and end points of the spline.

        Args:
            spline (BSpline2D): The spline object representing the airfoil.
            find_trailing_edge (bool): Whether to solve for the trailing edge.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                The trailing edge point and the parameter ``u`` at which it
                occurs.

        Note:
            this is some janky a** code. It works (for now, I think, hard to
            tell whether there is some other stupid edgecase), but at what cost.
        """
        if find_trailing_edge:
            res1 = opt.minimize(lambda u: -np.linalg.norm(spline.evaluate_at(u)[0]), 0, bounds=[(0, 1)])
            res2 = opt.minimize(lambda u: -np.linalg.norm(spline.evaluate_at(u)[0]), 1, bounds=[(0, 1)])
            # res1 = opt.minimize(lambda u: -spline.evaluate_at(u)[0][0], 0, bounds=[(0, 1)])
            # res2 = opt.minimize(lambda u: -spline.evaluate_at(u)[0][0], 1, bounds=[(0, 1)])

            if not res1.success or not res2.success:
                raise RuntimeError(
                    "Failed to find trailing edge. \n" +
                    f"{str(res1)} \n {str(res2)}"
                )

            start, end = spline.evaluate_at(0), spline.evaluate_at(1)

            # if x-locations are not the same, trailing edge is assumed to be
            # at the maximum x-value found, while the other side is missing data
            if abs(start[0] - end[0]) > 1e-5:
                u_te = res1.x[0] if -res1.fun > -res2.fun else res2.x[0]
                trailing_edge = spline.evaluate_at(u_te)
                return trailing_edge #, u_te
            elif abs(start[0] - end[0]) < 1e-5:
                # if the maximum x values found are the same, the trailing edge
                # is assumed to be at the midpoint of the start and end points
                trailing_edge = 0.5 * (start + end)
            else:
                # this is probably never reached
                raise ValueError(
                    f"Unable to determine trailing edge. \n" +
                    f"Start: {start}, End: {end}, u1: {res1.x[0]}, u2: {res2.x[0]} \n" +
                    f"{res1} \n {res2}"
                    )

            # # if x-locations are not the same, trailing edge is assumed to be
            # # at the maximum x-value found, while the other side is missing data
            # if abs(res1.fun - res2.fun) > 1e-5:
            #     u_te = res1.x[0] if -res1.fun > -res2.fun else res2.x[0]
            #     trailing_edge = spline.evaluate_at(u_te)
            #     return trailing_edge #, u_te
            # else:
            #     # if the maximum x values found are the same, the trailing edge
            #     # is assumed to be at the midpoint of the start and end points
            #     start, end = spline.evaluate_at(0), spline.evaluate_at(1)
            #     if abs(start[0] - end[0]) < 1e-5:
            #         trailing_edge = 0.5 * (start + end)
            #     else:
            #         raise ValueError(
            #             f"Unable to determine trailing edge. \n" +
            #             f"Start: {start}, End: {end}, u1: {res1.x[0]}, u2: {res2.x[0]} \n" +
            #             f"{res1} \n {res2}"
            #             )
        else:
            # if argument says not to find trailing edge, this whole ordeal is
            # skipped and the trailing edge is assumed (hoped) to be the
            # midpoint of the defined end points.
            trailing_edge = 0.5*(spline.evaluate_at(0) + spline.evaluate_at(1))
            trailing_edge[0] = np.min([spline.evaluate_at(0)[0], spline.evaluate_at(1)[0]])
        return trailing_edge

    @staticmethod
    def _find_leading_edge(spline: BSpline2D, trailing_edge: Point2D) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the leading edge of the airfoil by maximizing the distance from
        the trailing edge. This is done by minimizing the negative L2 norm of
        the distance between the trailing edge and the spline points.

        Args:
            spline (BSpline2D):
                spline object representing the airfoil.
            trailing_edge (Point2D):
                The trailing edge point coordinates.

        Returns:
            tuple[np.ndarray, float]:
                The leading edge point and the parameter ``u`` at which it occurs.
        """
        def objective(u):
            return -np.linalg.norm(trailing_edge - spline.evaluate_at(u))

        res = opt.minimize(objective, 0.5, bounds=[(0, 1)])
        if not res.success:
            raise RuntimeError("Failed to find leading edge. \n" + str(res))
        return spline.evaluate_at(res.x[0]), res.x[0]

    @classmethod
    def _find_leading_trailing_edges(cls, spline: BSpline2D, find_trailing_edge=True) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Simply combines the two methods for leading and trailing edge in
        a single function. just for convenience.

        Args:
            spline (BSpline2D):
                The spline object representing the airfoil.
            find_trailing_edge (bool):
                Whether to solve for the trailing edge.

        Returns:
            tuple[np.ndarray, np.ndarray, float]:
                leading edge point, trailing edge point, and the parameter ``u``
                at which the leading edge occurs.
        """
        trailing_edge = cls._find_trailing_edge(spline, find_trailing_edge)
        leading_edge, u_leading_edge = cls._find_leading_edge(spline, trailing_edge)
        return leading_edge, trailing_edge, u_leading_edge

    @staticmethod
    def _compute_transformation(leading_edge, trailing_edge):
        chord = trailing_edge - leading_edge
        scale = 1.0 / np.linalg.norm(chord)
        translation = -leading_edge[:, np.newaxis]

        angle = -np.arctan2(chord[1], chord[0])
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        return (scale, translation, rotation)


class TransformedArray(Point2D):
    """
    Numpy Array subclass to store transformed points in a 2D array and retain
    the transformation applied during normalization.
    """
    def __new__(cls, input_array, transformation):
        obj = np.asarray(input_array).view(cls)
        obj.transformation = transformation
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.transformation = getattr(obj, 'transformation', None)

    @property
    def scale(self):
        """Returns the scaling factor which was applied."""
        return self.transformation[0]

    @property
    def translation(self):
        """Returns the applied translation vector."""
        return self.transformation[1]

    @property
    def rotation(self) -> float:
        """Returns the applied rotation angle in radians."""
        return self.transformation[2]
