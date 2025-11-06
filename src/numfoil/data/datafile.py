from functools import cached_property
import numpy as np
from typing import Tuple
import scipy.interpolate as si
import scipy.optimize as opt
from ..geometry.spline import BSpline2D, SplevCBezier
from ..geometry.geom2d import Point2D, Geom2D
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


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
            str | None: The header lines as a single multi-line string.
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

