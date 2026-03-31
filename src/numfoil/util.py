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

"""Contains utility functions for pressure coefficient analysis."""

from typing import Sequence, Tuple, Union

import numpy as np
from math import comb
from scipy.interpolate import interp1d


class Container:
    """
    I don't like the dict() syntax.
    """
    def __init__(self, **kwargs):
        "Set initial values"
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        "Redundant but whatever"
        self.__dict__[name] = value

    def __call__(self):
        print("\n".join(f"{k}: {v!r}," for k, v in self.__dict__.items()))


def ensure_1d_vector(x: Union[float, np.ndarray]) -> np.ndarray:
    """Ensures that ``x`` is a 1D vector."""
    x = np.array([x]) if isinstance(x, (float, int)) else x
    if len(x.shape) != 1:
        raise ValueError("Only 1-D np.arrays are supported")
    return x


def delta_cp_from_cp(
    x_values: Sequence[float], pressure_coefficients: Sequence[float], num=1000
):
    """Assuming CCW airfoil TE -> Upper -> LE -> Lower -> TE."""
    (x_top, cp_top), (x_bot, cp_bot) = split_at_le(
        x_values, pressure_coefficients
    )

    interp1d_kwargs = {"kind": "cubic", "fill_value": "extrapolate"}

    f_top = interp1d(
        tuple(reversed(x_top)), tuple(reversed(cp_top)), **interp1d_kwargs
    )
    f_bot = interp1d(x_bot, cp_bot, **interp1d_kwargs)

    sample_x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=num)))

    return sample_x, f_bot(sample_x) - f_top(sample_x)


def split_at_le(
    x_values: Sequence[float], data: Sequence[float]
) -> Tuple[Tuple[Sequence[float], Sequence[float]]]:
    """Splits ``data`` at the leading-edge of the airfoil.

    The leading-edge is assumed to be the first occurance of the
    minimum in ``x_values``.
    """
    le_idx = x_values.index(min(x_values))
    return (
        (x_values[: le_idx + 1], data[: le_idx + 1]),
        (x_values[le_idx + 1 :], data[le_idx + 1 :]),
    )


def split_upper_lower(
    points: np.ndarray, le_tol: float = 1e-6, window: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Split Selig-format coordinates (
    upper TE→LE, lower LE→TE) into upper/lower arrays.

    Args:
        points (ndarray): Airfoil coordinates in Selig format.
        le_tol (float): Tolerance for detecting explicit leading-edge point at
        (0,0). Defaults to 1e-6.
        window (int): Number of points on either side of min-x point to search
        for y-sign change. Defaults to 3.

    Returns:
        tuple[ndarray, ndarray]: Upper and lower surface coordinates.

    Raises:
        ValueError: If no sign change is found near the leading edge.
    """
    pts = np.asarray(points, dtype=float)
    le_idx = int(np.argmin(pts[:, 0]))

    # Explicit [0,0] leading edge: share point between surfaces
    if np.allclose(pts[le_idx], 0.0, atol=le_tol):
        return pts[: le_idx + 1].copy(), pts[le_idx:].copy()

    # Find sign change in y within window around min-x point
    start, end = max(0, le_idx - window), min(len(pts), le_idx + window + 1)
    y_signs = np.sign(pts[start:end, 1])
    change = np.flatnonzero(y_signs[:-1] != y_signs[1:])

    if len(change) == 0:
        raise ValueError("no sign change near leading edge")

    split_idx = start + change[0] + 1
    upper, lower = pts[:split_idx].copy(), pts[split_idx:].copy()

    # Ensure upper surface has positive y near LE
    if upper[-1, 1] < lower[0, 1]:
        upper, lower = lower[::-1], upper[::-1]

    return upper, lower


def cosine_spacing(start: float, stop: float, num: int, a: float = 1.0) -> np.ndarray:
    """Return cosine-spaced numbers over a specified interval.
    Returns `num` cosine-spaced samples, calculated over the
    interval [`start`, `stop`].

    Args:
        start (float): the starting value of the sequence.
        stop (float): the end value of the sequence.
        num (int): number of samples to generate. Must be non-negative.
        a (float): endpoint-concentration parameter. Increases concentration
                near the endpoints when a>1.Default is 1.0.

    Returns:
        ndarray: `num` cosine-spaced samples in interval [`start`, `stop`]
    """
    # return start + (stop - start) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=num)))
    return start + (stop - start) * 0.5 * (
        1
        - np.sign(0.5 - 0.5 * (1 - np.cos(np.linspace(0, np.pi, num))))
        * np.abs(2 * 0.5 * (1 - np.cos(np.linspace(0, np.pi, num))) - 1) ** a
    )


def chebyshev_nodes(start: float, end:float , num: int) -> np.ndarray:
    """Return Chebyshev-Lobatto nodes over a specified interval.
    Chebyshev-Lobatto nodes cluster more densely at the interval ends and
    include the end-points.

    Currently appends the endpoints to the chebyshev sequence.
    Another option is to map the nodes from the interval [-1, 1] to the
    interval [`start`, `end`] directly instead of first mapping to [0, 1] and
    then scaling:
        # nodes = np.cos(np.pi * (num - 1 - np.arange(num)) / (num - 1))
        # return 0.5 * (end - start) * (nodes + 1) + start

    However, the current implementation results in higher concentration of
    nodes at the interval ends.

    Args:
        start (float): the starting value of the sequence.
        end (float): the end value of the sequence.
        num (int): number of samples to generate. Must be non-negative.

    Returns:
        np.ndarray: `num` Chebyshev nodes in interval [`start`, `end`]
    """
    num -= 2                            # account for the end points
    nodes = np.cos((2 * np.arange(num) + 1) / (2 * num) * np.pi)
    nodes = 0.5 * (1 - nodes)           # Map [-1,1] Chebyshev nodes to [0,1]
    return np.concatenate((
        [0],                            # append start
        start + (end - start) * nodes,  # scale nodes to interval
        [1]                             # append end
        ))


def selig(array) -> np.ndarray:
    """Return a given array of x-locations in Selig format."""
    if array[0] != 0.0 or array[-1] != 1.0 or not np.all(np.diff(array) > 0):
        raise ValueError(
            "Array must start with 0 and end with 1 and be strictly increasing."
        )
    return np.append(array[::-1], array[1:])


def weighted_endpoint_spacing(start, end, num_points, weight_func=np.sqrt):
    """Generate points with adjustable weighting for endpoint concentration."""
    linear_points = np.linspace(0, 1, num_points)
    weighted_points = weight_func(linear_points) / weight_func(1)
    return start + (end - start) * weighted_points


# def smootherstep(x):
#     """Smootherstep function for smooth endpoint tapering."""
#     return 6 * x**5 - 15 * x**4 + 10 * x**3

def smoothstep(x, N=2, deriv=0):
    """
    Generalized smoothstep S_N(x)

    Args:
        x (float or np.ndarray):
            Input value(s), expected in [0, 1]
        N (int):
            Order of smoothstep (polynomial degree = 2N + 1)
            Default is 2, which corresponds to the smootherstep
            :math:`S_2(x) = 6x^5 - 15x^4 + 10x^3`
        deriv (int {0,1,2}):
            - 0 -> function value `S_N(x)`
            - 1 -> first derivative `S_N'(x)`
            - 2 -> second derivative `S_N''(x)`

    Returns:
        float or np.ndarray, function evaluation(s) of S_N(x) or its derivatives.
    """
    # # I guess this does not matter...
    # EPS = 1e-10
    # if x.min() < -EPS or x.max() > 1 + EPS:
    #     raise ValueError(
    #         "Input x must be in the range [0, 1], "
    #         "but got values in [{:.3g}, {:.3g}]".format(x.min(), x.max())
    #     )
    # x = np.clip(x, 0.0, 1.0)

    # clip just in case
    x = np.asarray(x).clip(0.0, 1.0)

    match deriv:
        case 0:
            S = np.zeros_like(x, dtype=float)
            for n in range(N + 1):
                S += (
                    (-1)**n
                    * comb(N + n, n)
                    * comb(2 * N + 1, N - n)
                    * x**(N + n + 1)
                )
            return S

        case 1:
            return (
                (2 * N + 1)
                * comb(2 * N, N)
                * (x - x**2)**N
            )

        case 2:
            return (
                (2 * N + 1)
                * comb(2 * N, N)
                * N
                * (x - x**2)**(N - 1)
                * (1 - 2 * x)
            )

        case _:
            raise ValueError(
                "derivative must be 0, 1, or 2, "
                f"but got {deriv}."
            )

