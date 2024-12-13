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


def cosine_spacing(start: float, stop: float, num: int) -> np.ndarray:
    """Return cosine-spaced numbers over a specified interval.
    Returns `num` cosine-spaced samples, calculated over the
    interval [`start`, `stop`].

    Args:
        start (float): the starting value of the sequence.
        stop (float): the end value of the sequence.
        num (int): number of samples to generate. Must be non-negative.

    Returns:
        ndarray: `num` cosine-spaced samples in interval [`start`, `stop`]
    """
    return start + (stop - start) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=num)))

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


def weighted_endpoint_spacing(start, end, num_points, weight_func=np.sqrt):
    """Generate points with adjustable weighting for endpoint concentration."""
    linear_points = np.linspace(0, 1, num_points)
    weighted_points = weight_func(linear_points) / weight_func(1)
    return start + (end - start) * weighted_points
