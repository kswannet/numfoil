# Standalone NeuralFoil solver with XFoil-compatible interface (the wrapper, not
# actual xfoil itself), using the original Neuralfoil Numpy approach.
#
# This module reimplements NeuralFoil inference logic directly and returns
# AirfoilResults/PolarData/CpData/DumpData containers for API compatibility with
# numfoil's class-based XFoil solver.


# This implementation is based on Neuralfoil, subject to following license:
# MIT License
#
# Copyright (c) 2023-2023 Peter Sharpe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from pathlib import Path
from warnings import warn as warning

import numpy as np

from numfoil.aero.aeroresults import AeroResults
from numfoil.data.normalization import AirfoilNormalizer
from numfoil.geometry.airfoil import KulfanAirfoil

_eps: float = 10 / np.finfo(np.array(1.0).dtype).max
_ln_eps: float = np.log(_eps)


def _to_1d_array(value) -> np.ndarray:
    """Return value as a 1D numpy array."""
    return np.asarray(value).reshape(-1)


def _length(value) -> int:
    """Length helper compatible with scalar and vector inputs."""
    arr = np.asarray(value)
    return 1 if arr.ndim == 0 else arr.size


def _sind(x):
    """Sine of degrees."""
    return np.sin(np.deg2rad(x))


def _cosd(x):
    """Cosine of degrees."""
    return np.cos(np.deg2rad(x))


def _swish(x):
    """Swish activation."""
    x = np.asarray(x)
    x_clip = np.clip(x, _ln_eps, -_ln_eps)
    return x / (1 + np.exp(-x_clip))


def _normalize_batch_size(batch_size, total_cases: int) -> int:
    """Normalize a user batch size to an effective positive chunk size.

    Args:
        batch_size: User-provided batch size.
            - 0 means disabled (caller decides fallback path)
            - inf means one full batch
        total_cases (int): Total number of flattened cases.

    Returns:
        int: Effective batch size in [1, total_cases], or 0 when disabled.
    """
    if batch_size is None:
        return 0
    if batch_size == 0:
        return 0
    if np.isinf(batch_size):
        return int(total_cases)

    out = int(batch_size)
    if out < 0:
        raise ValueError("batch_size must be >= 0 (or inf).")
    if out == 0:
        return 0
    return min(out, int(total_cases))


def as_re_list(reynolds):
    """Normalize Reynolds input to a list of floats."""
    if isinstance(reynolds, (int, float, np.floating)):
        return [float(reynolds)]
    return [float(r) for r in reynolds]


def looks_like_points(value) -> bool:
    """Return True when *value* is a single ``(N, 2)`` point array."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.shape[1] == 2


def as_airfoil_batch(airfoil):
    """Normalize airfoil input to a list of airfoil-like objects."""
    if hasattr(airfoil, "points") or looks_like_points(airfoil):
        return [airfoil]

    if isinstance(airfoil, np.ndarray):
        arr = np.asarray(airfoil, dtype=float)
        if arr.ndim == 3 and arr.shape[2] == 2:
            return [arr[i] for i in range(arr.shape[0])]

    if isinstance(airfoil, (list, tuple)):
        if len(airfoil) == 0:
            raise ValueError("airfoil list cannot be empty")
        if looks_like_points(airfoil):
            return [np.asarray(airfoil, dtype=float)]
        return list(airfoil)

    return [airfoil]


def as_label_batch(
    label,
    n_items: int,
    fallback_labels=None,
    default_prefix: str | None = None,
    coerce_str: bool = False,
):
    """Normalize labels to exactly one entry per airfoil."""
    if label is None:
        if fallback_labels is not None:
            if len(fallback_labels) != n_items:
                raise ValueError("fallback label length mismatch")
            if coerce_str:
                return [str(value) for value in fallback_labels]
            return list(fallback_labels)
        if default_prefix is not None:
            return [f"{default_prefix}_{i + 1}" for i in range(n_items)]
        return [None] * n_items

    if isinstance(label, str):
        if n_items == 1:
            return [label]
        return [f"{label}_{i + 1}" for i in range(n_items)]

    labels = list(label)
    if len(labels) != n_items:
        raise ValueError(
            "label must be a single string or have the same length as "
            "the airfoil batch"
        )
    if coerce_str:
        return [str(value) for value in labels]
    return labels


class NeuralFoil:
    """Standalone NeuralFoil inference solver.

    Public method handles mirror the class-based XFoil API:
    - get_polar
    - get_cp
    - get_dump
    - analyze

    Inputs can be provided in exactly one of three mutually exclusive modes:
    - airfoil geometry: numfoil airfoil object with .points, or raw Selig points
    - full Kulfan parameter array
    - split Kulfan kwargs (upper/lower/leading-edge/trailing-edge)

    Args:
        model_size (str): Neural network size key.
        parameter_file (str | Path | None): Optional path to a single .pth file
            containing all NeuralFoil setup data (model state dicts and input
            distribution statistics).
    """

    def __init__(
        self,
        model_size: str = "xxxlarge",
    ):
        parameter_path = Path(__file__).resolve().parent / "NeuralFoilParameters.pth"

        if not parameter_path.is_file():
            raise FileNotFoundError(
                "Unable to locate NeuralFoil parameter file: "
                f"{parameter_path}"
            )

        import torch

        parameter_data = torch.load(parameter_path, map_location="cpu")
        self.parameter_file = parameter_path

        self._scaled_input_distribution = {
            "mean_inputs_scaled": np.asarray(parameter_data["mean_inputs_scaled"]),
            "inv_cov_inputs_scaled": np.asarray(parameter_data["inv_cov_inputs_scaled"]),
        }
        self._scaled_input_distribution["N_inputs"] = len(
            self._scaled_input_distribution["mean_inputs_scaled"]
        )

        self._nn_parameters = {}
        for key, value in parameter_data.items():
            if not key.endswith("_state_dict"):
                continue
            model_key = key[: -len("_state_dict")]
            self._nn_parameters[model_key] = {
                name: tensor.detach().cpu().numpy()
                for name, tensor in value.items()
            }
        if not self._nn_parameters:
            raise ValueError(
                "Parameter file does not contain any '<model>_state_dict' entries."
            )

        self._allowable_model_sizes = set(self._nn_parameters)

        if model_size not in self._allowable_model_sizes:
            raise ValueError(
                f"Invalid model_size={model_size!r}. Must be one of "
                f"{sorted(self._allowable_model_sizes)}."
            )

        self.model_size = model_size
        self.bl_x_points = self._compute_bl_x_points(32)

    def __repr__(self):
        return (
            f"NeuralFoil(model_size={self.model_size!r}, "
            f"parameter_file={str(self.parameter_file)!r})"
        )

    @staticmethod
    def _compute_bl_x_points(n_points: int) -> np.ndarray:
        """Compute BL x-locations used by NeuralFoil outputs."""
        s = np.linspace(0, 1, n_points + 1)
        return (s[1:] + s[:-1]) / 2

    @property
    def n_bl_points(self) -> int:
        """Number of per-surface BL points."""
        return len(self.bl_x_points)

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid with clipping."""
        x = np.clip(x, _ln_eps, -_ln_eps)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _progress(re_list, n_alpha):
        """Progress printout matching XFoil style."""
        n = len(re_list)
        print(f"NeuralFoil: {n} session{'s' if n != 1 else ''}, {n_alpha} alpha each")
        for i, re_value in enumerate(re_list, 1):
            print(f"  [{i}/{n}] Re = {re_value:.2e}")
            yield re_value

    def _build_kulfan_from_points(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Create Kulfan parameters from Selig points using numfoil fitting."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        spline = AirfoilNormalizer.normalized_bspline(pts)

        airfoil = KulfanAirfoil.fit(
            spline.evaluate_at(np.linspace(0.0, spline.u_leading_edge, 200))[::-1],
            spline.evaluate_at(np.linspace(spline.u_leading_edge, 1.0, 200)),
            n_coefficients=8
        )

        return {
            "upper_weights": airfoil.upper_surface.coefficients,
            "lower_weights": airfoil.lower_surface.coefficients,
            "leading_edge_weight": 0.0,
            "TE_thickness": airfoil.trailing_edge_thickness,
        }

    def _parse_kulfan_array(self, kulfan_parameters: np.ndarray) -> dict[str, np.ndarray]:
        """Parse full Kulfan vector [upper..., lower..., w_le, t_te]."""
        arr = np.asarray(kulfan_parameters, dtype=float).reshape(-1)
        if arr.size < 4:
            raise ValueError("Kulfan parameter array must have at least 4 values.")
        if arr.size % 2 != 0:
            raise ValueError(
                "Kulfan parameter array must have even length: 2*n_coeffs + 2."
            )

        n_coeff = (arr.size - 2) // 2
        if n_coeff != 8:
            raise ValueError(
                "NeuralFoil expects 8 upper and 8 lower CST coefficients, "
                f"got {n_coeff} per side."
            )

        return {
            "upper_weights": arr[:n_coeff],
            "lower_weights": arr[n_coeff:-2],
            "leading_edge_weight": float(arr[-2]),
            "TE_thickness": float(arr[-1]),
        }

    def _resolve_kulfan_parameters(
        self,
        airfoil: object | np.ndarray | None = None,
        kulfan_parameters: np.ndarray | None = None,
        upper_weights: np.ndarray | None = None,
        lower_weights: np.ndarray | None = None,
        leading_edge_weight: float | None = None,
        trailing_edge_thickness: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Resolve exactly one input mode into Kulfan parameter dict."""
        has_airfoil = airfoil is not None
        has_kulfan = kulfan_parameters is not None
        has_split = upper_weights is not None or lower_weights is not None

        active = int(has_airfoil) + int(has_kulfan) + int(has_split)
        if active != 1:
            raise ValueError(
                "Provide exactly one input mode: airfoil, kulfan_parameters, or split Kulfan weights."
            )

        if has_kulfan:
            return self._parse_kulfan_array(kulfan_parameters)

        if has_split:
            if upper_weights is None or lower_weights is None:
                raise ValueError("Both upper_weights and lower_weights are required in split mode.")
            upper = np.asarray(upper_weights, dtype=float).reshape(-1)
            lower = np.asarray(lower_weights, dtype=float).reshape(-1)
            if upper.size != lower.size:
                raise ValueError("upper_weights and lower_weights must have same length.")
            if upper.size != 8:
                raise ValueError(
                    "NeuralFoil expects exactly 8 upper and 8 lower CST coefficients."
                )

            w_le = 0.0 if leading_edge_weight is None else float(leading_edge_weight)
            t_te = 0.0 if trailing_edge_thickness is None else float(trailing_edge_thickness)

            return {
                "upper_weights": upper,
                "lower_weights": lower,
                "leading_edge_weight": w_le,
                "TE_thickness": t_te,
            }

        # Airfoil mode
        if hasattr(airfoil, "params"):
            return self._parse_kulfan_array(np.asarray(airfoil.params, dtype=float))
        if hasattr(airfoil, "parameters"):
            return self._parse_kulfan_array(np.asarray(airfoil.parameters, dtype=float))

        pts = airfoil.points if hasattr(airfoil, "points") else airfoil
        pts_arr = np.asarray(pts)

        # Treat 1D vectors passed as first arg as direct Kulfan input.
        if pts_arr.ndim == 1:
            return self._parse_kulfan_array(pts_arr)

        if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
            raise TypeError(
                "airfoil must be an object with .points, Selig point array (N,2), "
                "or direct Kulfan vector."
            )
        return self._build_kulfan_from_points(pts_arr)

    def _evaluate_from_kulfan(
        self,
        kulfan: dict[str, np.ndarray],
        alpha,
        re_value,
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0,
    ) -> dict[str, np.ndarray]:
        """Evaluate NeuralFoil outputs from Kulfan parameters."""

        model_size = self.model_size
        if model_size not in self._allowable_model_sizes:
            raise ValueError(
                f"Invalid model_size={model_size!r}. Must be one of {sorted(self._allowable_model_sizes)}."
            )

        nn_params = self._nn_parameters[model_size]

        input_rows = [
            *[kulfan["upper_weights"][i] for i in range(8)],
            *[kulfan["lower_weights"][i] for i in range(8)],
            kulfan["leading_edge_weight"],
            kulfan["TE_thickness"] * 50,
            _sind(2 * alpha),
            _cosd(alpha),
            1 - _cosd(alpha) ** 2,
            (np.log(re_value) - 12.5) / 3.5,
            (n_crit - 9) / 4.5,
            xtr_upper,
            xtr_lower,
        ]

        n_cases = 1
        for row in input_rows:
            if _length(row) > 1:
                if n_cases == 1:
                    n_cases = _length(row)
                elif _length(row) != n_cases:
                    raise ValueError(
                        "All vectorized inputs must have the same length. "
                        f"Conflicting lengths: {n_cases} and {_length(row)}"
                    )

        for i, row in enumerate(input_rows):
            input_rows[i] = np.ones(n_cases) * row

        x = np.stack(input_rows, axis=1)

        try:
            layer_indices = sorted(
                {int(key.split(".")[1]) for key in nn_params.keys() if key.startswith("net.")}
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("Unexpected neural-network parameter key format.") from exc

        def net(x_in: np.ndarray) -> np.ndarray:
            x_loc = np.transpose(x_in)
            remaining = layer_indices.copy()
            while remaining:
                i_layer = remaining.pop(0)
                w = nn_params[f"net.{i_layer}.weight"]
                b = nn_params[f"net.{i_layer}.bias"]
                x_loc = w @ x_loc + np.reshape(b, (-1, 1))
                if remaining:
                    x_loc = _swish(x_loc)
            return np.transpose(x_loc)

        y = net(x)
        y[:, 0] = y[:, 0] - self._squared_mahalanobis_distance(x) / (
            2 * self._scaled_input_distribution["N_inputs"]
        )

        x_flipped = x + 0.0
        x_flipped[:, :8] = x[:, 8:16] * -1
        x_flipped[:, 8:16] = x[:, :8] * -1
        x_flipped[:, 16] = -1 * x[:, 16]
        x_flipped[:, 18] = -1 * x[:, 18]
        x_flipped[:, 23] = x[:, 24]
        x_flipped[:, 24] = x[:, 23]

        y_flipped = net(x_flipped)
        y_flipped[:, 0] = y_flipped[:, 0] - self._squared_mahalanobis_distance(x_flipped) / (
            2 * self._scaled_input_distribution["N_inputs"]
        )

        y_unflipped = y_flipped + 0.0
        y_unflipped[:, 1] = y_flipped[:, 1] * -1
        y_unflipped[:, 3] = y_flipped[:, 3] * -1
        y_unflipped[:, 4] = y_flipped[:, 5]
        y_unflipped[:, 5] = y_flipped[:, 4]

        n_bl = self.n_bl_points
        y_unflipped[:, 6 : 6 + n_bl * 2] = y_flipped[:, 6 + n_bl * 3 : 6 + n_bl * 5]
        y_unflipped[:, 6 + n_bl * 3 : 6 + n_bl * 5] = y_flipped[:, 6 : 6 + n_bl * 2]

        y_unflipped[:, 6 + n_bl * 2 : 6 + n_bl * 3] = -1 * y_flipped[:, 6 + n_bl * 5 : 6 + n_bl * 6]
        y_unflipped[:, 6 + n_bl * 5 : 6 + n_bl * 6] = -1 * y_flipped[:, 6 + n_bl * 2 : 6 + n_bl * 3]

        y_fused = (y + y_unflipped) / 2
        y_fused[:, 0] = self._sigmoid(y_fused[:, 0])
        y_fused[:, 4] = np.clip(y_fused[:, 4], 0, 1)
        y_fused[:, 5] = np.clip(y_fused[:, 5], 0, 1)

        analysis_confidence = y_fused[:, 0]
        cl = y_fused[:, 1] / 2
        cd = np.exp((y_fused[:, 2] - 2) * 2)
        cm = y_fused[:, 3] / 20
        top_xtr = y_fused[:, 4]
        bot_xtr = y_fused[:, 5]

        upper_bl_ue_over_vinf = y_fused[:, 6 + n_bl * 2 : 6 + n_bl * 3]
        lower_bl_ue_over_vinf = y_fused[:, 6 + n_bl * 5 : 6 + n_bl * 6]

        upper_theta = ((10 ** y_fused[:, 6 : 6 + n_bl]) - 0.1) / (
            np.abs(upper_bl_ue_over_vinf) * np.reshape(re_value, (-1, 1))
        )
        upper_h = 2.6 * np.exp(y_fused[:, 6 + n_bl : 6 + n_bl * 2])

        lower_theta = ((10 ** y_fused[:, 6 + n_bl * 3 : 6 + n_bl * 4]) - 0.1) / (
            np.abs(lower_bl_ue_over_vinf) * np.reshape(re_value, (-1, 1))
        )
        lower_h = 2.6 * np.exp(y_fused[:, 6 + n_bl * 4 : 6 + n_bl * 5])

        result = {
            "analysis_confidence": analysis_confidence,
            "CL": cl,
            "CD": cd,
            "CM": cm,
            "Top_Xtr": top_xtr,
            "Bot_Xtr": bot_xtr,
            **{f"upper_bl_theta_{i}": upper_theta[:, i] for i in range(n_bl)},
            **{f"upper_bl_H_{i}": upper_h[:, i] for i in range(n_bl)},
            **{f"upper_bl_ue/vinf_{i}": upper_bl_ue_over_vinf[:, i] for i in range(n_bl)},
            **{f"lower_bl_theta_{i}": lower_theta[:, i] for i in range(n_bl)},
            **{f"lower_bl_H_{i}": lower_h[:, i] for i in range(n_bl)},
            **{f"lower_bl_ue/vinf_{i}": lower_bl_ue_over_vinf[:, i] for i in range(n_bl)},
        }
        return {key: np.reshape(value, -1) for key, value in result.items()}

    def _evaluate_by_re(
        self,
        kulfan: dict[str, np.ndarray],
        alpha_arr: np.ndarray,
        re_list: list[float],
        n_crit: float,
        xtr_upper: float,
        xtr_lower: float,
        batch_size=0,
    ) -> dict[float, dict[str, np.ndarray]]:
        """Evaluate NeuralFoil outputs grouped by Re.

        Args:
            kulfan: Kulfan parameter dictionary.
            alpha_arr: 1D angle array.
            re_list: Reynolds list.
            n_crit: N-crit value.
            xtr_upper: Forced upper transition.
            xtr_lower: Forced lower transition.
            batch_size: Chunk size for flattened Re×alpha batch.
                0 keeps the original per-Re loop.

        Returns:
            dict[float, dict[str, np.ndarray]]: Nested mapping grouped by Re.
        """
        n_alpha = len(alpha_arr)
        n_re = len(re_list)

        effective_bs = _normalize_batch_size(batch_size, n_alpha * n_re)

        # Safe/default behavior: one forward pass per Re (original path).
        if effective_bs == 0:
            out_by_re: dict[float, dict[str, np.ndarray]] = {}
            for re_value in self._progress(re_list, n_alpha):
                aero = self._evaluate_from_kulfan(
                    kulfan=kulfan,
                    alpha=alpha_arr,
                    re_value=re_value,
                    n_crit=n_crit,
                    xtr_upper=xtr_upper,
                    xtr_lower=xtr_lower,
                )
                out_by_re[re_value] = {
                    key: np.asarray(val)
                    for key, val in aero.items()
                }
            return out_by_re

        total = n_alpha * n_re
        print(
            "NeuralFoil: batched inference "
            f"({n_re} Re x {n_alpha} alpha = {total} cases), "
            f"chunk={effective_bs}"
        )

        alpha_flat = np.tile(alpha_arr, n_re)
        re_flat = np.repeat(np.asarray(re_list, dtype=float), n_alpha)

        storage: dict[str, np.ndarray] | None = None

        for start in range(0, total, effective_bs):
            end = min(start + effective_bs, total)
            aero = self._evaluate_from_kulfan(
                kulfan=kulfan,
                alpha=alpha_flat[start:end],
                re_value=re_flat[start:end],
                n_crit=n_crit,
                xtr_upper=xtr_upper,
                xtr_lower=xtr_lower,
            )

            if storage is None:
                storage = {
                    key: np.empty((n_re, n_alpha), dtype=np.asarray(val).dtype)
                    for key, val in aero.items()
                }

            idx = np.arange(start, end)
            re_idx = idx // n_alpha
            alpha_idx = idx % n_alpha

            for key, val in aero.items():
                storage[key][re_idx, alpha_idx] = np.asarray(val)

        assert storage is not None
        out_by_re = {}
        for i, re_value in enumerate(re_list):
            out_by_re[re_value] = {
                key: storage[key][i].copy()
                for key in storage
            }
        return out_by_re

    def _squared_mahalanobis_distance(self, x: np.ndarray) -> np.ndarray:
        """Compute squared Mahalanobis distance in latent input space."""
        d = self._scaled_input_distribution
        mean = np.reshape(d["mean_inputs_scaled"], (1, -1))
        x_minus_mean = (x.T - mean.T).T
        return np.sum(x_minus_mean @ d["inv_cov_inputs_scaled"] * x_minus_mean, axis=1)

    def _build_dump_raw(self, aero: dict[str, np.ndarray], alpha_idx: int) -> dict[str, np.ndarray]:
        """Build DumpData-style arrays from NeuralFoil BL outputs."""
        n = self.n_bl_points
        x_u = np.asarray(self.bl_x_points[::-1])
        x_l = np.asarray(self.bl_x_points)
        x = np.concatenate([x_u, x_l])

        upper_theta = np.array([aero[f"upper_bl_theta_{i}"][alpha_idx] for i in range(n)])
        lower_theta = np.array([aero[f"lower_bl_theta_{i}"][alpha_idx] for i in range(n)])
        upper_h = np.array([aero[f"upper_bl_H_{i}"][alpha_idx] for i in range(n)])
        lower_h = np.array([aero[f"lower_bl_H_{i}"][alpha_idx] for i in range(n)])
        upper_ue = np.array([aero[f"upper_bl_ue/vinf_{i}"][alpha_idx] for i in range(n)])
        lower_ue = np.array([aero[f"lower_bl_ue/vinf_{i}"][alpha_idx] for i in range(n)])

        theta = np.concatenate([upper_theta[::-1], lower_theta])
        h_shape = np.concatenate([upper_h[::-1], lower_h])
        ue_over_vinf = np.concatenate([upper_ue[::-1], lower_ue])
        dstar = theta * h_shape

        return {
            "s": np.linspace(0.0, 1.0, len(x)),
            "x": x,
            "y": np.full_like(x, np.nan),
            "Ue/Vinf": ue_over_vinf,
            "Dstar": dstar,
            "Theta": theta,
            "Cf": np.full_like(x, np.nan),
        }

    @staticmethod
    def _resolve_label(airfoil, fallback: str = "NeuralFoil") -> str:
        """Resolve a display label from input geometry."""
        if airfoil is None:
            return fallback
        label = getattr(airfoil, "name", None) or getattr(airfoil, "description", None)
        if label:
            return str(label).strip().replace(" ", "_")
        if hasattr(airfoil, "points"):
            pts = np.asarray(airfoil.points).reshape(-1, 2)
            return f"airfoil_{hash(pts.tobytes()) % 0xFFFF:04x}"
        arr = np.asarray(airfoil)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return f"airfoil_{hash(arr.tobytes()) % 0xFFFF:04x}"
        return fallback

    @staticmethod
    def _expand_batch_param(value, n_items: int, default: float) -> np.ndarray:
        """Expand scalar/batch optional params to length ``n_items``."""
        if value is None:
            return np.full(n_items, default, dtype=float)

        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(n_items, float(arr[0]), dtype=float)
        if arr.size != n_items:
            raise ValueError(
                "optional batch parameter must be scalar or match "
                "airfoil batch length"
            )
        return arr

    def _resolve_batch_inputs(
        self,
        airfoil,
        label,
        kulfan_parameters,
        upper_weights,
        lower_weights,
        leading_edge_weight,
        trailing_edge_thickness,
    ) -> tuple[list[str], list[dict[str, np.ndarray]]]:
        """Resolve one input mode into labels and per-airfoil Kulfan dicts."""
        has_airfoil = airfoil is not None
        has_kulfan = kulfan_parameters is not None
        has_split = upper_weights is not None or lower_weights is not None

        active = int(has_airfoil) + int(has_kulfan) + int(has_split)
        if active != 1:
            raise ValueError(
                "Provide exactly one input mode: airfoil, "
                "kulfan_parameters, or split Kulfan weights."
            )

        if has_airfoil:
            airfoils = as_airfoil_batch(airfoil)
            kulfan_list = [
                self._resolve_kulfan_parameters(airfoil=item)
                for item in airfoils
            ]
            fallback = [
                self._resolve_label(item, fallback=f"airfoil_{i + 1}")
                for i, item in enumerate(airfoils)
            ]
            labels = as_label_batch(
                label,
                len(kulfan_list),
                fallback_labels=fallback,
                coerce_str=True,
            )
            return labels, kulfan_list

        if has_kulfan:
            arr = np.asarray(kulfan_parameters, dtype=float)
            if arr.ndim == 1:
                kulfan_list = [self._parse_kulfan_array(arr)]
            elif arr.ndim == 2:
                kulfan_list = [
                    self._parse_kulfan_array(row)
                    for row in arr
                ]
            else:
                raise ValueError(
                    "kulfan_parameters must be shape (18,) or (B, 18)"
                )

            labels = as_label_batch(
                label,
                len(kulfan_list),
                default_prefix="airfoil",
                coerce_str=True,
            )
            return labels, kulfan_list

        if upper_weights is None or lower_weights is None:
            raise ValueError(
                "Both upper_weights and lower_weights are required in split mode."
            )

        upper = np.asarray(upper_weights, dtype=float)
        lower = np.asarray(lower_weights, dtype=float)
        if upper.shape != lower.shape:
            raise ValueError(
                "upper_weights and lower_weights must have the same shape"
            )

        if upper.ndim == 1:
            upper = upper.reshape(1, -1)
            lower = lower.reshape(1, -1)
        elif upper.ndim != 2:
            raise ValueError(
                "split Kulfan weights must be shape (8,) or (B, 8)"
            )

        if upper.shape[1] != 8:
            raise ValueError(
                "NeuralFoil expects exactly 8 upper and 8 lower coefficients"
            )

        n_items = upper.shape[0]
        le_batch = self._expand_batch_param(
            leading_edge_weight,
            n_items,
            default=0.0,
        )
        te_batch = self._expand_batch_param(
            trailing_edge_thickness,
            n_items,
            default=0.0,
        )

        kulfan_list = []
        for i in range(n_items):
            kulfan_list.append(
                {
                    "upper_weights": upper[i],
                    "lower_weights": lower[i],
                    "leading_edge_weight": float(le_batch[i]),
                    "TE_thickness": float(te_batch[i]),
                }
            )

        labels = as_label_batch(
            label,
            n_items,
            default_prefix="airfoil",
            coerce_str=True,
        )
        return labels, kulfan_list

    def _build_result_from_aero_by_re(
        self,
        label: str,
        Mach: float,
        alpha_arr: np.ndarray,
        re_list: list[float],
        aero_by_re: dict[float, dict[str, np.ndarray]],
    ) -> AeroResults:
        """Build an AeroResults object from grouped Re outputs."""
        result = AeroResults(label, Mach, source="NeuralFoil_np_numfoil")

        for re_value in re_list:
            aero = aero_by_re[re_value]

            result.polar._add(
                re_value,
                {
                    "alpha": np.asarray(alpha_arr),
                    "CL": np.asarray(aero["CL"]),
                    "CD": np.asarray(aero["CD"]),
                    "CM": np.asarray(aero["CM"]),
                    "Top_Xtr": np.asarray(aero["Top_Xtr"]),
                    "Bot_Xtr": np.asarray(aero["Bot_Xtr"]),
                    "analysis_confidence": np.asarray(
                        aero["analysis_confidence"]
                    ),
                },
            )

            for i, alpha in enumerate(alpha_arr):
                result.dump._add(
                    float(alpha),
                    re_value,
                    self._build_dump_raw(aero, i),
                )

        return result

    def _evaluate_multi_airfoil(
        self,
        labels: list[str],
        kulfan_list: list[dict[str, np.ndarray]],
        alpha_arr: np.ndarray,
        re_list: list[float],
        n_crit: float,
        xtr_upper: float,
        xtr_lower: float,
    ) -> dict[str, dict[float, dict[str, np.ndarray]]]:
        """Evaluate all airfoils x Re x alpha in one network call."""
        n_airfoils = len(kulfan_list)
        n_alpha = len(alpha_arr)
        n_re = len(re_list)
        n_cases_per_airfoil = n_alpha * n_re
        total_cases = n_airfoils * n_cases_per_airfoil

        print(
            "NeuralFoil: mega-batch inference "
            f"({n_airfoils} foils x {n_re} Re x {n_alpha} alpha = "
            f"{total_cases} cases)"
        )

        upper = np.asarray(
            [np.asarray(k["upper_weights"], dtype=float) for k in kulfan_list]
        )
        lower = np.asarray(
            [np.asarray(k["lower_weights"], dtype=float) for k in kulfan_list]
        )
        le = np.asarray(
            [
                float(np.asarray(k["leading_edge_weight"]).reshape(-1)[0])
                for k in kulfan_list
            ],
            dtype=float,
        )
        te = np.asarray(
            [
                float(np.asarray(k["TE_thickness"]).reshape(-1)[0])
                for k in kulfan_list
            ],
            dtype=float,
        )

        alpha_single = np.tile(alpha_arr, n_re)
        re_single = np.repeat(np.asarray(re_list, dtype=float), n_alpha)

        kulfan_batch = {
            "upper_weights": np.repeat(
                upper, n_cases_per_airfoil, axis=0
            ).T,
            "lower_weights": np.repeat(
                lower, n_cases_per_airfoil, axis=0
            ).T,
            "leading_edge_weight": np.repeat(
                le, n_cases_per_airfoil
            ),
            "TE_thickness": np.repeat(te, n_cases_per_airfoil),
        }
        alpha_flat = np.tile(alpha_single, n_airfoils)
        re_flat = np.tile(re_single, n_airfoils)

        aero_flat = self._evaluate_from_kulfan(
            kulfan=kulfan_batch,
            alpha=alpha_flat,
            re_value=re_flat,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
        )

        by_airfoil = {}
        for i, name in enumerate(labels):
            start = i * n_cases_per_airfoil
            end = (i + 1) * n_cases_per_airfoil

            grids = {
                key: np.asarray(value)[start:end].reshape(n_re, n_alpha)
                for key, value in aero_flat.items()
            }
            by_re = {}
            for j, re_value in enumerate(re_list):
                by_re[re_value] = {
                    key: grids[key][j].copy()
                    for key in grids
                }
            by_airfoil[name] = by_re

        return by_airfoil

    def analyze(
        self,
        airfoil=None,
        alphas=np.arange(-5, 15, 0.5),
        reynolds=0,
        Mach=0,
        flap=None,
        label=None,
        *,
        kulfan_parameters=None,
        upper_weights=None,
        lower_weights=None,
        leading_edge_weight=None,
        trailing_edge_thickness=None,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        batch_size: int | float = 0,
    ):
        """Primary NeuralFoil analysis implementation.

        For API compatibility, :meth:`get_polar`, :meth:`get_cp`, and
        :meth:`get_dump` are thin aliases that forward to this method.

        Supports single-airfoil and multi-airfoil evaluation. For
        multi-airfoil calls, all ``airfoil x Re x alpha`` operating
        points are flattened into one network inference call.
        """
        if Mach not in (0, 0.0):
            warning("NeuralFoil ignores Mach; argument kept for API compatibility.")
        if flap is not None:
            warning("NeuralFoil ignores flap; argument kept for API compatibility.")

        labels, kulfan_list = self._resolve_batch_inputs(
            airfoil=airfoil,
            label=label,
            kulfan_parameters=kulfan_parameters,
            upper_weights=upper_weights,
            lower_weights=lower_weights,
            leading_edge_weight=leading_edge_weight,
            trailing_edge_thickness=trailing_edge_thickness,
        )
        alpha_arr = _to_1d_array(alphas)
        re_list = as_re_list(reynolds)

        if len(kulfan_list) == 1:
            aero_by_re = self._evaluate_by_re(
                kulfan=kulfan_list[0],
                alpha_arr=alpha_arr,
                re_list=re_list,
                n_crit=n_crit,
                xtr_upper=xtr_upper,
                xtr_lower=xtr_lower,
                batch_size=batch_size,
            )
            return self._build_result_from_aero_by_re(
                label=labels[0],
                Mach=Mach,
                alpha_arr=alpha_arr,
                re_list=re_list,
                aero_by_re=aero_by_re,
            )

        by_airfoil = self._evaluate_multi_airfoil(
            labels=labels,
            kulfan_list=kulfan_list,
            alpha_arr=alpha_arr,
            re_list=re_list,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
        )

        children = {
            name: self._build_result_from_aero_by_re(
                label=name,
                Mach=Mach,
                alpha_arr=alpha_arr,
                re_list=re_list,
                aero_by_re=by_airfoil[name],
            )
            for name in labels
        }

        parent_label = label if isinstance(label, str) else "multi_airfoil"
        return AeroResults.from_airfoils(
            children,
            label=parent_label,
            Mach=Mach,
            source="NeuralFoil_np_numfoil",
        )

    # Aliases for compatibility with XFoil-like API.
    # All forward to the main analyze() method.
    get_polar = analyze
    get_cp = analyze
    get_dump = analyze
