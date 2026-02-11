from __future__ import annotations

"""Bokeh server app: interactive PARSEC airfoil sliders + overlays.

Run (from anywhere):
    python -m numfoil.interactive.run_parsec_app --port 5006

In VS Code Remote: forward port 5006 and open the forwarded URL.
"""

import numpy as np
import ast
import base64

from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, Div, FileInput, Slider, TextAreaInput
from bokeh.plotting import figure

try:
    from .bokeh_common import (
        OverlayState,
        cosine_x,
        to_float_tensor,
    )
except ImportError:  # When executed by bokeh as a script (no package context).
    from numfoil.interactive.bokeh_common import (
        OverlayState,
        cosine_x,
        to_float_tensor,
    )

import torch
from numfoil.torch import TorchKulfanAirfoil, TorchPARSECAirfoil
from numfoil.util import split_upper_lower


DEVICE = torch.device("cpu")
DTYPE = torch.float32

# Chordwise grid for plotting and for property estimation.
X_PLOT = cosine_x(220, device=DEVICE, dtype=DTYPE)
X_PROP = torch.linspace(0.0, 1.0, 600, device=DEVICE, dtype=DTYPE)

_BULK_UPDATE = False
_UPLOADED_DAT_POINTS: np.ndarray | None = None


KULFAN_FIT_N_COEFFS = 8


def _scalar(t: torch.Tensor) -> float:
    return float(t.detach().cpu().reshape(-1)[0].item())


def _parsec_params_from_fitted_kulfan(foil: TorchKulfanAirfoil) -> dict:
    """Derive PARSEC parameters from a fitted Kulfan airfoil.

    This avoids `TorchPARSECAirfoil.fit(...)` (brittle for some inputs) while
    still producing a PARSEC parameter set driven by the uploaded geometry.
    """
    with torch.no_grad():
        # Crest locations
        x_z_u, y_z_u = foil.upper_crest
        x_z_l, y_z_l = foil.lower_crest

        # PARSEC uses k_z := y''(x_z)
        xzu = x_z_u.reshape(1).to(device=DEVICE, dtype=DTYPE)
        xzl = x_z_l.reshape(1).to(device=DEVICE, dtype=DTYPE)
        k_z_u = foil.upper_surface.second_derivative_at(xzu, mode="analytic")
        k_z_l = foil.lower_surface.second_derivative_at(xzl, mode="analytic")

        # Trailing edge ordinate + thickness at x=1
        x1 = torch.tensor([1.0], device=DEVICE, dtype=DTYPE)
        y_u1, y_l1 = foil.forward(x1)
        y_te = 0.5 * (_scalar(y_u1) + _scalar(y_l1))
        t_te = abs(_scalar(y_u1) - _scalar(y_l1))

        # Trailing edge slopes -> theta/gamma (radians)
        x_te = torch.tensor([0.999], device=DEVICE, dtype=DTYPE)
        dy_u = foil.upper_surface.first_derivative_at(x_te, mode="analytic")
        dy_l = foil.lower_surface.first_derivative_at(x_te, mode="analytic")
        alpha_u = float(np.arctan(_scalar(dy_u)))
        alpha_l = float(np.arctan(_scalar(dy_l)))
        theta_te = 0.5 * (alpha_u + alpha_l)
        gamma_te = abs(alpha_u - alpha_l)

        # Leading edge radius (average curvature radius near LE)
        x_le = torch.tensor([0.01], device=DEVICE, dtype=DTYPE)
        kappa_u = foil.upper_surface.curvature_at(x_le, mode="analytic")
        kappa_l = foil.lower_surface.curvature_at(x_le, mode="analytic")
        kappa_avg = 0.5 * (torch.abs(kappa_u) + torch.abs(kappa_l))
        r_le = 1.0 / (_scalar(kappa_avg) + 1e-12)

    return dict(
        r_le=float(r_le),
        x_z_u=_scalar(x_z_u),
        y_z_u=_scalar(y_z_u),
        k_z_u=_scalar(k_z_u),
        x_z_l=_scalar(x_z_l),
        y_z_l=_scalar(y_z_l),
        k_z_l=_scalar(k_z_l),
        y_te=float(y_te),
        t_te=float(t_te),
        theta_te=float(theta_te),
        gamma_te=float(gamma_te),
    )


def make_initial_params() -> dict:
    # Reasonable-ish defaults; tweak as you like.
    return dict(
        r_le=0.015,
        x_z_u=0.35,
        y_z_u=0.10,
        k_z_u=-0.60,
        x_z_l=0.40,
        y_z_l=-0.06,
        k_z_l=0.50,
        y_te=0.0,
        t_te=0.002,
        theta_te=0.0,
        gamma_te=0.10,
    )


def build_airfoil(params: dict) -> TorchPARSECAirfoil:
    # Constructing the torch module per update is OK here (6x6 solve per surface).
    # NOTE: numfoil's TorchPARSECAirfoil currently maps (theta_te, gamma_te)
    # to per-surface trailing-edge slopes with the upper/lower assignment swapped
    # relative to the common PARSEC convention.
    # App-side workaround: negate gamma so increasing the slider opens the wedge
    # in the expected direction (upper surface goes up toward the LE, lower goes down).
    return TorchPARSECAirfoil.from_parsec_params(
        r_le=to_float_tensor(params["r_le"], device=DEVICE, dtype=DTYPE),
        x_z_u=to_float_tensor(params["x_z_u"], device=DEVICE, dtype=DTYPE),
        y_z_u=to_float_tensor(params["y_z_u"], device=DEVICE, dtype=DTYPE),
        k_z_u=to_float_tensor(params["k_z_u"], device=DEVICE, dtype=DTYPE),
        x_z_l=to_float_tensor(params["x_z_l"], device=DEVICE, dtype=DTYPE),
        y_z_l=to_float_tensor(params["y_z_l"], device=DEVICE, dtype=DTYPE),
        k_z_l=to_float_tensor(params["k_z_l"], device=DEVICE, dtype=DTYPE),
        y_te=to_float_tensor(params["y_te"], device=DEVICE, dtype=DTYPE),
        t_te=to_float_tensor(params["t_te"], device=DEVICE, dtype=DTYPE),
        theta_te=to_float_tensor(params["theta_te"], device=DEVICE, dtype=DTYPE),
        gamma_te=to_float_tensor(-params["gamma_te"], device=DEVICE, dtype=DTYPE),
        angles_in_degrees=False,
        device=DEVICE,
    )


def _parse_dat_points_from_b64(contents_b64: str) -> np.ndarray:
    """Parse a Selig-style .dat file upload (base64) into Nx2 float array."""
    try:
        raw = base64.b64decode(contents_b64)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid base64 upload: {e}")

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    pts: list[tuple[float, float]] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Skip header-ish lines (airfoil name, etc.)
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        pts.append((x, y))

    if len(pts) < 20:
        raise ValueError(f"Not enough numeric points found ({len(pts)}).")

    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Parsed points must have shape [N,2].")
    return arr


def _normalize_chord(points: np.ndarray) -> np.ndarray:
    """Normalize points so x spans [0,1] and scale y by chord length."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 10:
        raise ValueError("Too few finite points to normalize.")

    x = pts[:, 0]
    y = pts[:, 1]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    chord = x_max - x_min
    if not np.isfinite(chord) or chord <= 1e-12:
        raise ValueError("Invalid chord length during normalization.")

    x_n = (x - x_min) / chord
    y_n = y / chord
    out = np.column_stack([x_n, y_n])
    return out


def _resample_polyline(points: np.ndarray, *, n: int = 400) -> np.ndarray:
    """Resample a polyline by arclength to n points.

    This makes downstream spline-based routines more robust to sparse inputs.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Polyline points must have shape [N,2].")
    if pts.shape[0] < 4:
        raise ValueError(f"Need at least 4 points to resample; got {pts.shape[0]}.")

    # Drop consecutive duplicates / near-duplicates.
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], deltas > 1e-12])
    pts = pts[keep]
    if pts.shape[0] < 4:
        raise ValueError("Polyline is degenerate after de-duplication.")

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError("Polyline has near-zero total length.")

    sq = np.linspace(0.0, total, int(n))
    xq = np.interp(sq, s, pts[:, 0])
    yq = np.interp(sq, s, pts[:, 1])
    return np.column_stack([xq, yq]).astype(float)


def _set_slider_value(slider: Slider, value: float, *, pad: float = 1e-6) -> None:
    """Set slider value, expanding bounds if needed."""
    if value < slider.start:
        slider.start = float(value - pad)
    if value > slider.end:
        slider.end = float(value + pad)
    slider.value = float(value)

def _to_numpy_1d(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.ndim >= 2:
        t = t[0]
    return t.numpy()


params = make_initial_params()

source_upper = ColumnDataSource(data=dict(x=[], y=[]))
source_lower = ColumnDataSource(data=dict(x=[], y=[]))
source_camber = ColumnDataSource(data=dict(x=[], y=[]))
source_thickness = ColumnDataSource(data=dict(x=[], y=[]))
source_tmax_seg = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))
source_cmax_seg = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))
source_le_circle = ColumnDataSource(data=dict(xc=[], yc=[], w=[], h=[]))
source_le_center = ColumnDataSource(data=dict(x=[], y=[]))


plot = figure(
    height=520,
    width=900,
    title="PARSEC airfoil (interactive)",
    x_axis_label="x/c",
    y_axis_label="y/c",
    match_aspect=True,
)

CAMBER_COLOR = "orange"
THICKNESS_COLOR = "green"

upper_line = plot.line("x", "y", source=source_upper, line_width=2, color="navy")
lower_line = plot.line("x", "y", source=source_lower, line_width=2, color="navy")
camber_line = plot.line("x", "y", source=source_camber, line_width=2, line_dash="dashed", color=CAMBER_COLOR)
thickness_line = plot.line("x", "y", source=source_thickness, line_width=2, line_dash="dotted", color=THICKNESS_COLOR)

tmax_seg = plot.segment("x", "y0", "x", "y1", source=source_tmax_seg, line_width=3, color=THICKNESS_COLOR)
cmax_seg = plot.segment("x", "y0", "x", "y1", source=source_cmax_seg, line_width=3, color=CAMBER_COLOR)

le_circle = plot.ellipse("xc", "yc", width="w", height="h", source=source_le_circle, fill_alpha=0.0, line_width=2, line_color="magenta")
le_center = plot.scatter("x", "y", source=source_le_center, marker="circle", size=8, color="magenta")

camber_line.visible = False
thickness_line.visible = False
tmax_seg.visible = False
cmax_seg.visible = False
le_circle.visible = False
le_center.visible = False

status = Div(text="")
values = Div(text="")


def update_sources(*, overlay: OverlayState) -> None:
    foil = build_airfoil(params)
    with torch.no_grad():
        y_u, y_l = foil.forward(X_PLOT)
        x_np = _to_numpy_1d(X_PLOT)
        source_upper.data = dict(x=x_np, y=_to_numpy_1d(y_u))
        source_lower.data = dict(x=x_np, y=_to_numpy_1d(y_l))

    # toggles
    camber_line.visible = overlay.camber
    thickness_line.visible = overlay.thickness

    if overlay.camber:
        camber = foil.camber_at(X_PROP)
        source_camber.data = dict(x=_to_numpy_1d(X_PROP), y=_to_numpy_1d(camber))

    if overlay.thickness:
        thickness = foil.thickness_at(X_PROP)
        source_thickness.data = dict(x=_to_numpy_1d(X_PROP), y=_to_numpy_1d(thickness))

    # Always compute scalar properties for display
    t_max, x_t = foil.max_thickness(n_points=2048)
    c_max, x_c = foil.max_camber(n_points=2048)
    if getattr(t_max, "ndim", 0):
        t_max = t_max[0]
        x_t = x_t[0]
    if getattr(c_max, "ndim", 0):
        c_max = c_max[0]
        x_c = x_c[0]

    y_u_t, y_l_t = foil.forward(x_t.reshape(1))
    if y_u_t.ndim == 2:
        y_u_t = y_u_t[0]
        y_l_t = y_l_t[0]
    y_t0 = float(y_l_t.squeeze().item())
    y_t1 = float(y_u_t.squeeze().item())

    c_at = foil.camber_at(x_c.reshape(1))
    if c_at.ndim == 2:
        c_at = c_at[0]
    y_c1 = float(c_at.squeeze().item())

    r_le = float(params["r_le"])
    # min thickness for feasibility signal (negative => surfaces cross)
    with torch.no_grad():
        yu_p, yl_p = foil.forward(X_PROP)
        t_min = float((yu_p - yl_p).min())
        t_min_color = "red" if t_min < 0 else "black"
    values.text = (
        "<b>Properties</b>"
        f"<br>r_LE = {r_le:.5f}"
        f"<br>t_max ≈ {float(t_max):.5f} at x ≈ {float(x_t):.5f}"
        f"<br>c_max ≈ {abs(float(c_max)):.5f} at x ≈ {float(x_c):.5f}"
        f"<br><span style='color:{t_min_color}'>t_min ≈ {t_min:.5f}</span>"
    )

    tmax_seg.visible = overlay.max_thickness
    cmax_seg.visible = overlay.max_camber
    if overlay.max_thickness:
        source_tmax_seg.data = dict(x=[float(x_t)], y0=[y_t0], y1=[y_t1])
    else:
        source_tmax_seg.data = dict(x=[], y0=[], y1=[])

    if overlay.max_camber:
        source_cmax_seg.data = dict(x=[float(x_c)], y0=[0.0], y1=[y_c1])
    else:
        source_cmax_seg.data = dict(x=[], y0=[], y1=[])

    le_circle.visible = overlay.le_radius
    le_center.visible = overlay.le_radius
    if overlay.le_radius and r_le > 0:
        source_le_circle.data = dict(xc=[r_le], yc=[0.0], w=[2.0 * r_le], h=[2.0 * r_le])
        source_le_center.data = dict(x=[r_le], y=[0.0])
    else:
        source_le_circle.data = dict(xc=[], yc=[], w=[], h=[])
        source_le_center.data = dict(x=[], y=[])

    status.text = ""


# Widgets
SLIDER_W = 220

s_rle = Slider(title="r_le", start=0.001, end=0.08, value=params["r_le"], step=0.0005, format="0.00000", width=SLIDER_W)

s_xzu = Slider(title="x_z_u", start=0.05, end=0.95, value=params["x_z_u"], step=0.005, format="0.00000", width=SLIDER_W)
s_yzu = Slider(title="y_z_u", start=0.0, end=0.25, value=params["y_z_u"], step=0.002, format="0.00000", width=SLIDER_W)
s_kzu = Slider(title="k_z_u", start=-3.0, end=3.0, value=params["k_z_u"], step=0.02, format="0.00000", width=SLIDER_W)

s_xzl = Slider(title="x_z_l", start=0.05, end=0.95, value=params["x_z_l"], step=0.005, format="0.00000", width=SLIDER_W)
s_yzl = Slider(title="y_z_l", start=-0.25, end=0.0, value=params["y_z_l"], step=0.002, format="0.00000", width=SLIDER_W)
s_kzl = Slider(title="k_z_l", start=-3.0, end=3.0, value=params["k_z_l"], step=0.02, format="0.00000", width=SLIDER_W)

s_yte = Slider(title="y_te", start=-0.05, end=0.05, value=params["y_te"], step=0.001, format="0.00000", width=SLIDER_W)
s_tte = Slider(title="t_te", start=0.0, end=0.02, value=params["t_te"], step=0.0005, format="0.00000", width=SLIDER_W)

s_theta = Slider(title="theta_te (rad)", start=-0.8, end=0.8, value=params["theta_te"], step=0.01, format="0.00000", width=SLIDER_W)
s_gamma = Slider(title="gamma_te (rad)", start=0.0, end=1.0, value=params["gamma_te"], step=0.01, format="0.00000", width=SLIDER_W)


overlay_group = CheckboxGroup(
    labels=["Camber line", "Thickness", "Max thickness", "Max camber", "LE radius"],
    active=[],
)

reset_btn = Button(label="Reset", button_type="default")
invert_btn = Button(label="Invert (swap upper/lower)", button_type="default")

dat_upload = FileInput(accept=".dat", multiple=False)
fit_btn = Button(label="Fit uploaded .dat", button_type="primary")

param_text = TextAreaInput(
    title="Paste PARSEC parameter tensor/array",
    value="",
    placeholder=(
        "Accepted formats:\n"
        "- Preferred 10-vector: [r_le, x_z_u, y_z_u, k_z_u, x_z_l, y_z_l, k_z_l, t_te, theta_te, gamma_te]\n"
        "- Legacy 11-vector: [r_le, x_z_u, y_z_u, k_z_u, x_z_l, y_z_l, k_z_l, y_te, t_te, theta_te, gamma_te]\n"
        "- Dict: { 'r_le': ..., 'x_z_u': ..., ... }\n"
    ),
    rows=4,
    width=520,
)
apply_params_btn = Button(label="Apply pasted parameters", button_type="primary")


def current_overlay() -> OverlayState:
    active = set(overlay_group.active)
    return OverlayState(
        camber=0 in active,
        thickness=1 in active,
        max_thickness=2 in active,
        max_camber=3 in active,
        le_radius=4 in active,
    )


def bind_slider(slider: Slider, key: str) -> None:
    def _on_change(attr: str, old: float, new: float) -> None:
        if _BULK_UPDATE:
            return
        params[key] = float(new)
        update_sources(overlay=current_overlay())

    slider.on_change("value", _on_change)


for s, key in [
    (s_rle, "r_le"),
    (s_xzu, "x_z_u"), (s_yzu, "y_z_u"), (s_kzu, "k_z_u"),
    (s_xzl, "x_z_l"), (s_yzl, "y_z_l"), (s_kzl, "k_z_l"),
    (s_yte, "y_te"), (s_tte, "t_te"),
    (s_theta, "theta_te"), (s_gamma, "gamma_te"),
]:
    bind_slider(s, key)


def on_overlay_change(attr: str, old: list[int], new: list[int]) -> None:
    update_sources(overlay=current_overlay())


overlay_group.on_change("active", on_overlay_change)


def _reset() -> None:
    defaults = make_initial_params()
    for k, v in defaults.items():
        params[k] = float(v)
    global _BULK_UPDATE
    _BULK_UPDATE = True
    try:
        s_rle.value = params["r_le"]
        s_xzu.value = params["x_z_u"]
        s_yzu.value = params["y_z_u"]
        s_kzu.value = params["k_z_u"]
        s_xzl.value = params["x_z_l"]
        s_yzl.value = params["y_z_l"]
        s_kzl.value = params["k_z_l"]
        s_yte.value = params["y_te"]
        s_tte.value = params["t_te"]
        s_theta.value = params["theta_te"]
        s_gamma.value = params["gamma_te"]
    finally:
        _BULK_UPDATE = False
    update_sources(overlay=current_overlay())


def _invert() -> None:
    # Flip vertically: y -> -y.
    # For PARSEC, this is swapping surfaces and negating ordinate/curvature and camber-line angle.
    old_xzu, old_yzu, old_kzu = params["x_z_u"], params["y_z_u"], params["k_z_u"]
    old_xzl, old_yzl, old_kzl = params["x_z_l"], params["y_z_l"], params["k_z_l"]

    params["x_z_u"], params["y_z_u"], params["k_z_u"] = old_xzl, -old_yzl, -old_kzl
    params["x_z_l"], params["y_z_l"], params["k_z_l"] = old_xzu, -old_yzu, -old_kzu
    params["y_te"] = -params["y_te"]
    params["theta_te"] = -params["theta_te"]
    # r_le, t_te, gamma_te remain unchanged (non-negative radius/thickness; wedge angle invariant)

    global _BULK_UPDATE
    _BULK_UPDATE = True
    try:
        s_xzu.value = params["x_z_u"]
        s_yzu.value = params["y_z_u"]
        s_kzu.value = params["k_z_u"]
        s_xzl.value = params["x_z_l"]
        s_yzl.value = params["y_z_l"]
        s_kzl.value = params["k_z_l"]
        s_yte.value = params["y_te"]
        s_theta.value = params["theta_te"]
    finally:
        _BULK_UPDATE = False
    update_sources(overlay=current_overlay())


def _apply_pasted_params() -> None:
    raw = param_text.value.strip()
    if not raw:
        status.text = "<span style='color: red'>No parameters provided.</span>"
        return

    # Accept wrappers like tensor([...])
    if "tensor" in raw or "array" in raw:
        lb = raw.find("[")
        rb = raw.rfind("]")
        if 0 <= lb < rb:
            raw = raw[lb : rb + 1]

    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            for k in params.keys():
                if k in obj:
                    params[k] = float(obj[k])
        else:
            vec = np.asarray(obj, dtype=float).reshape(-1)
            if vec.size == 10:
                (
                    r_le,
                    x_z_u, y_z_u, k_z_u,
                    x_z_l, y_z_l, k_z_l,
                    t_te, theta_te, gamma_te,
                ) = vec.tolist()
                params.update(
                    r_le=r_le,
                    x_z_u=x_z_u, y_z_u=y_z_u, k_z_u=k_z_u,
                    x_z_l=x_z_l, y_z_l=y_z_l, k_z_l=k_z_l,
                    t_te=t_te, theta_te=theta_te, gamma_te=gamma_te,
                )
            elif vec.size == 11:
                (
                    r_le,
                    x_z_u, y_z_u, k_z_u,
                    x_z_l, y_z_l, k_z_l,
                    y_te, t_te, theta_te, gamma_te,
                ) = vec.tolist()
                params.update(
                    r_le=r_le,
                    x_z_u=x_z_u, y_z_u=y_z_u, k_z_u=k_z_u,
                    x_z_l=x_z_l, y_z_l=y_z_l, k_z_l=k_z_l,
                    y_te=y_te, t_te=t_te, theta_te=theta_te, gamma_te=gamma_te,
                )
            else:
                raise ValueError("Expected a 10- or 11-length PARSEC vector.")

        global _BULK_UPDATE
        _BULK_UPDATE = True
        try:
            s_rle.value = params["r_le"]
            s_xzu.value = params["x_z_u"]
            s_yzu.value = params["y_z_u"]
            s_kzu.value = params["k_z_u"]
            s_xzl.value = params["x_z_l"]
            s_yzl.value = params["y_z_l"]
            s_kzl.value = params["k_z_l"]
            s_yte.value = params["y_te"]
            s_tte.value = params["t_te"]
            s_theta.value = params["theta_te"]
            s_gamma.value = params["gamma_te"]
        finally:
            _BULK_UPDATE = False

        update_sources(overlay=current_overlay())
        status.text = "<span style='color: green'>Applied parameters.</span>"
    except Exception as e:  # noqa: BLE001
        status.text = f"<span style='color: red'>Param parse/apply failed: {e}</span>"


def _on_dat_upload(attr: str, old: str, new: str) -> None:
    global _UPLOADED_DAT_POINTS
    if not new:
        _UPLOADED_DAT_POINTS = None
        status.text = ""
        return
    try:
        pts = _parse_dat_points_from_b64(new)
        pts = _normalize_chord(pts)
        _UPLOADED_DAT_POINTS = pts
        status.text = f"<span style='color: green'>Loaded .dat with {pts.shape[0]} points (normalized).</span>"
    except Exception as e:  # noqa: BLE001
        _UPLOADED_DAT_POINTS = None
        status.text = f"<span style='color: red'>Failed to parse .dat: {e}</span>"


def _fit_uploaded() -> None:
    global _UPLOADED_DAT_POINTS
    if _UPLOADED_DAT_POINTS is None:
        status.text = "<span style='color: red'>No .dat file loaded.</span>"
        return

    try:
        pts = np.asarray(_UPLOADED_DAT_POINTS, dtype=float)
        upper_pts, lower_pts = split_upper_lower(pts)
        fitted_kulfan = TorchKulfanAirfoil.fit(
            upper_points=upper_pts,
            lower_points=lower_pts,
            n_coefficients=KULFAN_FIT_N_COEFFS,
            device=DEVICE,
        )

        params.update(_parsec_params_from_fitted_kulfan(fitted_kulfan))

        global _BULK_UPDATE
        _BULK_UPDATE = True
        try:
            _set_slider_value(s_rle, params["r_le"])
            _set_slider_value(s_xzu, params["x_z_u"])
            _set_slider_value(s_yzu, params["y_z_u"])
            _set_slider_value(s_kzu, params["k_z_u"])
            _set_slider_value(s_xzl, params["x_z_l"])
            _set_slider_value(s_yzl, params["y_z_l"])
            _set_slider_value(s_kzl, params["k_z_l"])
            _set_slider_value(s_yte, params["y_te"])
            _set_slider_value(s_tte, params["t_te"])
            _set_slider_value(s_theta, params["theta_te"])
            _set_slider_value(s_gamma, params["gamma_te"])
        finally:
            _BULK_UPDATE = False

        update_sources(overlay=current_overlay())
        status.text = "<span style='color: green'>Fitted .dat via Kulfan → PARSEC parameters.</span>"
    except Exception as e:  # noqa: BLE001
        status.text = f"<span style='color: red'>Fit failed: {e}</span>"


reset_btn.on_click(_reset)
invert_btn.on_click(_invert)
apply_params_btn.on_click(_apply_pasted_params)
dat_upload.on_change("value", _on_dat_upload)
fit_btn.on_click(_fit_uploaded)

# Right side: buttons + property values + toggles + paste
right_panel = column(
    row(reset_btn, invert_btn),
    values,
    Spacer(height=8),
    Div(text="<b>Overlays</b>"),
    overlay_group,
    Spacer(height=8),
    Div(text="<b>Set airfoil</b>"),
    param_text,
    apply_params_btn,
    Div(text="<b>Load .dat + fit</b>"),
    dat_upload,
    fit_btn,
    status,
    width=520,
)

# Under-plot sliders
slider_block = column(
    Div(text="<b>PARSEC parameters</b>"),
    row(s_rle, s_tte, s_yte, Spacer(width=SLIDER_W)),
    Div(text="<b>Upper surface</b>"),
    row(s_xzu, s_yzu, s_kzu, Spacer(width=SLIDER_W)),
    Div(text="<b>Lower surface</b>"),
    row(s_xzl, s_yzl, s_kzl, Spacer(width=SLIDER_W)),
    row(s_theta, s_gamma, Spacer(width=SLIDER_W), Spacer(width=SLIDER_W)),
    width=900,
)

layout = column(
    row(plot, right_panel),
    row(slider_block, Spacer(width=520)),
)

curdoc().add_root(layout)
curdoc().title = "PARSEC interactive"

# initial draw
update_sources(overlay=current_overlay())
