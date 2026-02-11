from __future__ import annotations

"""Bokeh server app: interactive modified Kulfan/CST airfoil sliders + overlays.

Run (from anywhere):
    python -m numfoil.interactive.run_kulfan_app --port 5007

In VS Code Remote: forward port 5007 and open the forwarded URL.

Notes:
- `numfoil` currently errors if `t_te` is passed as a Python float (it calls
  `torch.all()` on a bool). This experiment works around it by always passing
  torch tensors for `w_le` and `t_te`.
"""

import numpy as np
import base64
import binascii
import ast
import json

from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, Div, Slider, TextAreaInput, FileInput, CustomJS
from bokeh.plotting import figure

try:
    from .bokeh_common import (
        OverlayState,
        cosine_x,
        estimate_le_radius_from_second_derivative,
    )
except ImportError:  # When executed by bokeh as a script (no package context).
    from numfoil.interactive.bokeh_common import (
        OverlayState,
        cosine_x,
        estimate_le_radius_from_second_derivative,
    )

import torch
from numfoil.torch import TorchKulfanAirfoil
from numfoil.util import split_upper_lower


DEVICE = torch.device("cpu")
DTYPE = torch.float32

N_COEFFS = 8  # keep UI small; increase later if needed

X_PLOT = cosine_x(1000, device=DEVICE, dtype=DTYPE)
X_PROP = torch.linspace(0.0, 1.0, 2000, device=DEVICE, dtype=DTYPE)

_BULK_UPDATE = False


def make_initial_params() -> dict:
    upper = np.array([0.20, 0.10, 0.05, 0.02, 0.00, 0.00, 0.00, 0.00], dtype=float)
    lower = -0.90 * upper
    return dict(
        upper=upper,
        lower=lower,
        w_le=0.02,
        t_te=0.002,
    )


params = make_initial_params()


def build_airfoil() -> TorchKulfanAirfoil:
    # Workaround for numfoil float bug: always pass tensors.
    upper = torch.as_tensor(params["upper"], device=DEVICE, dtype=DTYPE)
    lower = torch.as_tensor(params["lower"], device=DEVICE, dtype=DTYPE)
    w_le = torch.as_tensor(float(params["w_le"]), device=DEVICE, dtype=DTYPE)
    t_te = torch.as_tensor(float(params["t_te"]), device=DEVICE, dtype=DTYPE)

    return TorchKulfanAirfoil.from_kulfan_params(
        upper_coeffs=upper,
        lower_coeffs=lower,
        w_le=w_le,
        t_te=t_te,
        n1=0.5,
        n2=1.0,
        device=DEVICE,
    )


def _to_numpy_1d(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.ndim >= 2:
        t = t[0]
    return t.numpy()


source_main = ColumnDataSource(data=dict(x=[], y=[]))
source_camber = ColumnDataSource(data=dict(x=[], y=[]))
source_thickness = ColumnDataSource(data=dict(x=[], y=[]))
source_tmax_seg = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))
source_cmax_seg = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))
source_le_circle = ColumnDataSource(data=dict(xc=[], yc=[], w=[], h=[]))
source_le_center = ColumnDataSource(data=dict(x=[], y=[]))

plot = figure(
    height=520,
    width=900,
    title="Kulfan modified CST airfoil (interactive)",
    x_axis_label="x/c",
    y_axis_label="y/c",
    match_aspect=True,
)

CAMBER_COLOR = "orange"
THICKNESS_COLOR = "green"

main_line = plot.line("x", "y", source=source_main, line_width=2, color="navy")
camber_line = plot.line("x", "y", source=source_camber, line_width=2, line_dash="dashed", color=CAMBER_COLOR)
thickness_line = plot.line("x", "y", source=source_thickness, line_width=2, line_dash="dotted", color=THICKNESS_COLOR)

# Vertical segments for max camber/thickness
tmax_seg = plot.segment("x", "y0", "x", "y1", source=source_tmax_seg, line_width=3, color=THICKNESS_COLOR)
cmax_seg = plot.segment("x", "y0", "x", "y1", source=source_cmax_seg, line_width=3, color=CAMBER_COLOR)

# Leading-edge osculating circle + center
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


def current_overlay(active: list[int]) -> OverlayState:
    a = set(active)
    return OverlayState(
        camber=0 in a,
        thickness=1 in a,
        max_thickness=2 in a,
        max_camber=3 in a,
        le_radius=4 in a,
    )


def update_sources(overlay: OverlayState) -> None:
    foil = build_airfoil()

    # NOTE: TorchKulfanAirfoil.coordinates_at currently assumes batched tensors
    # and can error for single-airfoil evaluation. Use `.points` instead.
    pts = foil.points  # [199, 2] or [B, 199, 2]
    # pts = foil.coordinates_at(X_PLOT)  # [2000, 2] or [B, 2000, 2]
    pts_np = pts.detach().cpu().numpy()
    if pts_np.ndim == 3:
        pts_np = pts_np[0]
    source_main.data = dict(x=pts_np[:, 0], y=pts_np[:, 1])

    camber_line.visible = overlay.camber
    thickness_line.visible = overlay.thickness

    if overlay.camber:
        camber = foil.camber_at(X_PROP)
        source_camber.data = dict(x=_to_numpy_1d(X_PROP), y=_to_numpy_1d(camber))
    if overlay.thickness:
        thickness = foil.thickness_at(X_PROP)
        source_thickness.data = dict(x=_to_numpy_1d(X_PROP), y=_to_numpy_1d(thickness))

    # Always compute scalar properties for display.
    # NOTE: TorchKulfanAirfoil.max_camber currently assumes batched tensors and
    # can error for a single airfoil, so compute robustly from camber_at.
    x_dense = torch.linspace(0.0, 1.0, 6000, device=DEVICE, dtype=DTYPE)

    t = foil.thickness_at(x_dense)
    if t.ndim == 2:
        t0 = t[0]
    else:
        t0 = t
    t_max, idx_t = torch.max(t0, dim=-1)
    x_t = x_dense[idx_t]

    c = foil.camber_at(x_dense)
    if c.ndim == 2:
        c0 = c[0]
    else:
        c0 = c
    idx_c = torch.argmax(torch.abs(c0), dim=-1)
    x_c = x_dense[idx_c]
    c_max = c0[idx_c]

    # Thickness segment: connect lower to upper at x_t
    y_u_t, y_l_t = foil.forward(x_t.reshape(1))
    if y_u_t.ndim:
        y_u_t = y_u_t[0]
        y_l_t = y_l_t[0]
    y_t0 = float(y_l_t.squeeze().item())
    y_t1 = float(y_u_t.squeeze().item())

    # Camber segment: from y=0 to y=c(x_c) at x_c
    c_at = foil.camber_at(x_c.reshape(1))
    if c_at.ndim:
        c_at = c_at[0]
    y_c1 = float(c_at.squeeze().item())

    # LE radius estimate (finite-difference on the evaluated surface)
    y_u_plot, _ = foil.forward(X_PLOT)
    if y_u_plot.ndim == 2:
        y_u_plot = y_u_plot[0]
    r_le = estimate_le_radius_from_second_derivative(X_PLOT, y_u_plot, x_eval=0.01)

    values.text = (
        "<b>Properties</b>"
        f"<br>r_LE ≈ {r_le:.5f}"
        f"<br>t_max ≈ {float(t_max):.5f} at x ≈ {float(x_t):.5f}"
        f"<br>c_max ≈ {abs(float(c_max)):.5f} at x ≈ {float(x_c):.5f}"
    )

    # Overlay visibility + data
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

    # LE osculating circle approximation centered at (r, 0)
    le_circle.visible = overlay.le_radius
    le_center.visible = overlay.le_radius
    if overlay.le_radius and np.isfinite(r_le) and r_le > 0:
        source_le_circle.data = dict(xc=[r_le], yc=[0.0], w=[2.0 * r_le], h=[2.0 * r_le])
        source_le_center.data = dict(x=[r_le], y=[0.0])
    else:
        source_le_circle.data = dict(xc=[], yc=[], w=[], h=[])
        source_le_center.data = dict(x=[], y=[])

    # Keep a small status line for any future debugging
    status.text = ""


# Widgets
coeff_sliders_upper: list[Slider] = []
coeff_sliders_lower: list[Slider] = []

for i in range(N_COEFFS):
    su = Slider(
        title=f"upper a{i}",
        start=-0.5,
        end=0.5,
        value=float(params["upper"][i]),
        step=0.002,
        format="0.00000",
        width=150,
    )
    sl = Slider(
        title=f"lower a{i}",
        start=-0.5,
        end=0.5,
        value=float(params["lower"][i]),
        step=0.002,
        format="0.00000",
        width=150,
    )
    coeff_sliders_upper.append(su)
    coeff_sliders_lower.append(sl)

s_wle = Slider(title="w_le", start=-0.2, end=0.2, value=float(params["w_le"]), step=0.001, format="0.00000", width=300)
s_tte = Slider(title="t_te", start=0.0, end=0.02, value=float(params["t_te"]), step=0.0005, format="0.00000", width=300)

overlay_group = CheckboxGroup(
    labels=["Camber line", "Thickness", "Max thickness", "Max camber", "LE radius"],
    active=[],
)

reset_btn = Button(label="Reset", button_type="default")
invert_btn = Button(label="Invert (swap upper/lower)", button_type="default")

param_text = TextAreaInput(
    title="Paste Kulfan parameter tensor/array",
    value="",
    placeholder=(
        "Accepted formats:\n"
        "- Flat list/array (len=2*n_coeffs+2): [upper... , lower... , w_le, t_te]\n"
        "- Dict: { 'upper': [...], 'lower': [...], 'w_le': 0.02, 't_te': 0.002 }\n"
    ),
    rows=4,
    width=520,
)
apply_params_btn = Button(label="Apply pasted parameters", button_type="primary")
export_params_btn = Button(label="Export current parameters", button_type="default")
copy_params_btn = Button(label="Copy parameters to clipboard", button_type="default")

dat_upload = FileInput(accept=".dat", multiple=False)
fit_dat_btn = Button(label="Fit + display uploaded .dat", button_type="primary")


def on_coeff_change(which: str, idx: int):
    def _cb(attr: str, old: float, new: float) -> None:
        if _BULK_UPDATE:
            return
        params[which][idx] = float(new)
        update_sources(current_overlay(overlay_group.active))

    return _cb


for i, s in enumerate(coeff_sliders_upper):
    s.on_change("value", on_coeff_change("upper", i))
for i, s in enumerate(coeff_sliders_lower):
    s.on_change("value", on_coeff_change("lower", i))


def on_simple_change(key: str):
    def _cb(attr: str, old: float, new: float) -> None:
        if _BULK_UPDATE:
            return
        params[key] = float(new)
        update_sources(current_overlay(overlay_group.active))

    return _cb


s_wle.on_change("value", on_simple_change("w_le"))
s_tte.on_change("value", on_simple_change("t_te"))


def on_overlay_change(attr: str, old: list[int], new: list[int]) -> None:
    update_sources(current_overlay(new))


overlay_group.on_change("active", on_overlay_change)


def _reset() -> None:
    defaults = make_initial_params()
    params["upper"] = defaults["upper"].copy()
    params["lower"] = defaults["lower"].copy()
    params["w_le"] = float(defaults["w_le"])
    params["t_te"] = float(defaults["t_te"])

    global _BULK_UPDATE
    _BULK_UPDATE = True
    try:
        s_wle.value = params["w_le"]
        s_tte.value = params["t_te"]
        for i in range(N_COEFFS):
            coeff_sliders_upper[i].value = float(params["upper"][i])
            coeff_sliders_lower[i].value = float(params["lower"][i])
    finally:
        _BULK_UPDATE = False
    update_sources(current_overlay(overlay_group.active))


def _invert() -> None:
    # Flip vertically: y -> -y.
    # For Kulfan, this corresponds to swapping surfaces AND negating their parameters.
    old_upper = params["upper"].copy()
    old_lower = params["lower"].copy()

    params["upper"] = (-old_lower).copy()
    params["lower"] = (-old_upper).copy()
    params["w_le"] = -float(params["w_le"])  # LE modification is additive on both surfaces
    # t_te is a thickness; keep it non-negative

    global _BULK_UPDATE
    _BULK_UPDATE = True
    try:
        s_wle.value = float(params["w_le"])
        for i in range(N_COEFFS):
            coeff_sliders_upper[i].value = float(params["upper"][i])
            coeff_sliders_lower[i].value = float(params["lower"][i])
    finally:
        _BULK_UPDATE = False
    update_sources(current_overlay(overlay_group.active))


def _apply_kulfan_vector(vec: np.ndarray) -> None:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    expected = 2 * N_COEFFS + 2
    if vec.size != expected:
        raise ValueError(f"Expected {expected} values (2*{N_COEFFS}+2), got {vec.size}.")

    params["upper"] = vec[:N_COEFFS].copy()
    params["lower"] = vec[N_COEFFS : 2 * N_COEFFS].copy()
    params["w_le"] = float(vec[-2])
    params["t_te"] = float(abs(vec[-1]))

    global _BULK_UPDATE
    _BULK_UPDATE = True
    try:
        s_wle.value = float(params["w_le"])
        s_tte.value = float(params["t_te"])
        for i in range(N_COEFFS):
            coeff_sliders_upper[i].value = float(params["upper"][i])
            coeff_sliders_lower[i].value = float(params["lower"][i])
    finally:
        _BULK_UPDATE = False
    update_sources(current_overlay(overlay_group.active))


def _parse_params_text(text: str) -> np.ndarray:
    raw = text.strip()
    if not raw:
        raise ValueError("No parameters provided.")

    # Accept common wrappers like tensor([...]) / torch.tensor([...]) / array([...])
    if "tensor" in raw or "array" in raw:
        lb = raw.find("[")
        rb = raw.rfind("]")
        if 0 <= lb < rb:
            raw = raw[lb : rb + 1]

    try:
        obj = ast.literal_eval(raw)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Could not parse parameters: {e}")

    if isinstance(obj, dict):
        upper = np.asarray(obj.get("upper"), dtype=float)
        lower = np.asarray(obj.get("lower"), dtype=float)
        w_le = float(obj.get("w_le"))
        t_te = float(obj.get("t_te"))
        if upper.size != N_COEFFS or lower.size != N_COEFFS:
            raise ValueError(f"upper/lower must each have {N_COEFFS} coefficients.")
        return np.concatenate([upper.reshape(-1), lower.reshape(-1), [w_le, t_te]])

    arr = np.asarray(obj, dtype=float).reshape(-1)
    if arr.size == 2 * N_COEFFS + 2:
        return arr
    if arr.size == 2 * N_COEFFS:
        return np.concatenate([arr, [params["w_le"], params["t_te"]]])
    raise ValueError(
        f"Expected length {2*N_COEFFS+2} (full) or {2*N_COEFFS} (coeffs-only); got {arr.size}."
    )


def _apply_pasted_params() -> None:
    try:
        vec = _parse_params_text(param_text.value)
        _apply_kulfan_vector(vec)
        status.text = "<span style='color: green'>Applied parameters.</span>"
    except Exception as e:  # noqa: BLE001
        status.text = f"<span style='color: red'>Param parse/apply failed: {e}</span>"


def _export_current_params() -> None:
    upper = [round(float(v), 5) for v in params["upper"]]
    lower = [round(float(v), 5) for v in params["lower"]]
    payload = [
        *upper,
        *lower,
        round(float(params["w_le"]), 5),
        round(float(params["t_te"]), 5),
    ]
    param_text.value = json.dumps(payload)
    status.text = "<span style='color: green'>Exported current parameters.</span>"


def _parse_dat_bytes(contents_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(contents_b64)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 upload: {e}")

    text = raw.decode("utf-8", errors="ignore")
    pts: list[tuple[float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        pts.append((x, y))

    if len(pts) < 10:
        raise ValueError(f"Too few coordinate rows parsed ({len(pts)}).")

    arr = np.asarray(pts, dtype=float)
    x_min = float(arr[:, 0].min())
    x_max = float(arr[:, 0].max())
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError("Invalid chord length.")
    arr[:, 0] = (arr[:, 0] - x_min) / chord
    arr[:, 1] = arr[:, 1] / chord
    return arr


def _fit_uploaded_dat() -> None:
    try:
        if not dat_upload.value:
            raise ValueError("No .dat content uploaded.")
        pts = _parse_dat_bytes(dat_upload.value)
        upper_pts, lower_pts = split_upper_lower(pts)
        fitted = TorchKulfanAirfoil.fit(
            upper_points=upper_pts,
            lower_points=lower_pts,
            n_coefficients=N_COEFFS,
            device=DEVICE,
        )
        vec = fitted.kulfan_params.detach().cpu().numpy().reshape(-1)
        _apply_kulfan_vector(vec)
        status.text = "<span style='color: green'>Fitted .dat and updated sliders.</span>"
    except Exception as e:  # noqa: BLE001
        status.text = f"<span style='color: red'>.dat fit failed: {e}</span>"


reset_btn.on_click(_reset)
invert_btn.on_click(_invert)
apply_params_btn.on_click(_apply_pasted_params)
export_params_btn.on_click(_export_current_params)
copy_params_btn.js_on_click(
        CustomJS(
                args=dict(textbox=param_text, status=status),
                code="""
                const text = textbox.value || "";
                if (!text) {
                    status.text = "<span style='color: red'>Nothing to copy.</span>";
                    return;
                }
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(() => {
                        status.text = "<span style='color: green'>Copied to clipboard.</span>";
                    }).catch((err) => {
                        status.text = `<span style='color: red'>Clipboard copy failed: ${err}</span>`;
                    });
                } else {
                    status.text = "<span style='color: red'>Clipboard API not available.</span>";
                }
                """,
        )
)
fit_dat_btn.on_click(_fit_uploaded_dat)


# Right side: buttons + property values + toggles only
right_panel = column(
    row(reset_btn, invert_btn),
    values,
    Spacer(height=8),
    Div(text="<b>Overlays</b>"),
    overlay_group,
    Spacer(height=8),
    Div(text="<b>Set airfoil</b>"),
    param_text,
    row(apply_params_btn, export_params_btn, copy_params_btn),
    Spacer(height=8),
    Div(text="<b>Load .dat</b>"),
    dat_upload,
    fit_dat_btn,
    status,
    width=520,
)

# Under-plot sliders
upper_row = row(*coeff_sliders_upper)
lower_row = row(*coeff_sliders_lower)

slider_block_width = max(900, 8 * 150)

slider_block = column(
    row(s_wle, s_tte),
    Div(text="<b>Upper surface</b>"),
    upper_row,
    Div(text="<b>Lower surface</b>"),
    lower_row,
    width=slider_block_width,
)

layout = column(
    row(plot, right_panel),
    row(slider_block, Spacer(width=520)),
)

curdoc().add_root(layout)
curdoc().title = "Kulfan interactive"

update_sources(current_overlay(overlay_group.active))
