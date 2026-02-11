from __future__ import annotations

import numpy as np
import torch

from .interactive_plot import (
    InteractiveAirfoilPlot,
    OverlaySpec,
    SliderSpec,
)
from .parametric_eval import kulfan_modified_cst_y


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Coefficient count kept small for responsive sliders.
    n_coeffs = 8

    # Start with a mild symmetric-ish airfoil.
    upper0 = np.array([0.20, 0.12, 0.06, 0.02, 0.00, -0.01, -0.01, -0.005], dtype=float)
    lower0 = -0.9 * upper0
    initial = {
        **{f"a_u{i}": float(upper0[i]) for i in range(n_coeffs)},
        **{f"a_l{i}": float(lower0[i]) for i in range(n_coeffs)},
        "w_le": 0.02,
        "t_te": 0.002,
    }

    slider_specs = []
    for i in range(n_coeffs):
        slider_specs.append(SliderSpec(f"a_u{i}", f"upper a{i}", -0.6, 0.6, initial[f"a_u{i}"]))
    for i in range(n_coeffs):
        slider_specs.append(SliderSpec(f"a_l{i}", f"lower a{i}", -0.6, 0.6, initial[f"a_l{i}"]))

    slider_specs += [
        SliderSpec("w_le", "w_LE", -0.2, 0.2, initial["w_le"]),
        SliderSpec("t_te", "t_TE", 0.0, 0.05, initial["t_te"]),
    ]

    overlay_specs = [
        OverlaySpec("camber", "camber", default=False),
        OverlaySpec("tmax", "max thickness", default=False),
        OverlaySpec("cmax", "max camber", default=False),
    ]

    x = torch.linspace(0.0, 1.0, 400, device=device)

    def _coeff_tensors(params: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        upper = torch.tensor([params[f"a_u{i}"] for i in range(n_coeffs)], device=device, dtype=torch.float32)
        lower = torch.tensor([params[f"a_l{i}"] for i in range(n_coeffs)], device=device, dtype=torch.float32)
        w_le = torch.tensor(params["w_le"], device=device, dtype=torch.float32)
        t_te = torch.tensor(params["t_te"], device=device, dtype=torch.float32)
        return upper, lower, w_le, t_te

    def compute_airfoil(params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            upper, lower, w_le, t_te = _coeff_tensors(params)
            y_u = kulfan_modified_cst_y(
                x=x,
                coefficients=upper,
                w_le=w_le,
                t_te=t_te,
                surface="upper",
            )
            y_l = kulfan_modified_cst_y(
                x=x,
                coefficients=lower,
                w_le=w_le,
                t_te=t_te,
                surface="lower",
            )

        x_np = x.detach().cpu().numpy()
        yu = y_u.detach().cpu().numpy()
        yl = y_l.detach().cpu().numpy()

        return np.column_stack([x_np, yu]), np.column_stack([x_np, yl])

    def compute_overlays(params: dict[str, float]) -> dict[str, object]:
        out: dict[str, object] = {}

        xs = torch.linspace(0.0, 1.0, 1200, device=device)
        with torch.no_grad():
            upper, lower, w_le, t_te = _coeff_tensors(params)
            y_u = kulfan_modified_cst_y(
                x=xs,
                coefficients=upper,
                w_le=w_le,
                t_te=t_te,
                surface="upper",
            )
            y_l = kulfan_modified_cst_y(
                x=xs,
                coefficients=lower,
                w_le=w_le,
                t_te=t_te,
                surface="lower",
            )
            t = y_u - y_l
            c = 0.5 * (y_u + y_l)

            _, idx_t = torch.max(t, dim=-1)
            xt = xs[idx_t]

            _, idx_c = torch.max(torch.abs(c), dim=-1)
            xc = xs[idx_c]

        out["tmax"] = {"vline": float(xt.item())}
        out["cmax"] = {"vline": float(xc.item())}
        return out

    ui = InteractiveAirfoilPlot(
        slider_specs=slider_specs,
        overlay_specs=overlay_specs,
        compute_airfoil=compute_airfoil,
        compute_overlays=compute_overlays,
        title=f"Kulfan modified CST (torch) interactive (n={n_coeffs})",
        xlim=(-0.05, 1.05),
        ylim=(-0.25, 0.35),
    )
    ui.show()


if __name__ == "__main__":
    main()
