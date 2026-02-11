from __future__ import annotations

import numpy as np
import torch

from .interactive_plot import (
    InteractiveAirfoilPlot,
    OverlaySpec,
    SliderSpec,
)
from .parametric_eval import parsec_airfoil_y


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A reasonable-ish starting PARSEC parameter set (10-param layout; y_te assumed 0):
    # [r_le, x_z_u, y_z_u, k_z_u, x_z_l, y_z_l, k_z_l, t_te, theta_te, gamma_te]
    initial = dict(
        r_le=0.015,
        x_z_u=0.35,
        y_z_u=0.10,
        k_z_u=-0.6,
        x_z_l=0.40,
        y_z_l=-0.06,
        k_z_l=0.5,
        t_te=0.002,
        theta_te=0.0,
        gamma_te=8.0 * np.pi / 180.0,
    )

    slider_specs = [
        SliderSpec("r_le", "r_LE", 0.001, 0.08, initial["r_le"], step=None),
        SliderSpec("x_z_u", "x_zu", 0.05, 0.95, initial["x_z_u"], step=None),
        SliderSpec("y_z_u", "y_zu", 0.00, 0.25, initial["y_z_u"], step=None),
        SliderSpec("k_z_u", "k_zu", -5.0, 1.0, initial["k_z_u"], step=None),
        SliderSpec("x_z_l", "x_zl", 0.05, 0.95, initial["x_z_l"], step=None),
        SliderSpec("y_z_l", "y_zl", -0.25, 0.00, initial["y_z_l"], step=None),
        SliderSpec("k_z_l", "k_zl", -1.0, 5.0, initial["k_z_l"], step=None),
        SliderSpec("t_te", "t_TE", 0.0, 0.03, initial["t_te"], step=None),
        SliderSpec("theta_te", "theta_TE (rad)", -0.7, 0.7, initial["theta_te"], step=None),
        SliderSpec("gamma_te", "gamma_TE (rad)", 0.0, 1.2, initial["gamma_te"], step=None),
    ]

    overlay_specs = [
        OverlaySpec("camber", "camber", default=False),
        OverlaySpec("le_radius", "LE radius", default=False),
        OverlaySpec("tmax", "max thickness", default=False),
        OverlaySpec("cmax", "max camber", default=False),
    ]

    x = torch.linspace(0.0, 1.0, 400, device=device)

    def compute_airfoil(params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            y_u, y_l = parsec_airfoil_y(
                x=x,
                r_le=torch.tensor(params["r_le"], device=device),
                x_z_u=torch.tensor(params["x_z_u"], device=device),
                y_z_u=torch.tensor(params["y_z_u"], device=device),
                k_z_u=torch.tensor(params["k_z_u"], device=device),
                x_z_l=torch.tensor(params["x_z_l"], device=device),
                y_z_l=torch.tensor(params["y_z_l"], device=device),
                k_z_l=torch.tensor(params["k_z_l"], device=device),
                y_te=torch.tensor(0.0, device=device),
                t_te=torch.tensor(params["t_te"], device=device),
                theta_te=torch.tensor(params["theta_te"], device=device),
                gamma_te=torch.tensor(params["gamma_te"], device=device),
            )

        x_np = x.detach().cpu().numpy()
        yu = y_u.detach().cpu().numpy()
        yl = y_l.detach().cpu().numpy()

        xy_u = np.column_stack([x_np, yu])
        xy_l = np.column_stack([x_np, yl])
        return xy_u, xy_l

    def compute_overlays(params: dict[str, float]) -> dict[str, object]:
        out: dict[str, object] = {}

        # Keep overlay computations cheap: use coarse sampling and argmax.
        xs = torch.linspace(0.0, 1.0, 1200, device=device)
        with torch.no_grad():
            y_u, y_l = parsec_airfoil_y(
                x=xs,
                r_le=torch.tensor(params["r_le"], device=device),
                x_z_u=torch.tensor(params["x_z_u"], device=device),
                y_z_u=torch.tensor(params["y_z_u"], device=device),
                k_z_u=torch.tensor(params["k_z_u"], device=device),
                x_z_l=torch.tensor(params["x_z_l"], device=device),
                y_z_l=torch.tensor(params["y_z_l"], device=device),
                k_z_l=torch.tensor(params["k_z_l"], device=device),
                y_te=torch.tensor(0.0, device=device),
                t_te=torch.tensor(params["t_te"], device=device),
                theta_te=torch.tensor(params["theta_te"], device=device),
                gamma_te=torch.tensor(params["gamma_te"], device=device),
            )
            t = y_u - y_l
            c = 0.5 * (y_u + y_l)

            tmax, idx_t = torch.max(t, dim=-1)
            xt = xs[idx_t]

            cabs = torch.abs(c)
            _, idx_c = torch.max(cabs, dim=-1)
            xc = xs[idx_c]
            cmax = c[idx_c]

        out["tmax"] = {"vline": float(xt.item())}
        out["cmax"] = {"vline": float(xc.item())}
        out["le_radius"] = {"text": (0.02, 0.95 * 0.35, f"r_LE = {params['r_le']:.4f}")}

        return out

    ui = InteractiveAirfoilPlot(
        slider_specs=slider_specs,
        overlay_specs=overlay_specs,
        compute_airfoil=compute_airfoil,
        compute_overlays=compute_overlays,
        title="PARSEC (torch) interactive",
        xlim=(-0.05, 1.05),
        ylim=(-0.25, 0.35),
    )
    ui.show()


if __name__ == "__main__":
    main()
