import numpy as np
import scipy.optimize as opt

from numfoil.geometry.spline import BSpline2D
from numfoil.geometry.geom2d import normalize_2d, rotate_2d_90ccw
from numfoil.util import cosine_spacing


def solve_normal_offset_decomposition(
    upper_curve,
    lower_curve,
    *,
    n_samples: int = 300,
    max_iter: int = 80,
    tol: float = 1e-10,
    u_grid: int = 4000,
):
    """
    Recover camber/thickness such that:
        U(s) = C(s) + t(s) * n_C(s)
        L(s) = C(s) - t(s) * n_C(s)

    where n_C is the camber normal. Raises if no exact solution is found
    within tolerance.

    Returns:
        camber_curve: BSpline2D through camber points.
        thickness_curve: BSpline2D over [x, thickness] with thickness=2*t.
        diagnostics: dict with reconstruction errors.
    """
    # Common parameter used only as a marching index.
    s = cosine_spacing(0.0, 1.0, num=n_samples)

    # Start with equal-parameter pairing.
    u_up = s.copy()
    u_lo = s.copy()

    # Keep endpoints pinned.
    u_up[0], u_up[-1] = 0.0, 1.0
    u_lo[0], u_lo[-1] = 0.0, 1.0

    # Dense search grid for robust bracketing.
    ug = np.linspace(0.0, 1.0, u_grid)

    def _intersect_normal(curve, c_pt, n_vec, u_prev):
        """
        Find curve parameter u where curve point lies on line:
            c_pt + alpha * n_vec
        by solving dot(curve(u)-c_pt, t_vec)=0
        with t_vec perpendicular to n_vec.
        """
        t_vec = rotate_2d_90ccw(n_vec.reshape(1, 2))[0]

        vals = curve.evaluate_at(ug) - c_pt
        f = vals @ t_vec

        # Find candidate sign changes; choose one nearest previous u.
        idx = np.where(np.signbit(f[:-1]) != np.signbit(f[1:]))[0]
        if idx.size == 0:
            raise RuntimeError("No normal-line intersection found.")

        mids = 0.5 * (ug[idx] + ug[idx + 1])
        k = idx[np.argmin(np.abs(mids - u_prev))]
        a, b = ug[k], ug[k + 1]

        root = opt.brentq(
            lambda u: float((curve.evaluate_at(u)[0] - c_pt) @ t_vec),
            a,
            b,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=200,
        )
        return root

    # Iterative re-pairing + camber update.
    prev_err = np.inf
    for _ in range(max_iter):
        U = upper_curve.evaluate_at(u_up)
        L = lower_curve.evaluate_at(u_lo)

        C = 0.5 * (U + L)
        d = U - L
        half_t = 0.5 * np.linalg.norm(d, axis=1)

        # Build temporary camber curve and normals from its tangent.
        camber_curve = BSpline2D(C)
        t_c = normalize_2d(camber_curve.first_deriv_at(s))
        n_c = rotate_2d_90ccw(t_c)

        # Update correspondence by intersecting normals.
        new_u_up = u_up.copy()
        new_u_lo = u_lo.copy()

        for i in range(1, n_samples - 1):
            new_u_up[i] = _intersect_normal(upper_curve, C[i], n_c[i], u_up[i])
            new_u_lo[i] = _intersect_normal(lower_curve, C[i], -n_c[i], u_lo[i])

        # Enforce monotonicity (non-folding correspondence).
        if np.any(np.diff(new_u_up) <= 0) or np.any(np.diff(new_u_lo) <= 0):
            raise RuntimeError("Normal correspondence folded (non-monotone mapping).")

        # Convergence on parameter movement.
        du = max(
            np.max(np.abs(new_u_up - u_up)),
            np.max(np.abs(new_u_lo - u_lo)),
        )
        u_up, u_lo = new_u_up, new_u_lo

        # Optional stagnation guard.
        if abs(prev_err - du) < 1e-16 and du > tol:
            pass
        prev_err = du

        if du < tol:
            break
    else:
        raise RuntimeError("Did not converge to a normal-offset decomposition.")

    # Final curves from converged correspondence.
    U = upper_curve.evaluate_at(u_up)
    L = lower_curve.evaluate_at(u_lo)
    C = 0.5 * (U + L)

    camber_curve = BSpline2D(C)
    t_c = normalize_2d(camber_curve.first_deriv_at(s))
    n_c = rotate_2d_90ccw(t_c)

    # Thickness from signed projection on camber normals.
    half_t = 0.5 * np.sum((U - L) * n_c, axis=1)
    thickness = 2.0 * half_t

    # Reconstruct and compute exactness error at sample nodes.
    U_hat = C + half_t[:, None] * n_c
    L_hat = C - half_t[:, None] * n_c

    err_u = np.max(np.linalg.norm(U_hat - U, axis=1))
    err_l = np.max(np.linalg.norm(L_hat - L, axis=1))
    max_err = max(err_u, err_l)

    if max_err > tol * 50:
        raise RuntimeError(
            f"No exact normal-offset solution for this data at tolerance; max_err={max_err:.3e}"
        )

    # Thickness distribution as x-based curve for compatibility.
    # (If x is non-monotone, use camber parameter s instead.)
    thickness_curve = BSpline2D(np.column_stack([C[:, 0], thickness]))

    diagnostics = {
        "max_reconstruction_error": float(max_err),
        "max_upper_error": float(err_u),
        "max_lower_error": float(err_l),
        "converged": True,
        "n_samples": int(n_samples),
    }
    return camber_curve, thickness_curve, diagnostics


def solve_normal_offset_cst(
    self,
    n_coefficients: int = 8,
    n_samples: int = 300,
    max_nfev: int = 300,
    loss: str = "soft_l1",
    f_scale: float = 1e-3,
):
    """
    Fit camber/thickness CST curves so that normal-offset reconstruction matches
    this airfoil's upper/lower surfaces as closely as possible.

    Returns:
        tuple[CSTCurve, CSTCurve, dict]:
            camber_curve, thickness_curve, diagnostics
    """
    import numpy as np
    import scipy.optimize as opt

    from .spline import CSTCurve
    from ..util import cosine_spacing

    # Chordwise stations for fitting.
    x = cosine_spacing(0.0, 1.0, num=n_samples)

    # Target surfaces as y(x). These are already available on AirfoilBase.
    y_u_target = self.upper_surface_at(x)
    y_l_target = self.lower_surface_at(x)

    # Vertical initialization (very stable starting point).
    y_c0 = 0.5 * (y_u_target + y_l_target)
    t0 = np.maximum(y_u_target - y_l_target, 1e-6)

    camber0 = CSTCurve.fit(
        np.column_stack([x, y_c0]),
        num_coefficients=n_coefficients,
        n1=1.0,  # avoids LE derivative singularity in camber
        n2=1.0,
    )
    thick0 = CSTCurve.fit(
        np.column_stack([x, t0]),
        num_coefficients=n_coefficients,
        n1=0.5,
        n2=1.0,
    )

    p0 = np.concatenate([camber0.coefficients, thick0.coefficients])

    # Residual weights (tune as needed).
    w_data = 1.0
    w_pos = 50.0
    w_smooth = 1e-3
    w_te = 20.0

    x_eval = np.clip(x, 1e-8, 1.0 - 1e-8)

    def residuals(p: np.ndarray) -> np.ndarray:
        c_coef = p[:n_coefficients]
        t_coef = p[n_coefficients:]

        camber = CSTCurve(c_coef, n1=1.0, n2=1.0)
        thickness = CSTCurve(t_coef, n1=0.5, n2=1.0)

        y_c = camber(x_eval)
        dyc_dx = camber.first_deriv_at(x_eval)

        # Unit camber normal n = (-dy, 1) / sqrt(1+dy^2)
        denom = np.sqrt(1.0 + dyc_dx**2)
        nx = -dyc_dx / denom
        ny = 1.0 / denom

        t_full = thickness(x_eval)
        t_half = 0.5 * t_full

        # Normal-offset reconstruction
        x_u = np.clip(x_eval + nx * t_half, 0.0, 1.0)
        y_u = y_c + ny * t_half

        x_l = np.clip(x_eval - nx * t_half, 0.0, 1.0)
        y_l = y_c - ny * t_half

        # Compare reconstructed y against target surfaces at reconstructed x
        r_data = np.concatenate([
            y_u - self.upper_surface_at(x_u),
            y_l - self.lower_surface_at(x_l),
        ]) * w_data

        # Soft positivity constraint: thickness >= 0
        neg_t = np.minimum(0.0, t_full)
        r_pos = w_pos * neg_t

        # Mild smoothness on coefficients (2nd finite diff)
        d2c = np.diff(c_coef, n=2)
        d2t = np.diff(t_coef, n=2)
        r_smooth = w_smooth * np.concatenate([d2c, d2t])

        # Optional TE consistency (shared x at trailing edge)
        r_te = w_te * np.array([
            x_u[-1] - 1.0,
            x_l[-1] - 1.0,
        ])

        return np.concatenate([r_data, r_pos, r_smooth, r_te])

    result = opt.least_squares(
        residuals,
        p0,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=0,
    )

    c_opt = result.x[:n_coefficients]
    t_opt = result.x[n_coefficients:]

    camber_curve = CSTCurve(c_opt, n1=1.0, n2=1.0)
    thickness_curve = CSTCurve(t_opt, n1=0.5, n2=1.0)

    # Diagnostics on fit grid
    y_c = camber_curve(x_eval)
    dy = camber_curve.first_deriv_at(x_eval)
    denom = np.sqrt(1.0 + dy**2)
    nx = -dy / denom
    ny = 1.0 / denom
    t_half = 0.5 * thickness_curve(x_eval)

    x_u = np.clip(x_eval + nx * t_half, 0.0, 1.0)
    y_u = y_c + ny * t_half
    x_l = np.clip(x_eval - nx * t_half, 0.0, 1.0)
    y_l = y_c - ny * t_half

    yu_err = y_u - self.upper_surface_at(x_u)
    yl_err = y_l - self.lower_surface_at(x_l)

    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "rmse_upper": float(np.sqrt(np.mean(yu_err**2))),
        "rmse_lower": float(np.sqrt(np.mean(yl_err**2))),
        "max_abs_upper": float(np.max(np.abs(yu_err))),
        "max_abs_lower": float(np.max(np.abs(yl_err))),
    }

    return camber_curve, thickness_curve, info
