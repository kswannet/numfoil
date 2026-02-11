from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider


@dataclass(frozen=True)
class SliderSpec:
    key: str
    label: str
    vmin: float
    vmax: float
    v0: float
    step: Optional[float] = None


@dataclass(frozen=True)
class OverlaySpec:
    key: str
    label: str
    default: bool = False


class InteractiveAirfoilPlot:
    """Fast interactive airfoil plot using matplotlib + blitting.

    Design goals:
      - Smooth realtime updates (avoid full redraw)
      - No dependency on any project internals besides a geometry callback
      - Overlay toggles for extra geometric annotations

    Callbacks:
      - compute_airfoil(params) -> (xy_upper, xy_lower)
      - compute_overlays(params) -> dict[str, overlay_payload]

    Where xy_upper/xy_lower are arrays of shape [N, 2] in chord-line coords.
    """

    def __init__(
        self,
        *,
        slider_specs: Sequence[SliderSpec],
        overlay_specs: Sequence[OverlaySpec],
        compute_airfoil: Callable[[Dict[str, float]], Tuple[np.ndarray, np.ndarray]],
        compute_overlays: Optional[Callable[[Dict[str, float]], Dict[str, object]]] = None,
        title: str = "Interactive airfoil",
        xlim: Tuple[float, float] = (-0.05, 1.05),
        ylim: Tuple[float, float] = (-0.2, 0.3),
        n_points_hint: int = 200,
    ) -> None:
        self.slider_specs = list(slider_specs)
        self.overlay_specs = list(overlay_specs)
        self.compute_airfoil = compute_airfoil
        self.compute_overlays = compute_overlays
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.n_points_hint = n_points_hint

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._sliders: Dict[str, Slider] = {}
        self._checks: Optional[CheckButtons] = None
        self._overlay_state: Dict[str, bool] = {s.key: s.default for s in self.overlay_specs}
        self._overlay_artists: Dict[str, List[plt.Artist]] = {s.key: [] for s in self.overlay_specs}

        self._line_upper = None
        self._line_lower = None
        self._line_camber = None
        self._background = None

        # cached arrays to avoid allocations
        self._upper_xy = np.zeros((self.n_points_hint, 2), dtype=float)
        self._lower_xy = np.zeros((self.n_points_hint, 2), dtype=float)

    def _get_params(self) -> Dict[str, float]:
        return {k: float(sl.val) for k, sl in self._sliders.items()}

    def _draw_static(self) -> None:
        assert self._ax is not None
        self._ax.set_title(self.title)
        self._ax.set_xlim(self.xlim)
        self._ax.set_ylim(self.ylim)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlabel("x/c")
        self._ax.set_ylabel("y/c")
        self._ax.grid(True, alpha=0.25)

    def _init_lines(self) -> None:
        assert self._ax is not None

        xy_u, xy_l = self.compute_airfoil(self._get_params())

        self._line_upper, = self._ax.plot(xy_u[:, 0], xy_u[:, 1], lw=2)
        self._line_lower, = self._ax.plot(xy_l[:, 0], xy_l[:, 1], lw=2)

        # Optional camber overlay drawn as a normal line artist (toggle-controlled)
        self._line_camber, = self._ax.plot([], [], lw=1.5, linestyle="--")

    def _update_lines(self) -> None:
        assert self._ax is not None
        assert self._line_upper is not None and self._line_lower is not None

        params = self._get_params()
        xy_u, xy_l = self.compute_airfoil(params)
        self._line_upper.set_data(xy_u[:, 0], xy_u[:, 1])
        self._line_lower.set_data(xy_l[:, 0], xy_l[:, 1])

        if "camber" in self._overlay_state:
            if self._overlay_state["camber"]:
                # use fast midpoint camber if already have same x sampling
                if xy_u.shape == xy_l.shape and np.allclose(xy_u[:, 0], xy_l[:, 0], atol=0, rtol=0):
                    x = xy_u[:, 0]
                    y = 0.5 * (xy_u[:, 1] + xy_l[:, 1])
                    self._line_camber.set_data(x, y)
                else:
                    self._line_camber.set_data([], [])
            else:
                self._line_camber.set_data([], [])

        # Update extra overlays (markers/annotations) only if callback exists
        if self.compute_overlays is not None:
            payloads = self.compute_overlays(params)
            for key in self._overlay_artists.keys():
                # clear artists for that key each update (keeps code simple)
                for artist in self._overlay_artists[key]:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                self._overlay_artists[key] = []

                if not self._overlay_state.get(key, False):
                    continue

                payload = payloads.get(key)
                if payload is None:
                    continue

                # Payload conventions:
                # - "point": np.ndarray [2]
                # - "vline": x float
                # - "hline": y float
                # - "text": (x: float, y: float, s: str)
                if isinstance(payload, dict) and "point" in payload:
                    p = np.asarray(payload["point"]).reshape(2)
                    (artist,) = self._ax.plot([p[0]], [p[1]], marker="o")
                    self._overlay_artists[key].append(artist)
                if isinstance(payload, dict) and "vline" in payload:
                    x = float(payload["vline"])
                    artist = self._ax.axvline(x, lw=1)
                    self._overlay_artists[key].append(artist)
                if isinstance(payload, dict) and "hline" in payload:
                    y = float(payload["hline"])
                    artist = self._ax.axhline(y, lw=1)
                    self._overlay_artists[key].append(artist)
                if isinstance(payload, dict) and "text" in payload:
                    tx, ty, s = payload["text"]
                    artist = self._ax.text(float(tx), float(ty), str(s))
                    self._overlay_artists[key].append(artist)

    def _blit(self) -> None:
        assert self._fig is not None
        assert self._ax is not None

        if self._background is None:
            self._fig.canvas.draw()
            self._background = self._fig.canvas.copy_from_bbox(self._ax.bbox)

        self._fig.canvas.restore_region(self._background)

        for ln in (self._line_upper, self._line_lower, self._line_camber):
            if ln is not None:
                self._ax.draw_artist(ln)

        for artists in self._overlay_artists.values():
            for a in artists:
                self._ax.draw_artist(a)

        self._fig.canvas.blit(self._ax.bbox)
        self._fig.canvas.flush_events()

    def _on_slider_change(self, _val: float) -> None:
        self._update_lines()
        self._blit()

    def _on_check_click(self, label: str) -> None:
        # map label back to key
        key = None
        for spec in self.overlay_specs:
            if spec.label == label:
                key = spec.key
                break
        if key is None:
            return
        self._overlay_state[key] = not self._overlay_state.get(key, False)
        self._update_lines()
        # background invalidated because we may have created/removed artists
        self._background = None
        self._blit()

    def show(self) -> None:
        plt.ion()

        fig = plt.figure(figsize=(11.5, 6.5))
        ax = fig.add_axes([0.06, 0.12, 0.62, 0.82])
        self._fig, self._ax = fig, ax

        self._draw_static()
        self._init_lines()

        # Slider panel (multi-column if needed)
        slider_left = 0.71
        slider_right = 0.98
        slider_top = 0.94
        slider_bottom = 0.16
        slider_height = 0.022
        slider_gap = 0.004

        per_col = max(1, int((slider_top - slider_bottom) / (slider_height + slider_gap)))
        n_cols = int(np.ceil(len(self.slider_specs) / per_col))
        col_width = (slider_right - slider_left) / max(1, n_cols)

        for i, spec in enumerate(self.slider_specs):
            col = i // per_col
            row = i % per_col
            y = slider_top - row * (slider_height + slider_gap)
            x0 = slider_left + col * col_width
            sax = fig.add_axes([x0, y, col_width - 0.01, slider_height])
            slider = Slider(
                ax=sax,
                label=spec.label,
                valmin=spec.vmin,
                valmax=spec.vmax,
                valinit=spec.v0,
                valstep=spec.step,
            )
            slider.on_changed(self._on_slider_change)
            self._sliders[spec.key] = slider

        # Overlay checkboxes
        if self.overlay_specs:
            check_ax = fig.add_axes([0.71, 0.03, 0.27, 0.12])
            labels = [s.label for s in self.overlay_specs]
            actives = [bool(self._overlay_state[s.key]) for s in self.overlay_specs]
            checks = CheckButtons(check_ax, labels, actives)
            checks.on_clicked(self._on_check_click)
            self._checks = checks

        self._update_lines()
        self._background = None
        self._blit()

        plt.show(block=True)
