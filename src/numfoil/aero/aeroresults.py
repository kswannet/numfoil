import numpy as np

from warnings import warn as warning


def _fmt_re(re_list):
    """Format a list of Re values for display.

    Uses engineering notation (e.g. ``1M``, ``500k``).
    Trailing zeros after the decimal are stripped and at
    most 3 decimals are shown.

    Args:
        re_list (list): reynolds numbers.

    Returns:
        str: Comma-separated formatted string.
    """
    parts = []
    for r in sorted(re_list):
        if abs(r) >= 1e6:
            s = f"{r / 1e6:.3f}".rstrip("0").rstrip(".")
            parts.append(f"{s}M")
        elif abs(r) >= 1e3:
            s = f"{r / 1e3:.3f}".rstrip("0").rstrip(".")
            parts.append(f"{s}k")
        else:
            s = f"{r:.3f}".rstrip("0").rstrip(".")
            parts.append(s)
    return ", ".join(parts)


def _keyed_styles(keys):
    """Assign colors to (alpha, Re) keys.

    Each unique Re gets a distinct base color.  Within one
    Re, increasing alpha values are shown as progressively
    darker shades.

    Args:
        keys (list): Sorted ``(alpha, Re)`` tuples.

    Returns:
        dict: ``(alpha, Re)`` to RGBA color tuple.
    """
    import matplotlib.cm as cm

    base_colors = [cm.tab10(i) for i in range(10)]

    # Map each unique Re to a base color
    re_vals = list(dict.fromkeys(r for _, r in keys))
    re_color = {
        r: np.array(base_colors[i % len(base_colors)][:3])
        for i, r in enumerate(re_vals)
    }

    # Group alphas per Re for shading
    alphas_per_re = {}
    for a, r in keys:
        alphas_per_re.setdefault(r, []).append(a)

    styles = {}
    for a, r in keys:
        a_list = sorted(set(alphas_per_re[r]))
        idx = a_list.index(a)
        n = len(a_list)
        # Blend from light tint (frac=0.45) to dark shade (frac=1.3).
        # frac < 1 lightens towards white; frac > 1 darkens past the
        # base color so both ends of the range are clearly visible.
        frac = (0.45 + 0.85 * idx / (n - 1)) if n > 1 else 1.0
        base = re_color[r]
        color = tuple(
            np.clip(1.0 - frac * (1.0 - base), 0.0, 1.0)
        ) + (1.0,)
        styles[(a, r)] = color
    return styles


class PolarData:
    """Polar sweep data keyed by reynolds number.

    Stores alpha-sweep results (CL, CD, CM, etc.) for one or
    more reynolds numbers.  Provides plotting of lift curves,
    drag polars, moment curves, and lift-to-drag ratio.

    Each reynolds entry is a dict of numpy arrays with
    keys like ``'alpha'``, ``'CL'``, ``'CD'``, ``'CDp'``,
    ``'CM'``, ``'Top_Xtr'``, ``'Bot_Xtr'``.

    Not intended to be instantiated directly; populated by
    e.g. :class:`XFoil` methods.

    Example::

        >>> polar = res.polar
        >>> polar(Re=1e6)["CL"]              # 1D ndarray
        >>> polar(Re=1e6)[["alpha", "CL"]]   # (N, 2) ndarray
        >>> polar["CL"]                       # {Re: ndarray} or ndarray
        >>> polar[0]                          # dict for first Re
        >>> polar.columns                     # ['alpha', 'CL', ...]
        >>> polar.to_dataframe()              # pandas DataFrame
    """

    def __init__(self):
        self._data = {}

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        if not self._data:
            return "PolarData(empty)"
        return f"PolarData(Re=[{_fmt_re(self._data)}])"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        """Access data by positional index or column name.

        Args:
            key (int): Positional index into sorted Re
                keys.  Returns the dict-of-arrays for
                that Re.
            key (str): Column name (e.g. ``'CL'``).
                Returns a plain ndarray if one Re, else
                ``{Re: ndarray}`` dict.
            key (list[str]): Multiple column names.
                Returns ``(N, cols)`` ndarray if one Re,
                else ``{Re: (N, cols) ndarray}`` dict.

        Returns:
            dict or ndarray: Depends on key type.
        """
        if isinstance(key, (int, np.integer)):
            sorted_keys = sorted(self._data)
            return self._data[sorted_keys[key]]
        if isinstance(key, str):
            out = {
                r: self._data[r][key]
                for r in sorted(self._data)
            }
            if len(out) == 1:
                return next(iter(out.values()))
            return out
        if isinstance(key, list):
            out = {
                r: np.column_stack(
                    [self._data[r][k] for k in key]
                )
                for r in sorted(self._data)
            }
            if len(out) == 1:
                return next(iter(out.values()))
            return out
        raise TypeError(
            f"key must be int, str, or list[str], "
            f"got {type(key).__name__}"
        )

    def __contains__(self, item):
        """Check column name or Re membership."""
        if isinstance(item, str):
            if not self._data:
                return False
            return item in next(
                iter(self._data.values())
            )
        return item in self._data

    def __iter__(self):
        return iter(sorted(self._data))

    def __call__(self, Re=None):
        """Filter by Re, returning a new PolarData.

        Returns a shallow copy containing only the
        matching Re groups (arrays are shared, not
        copied).

        Args:
            Re (float, list, or None): Reynolds filter.
                ``None`` returns a copy of all data.

        Returns:
            PolarData: Filtered subset.
        """
        new = PolarData()
        for r in self._select_Re(Re):
            if r in self._data:
                new._data[r] = self._data[r]
        return new

    def _add(self, Re, raw):
        """Store a polar dict for a reynolds number.

        Values are converted to numpy arrays on insertion.
        """
        self._data[Re] = {
            k: np.asarray(v) for k, v in raw.items()
        }

    @property
    def reynolds(self):
        """Sorted list of reynolds numbers with data."""
        return sorted(self._data.keys())

    Re = reynolds

    @property
    def columns(self):
        """Column names in the data."""
        if not self._data:
            return []
        return list(
            next(iter(self._data.values())).keys()
        )

    def _arrays(self, Re):
        """Return (alpha, CL, CD, CM) as numpy arrays.

        Args:
            Re (float): reynolds number.

        Returns:
            tuple: Four ndarrays.
        """
        d = self._data[Re]
        return (
            np.asarray(d["alpha"]),
            np.asarray(d["CL"]),
            np.asarray(d["CD"]),
            np.asarray(d.get("CM", [])),
        )

    def _select_Re(self, Re):
        """Return list of Re values to operate on.

        Args:
            Re: ``None`` (all), scalar, or list.

        Returns:
            list: reynolds numbers.
        """
        if Re is None:
            return self.reynolds
        if isinstance(Re, (int, float, np.floating)):
            return [float(Re)]
        return [float(r) for r in Re]

    # -- interpolation shortcuts ------------------------------

    def CL(self, alpha, Re=None):
        """Interpolated CL at a given angle of attack.

        Uses linear interpolation on the polar data.

        Args:
            alpha (float): Angle of attack in degrees.
            Re (float or None): Specific reynolds number.
                If ``None`` and only one Re exists, uses
                that one.  If ``None`` and multiple Re
                exist, returns a dict.

        Returns:
            float or dict: Interpolated CL value(s).
        """
        if Re is not None:
            a, cl, _, _ = self._arrays(Re)
            return float(np.interp(alpha, a, cl))
        out = {}
        for r in self:
            a, cl, _, _ = self._arrays(r)
            out[r] = float(np.interp(alpha, a, cl))
        return self._scalar_or_dict(out)

    def CD(self, alpha, Re=None):
        """Interpolated CD at a given angle of attack.

        Uses linear interpolation on the polar data.

        Args:
            alpha (float): Angle of attack in degrees.
            Re (float or None): Specific reynolds number.

        Returns:
            float or dict: Interpolated CD value(s).
        """
        if Re is not None:
            a, _, cd, _ = self._arrays(Re)
            return float(np.interp(alpha, a, cd))
        out = {}
        for r in self:
            a, _, cd, _ = self._arrays(r)
            out[r] = float(np.interp(alpha, a, cd))
        return self._scalar_or_dict(out)

    def CM(self, alpha, Re=None):
        """Interpolated CM at a given angle of attack.

        Uses linear interpolation on the polar data.

        Args:
            alpha (float): Angle of attack in degrees.
            Re (float or None): Specific reynolds number.

        Returns:
            float or dict: Interpolated CM value(s).
        """
        if Re is not None:
            a, _, _, cm = self._arrays(Re)
            if len(cm) == 0:
                return float("nan")
            return float(np.interp(alpha, a, cm))
        out = {}
        for r in self:
            a, _, _, cm = self._arrays(r)
            if len(cm) == 0:
                out[r] = float("nan")
            else:
                out[r] = float(np.interp(alpha, a, cm))
        return self._scalar_or_dict(out)

    # -- aerodynamic properties (scalar or dict) --------------

    @property
    def CL_max(self):
        """Maximum lift coefficient.

        Returns:
            float or dict: Peak CL value from the polar.
        """
        out = {}
        for r in self:
            _, CL, _, _ = self._arrays(r)
            out[r] = float(np.nanmax(CL))
        return self._scalar_or_dict(out)

    @property
    def alpha_CL_max(self):
        """Angle of attack at maximum CL (stall angle).

        Returns:
            float or dict: Alpha in degrees at CL_max.
        """
        out = {}
        for r in self:
            a, CL, _, _ = self._arrays(r)
            out[r] = float(a[np.nanargmax(CL)])
        return self._scalar_or_dict(out)

    @property
    def CL0(self):
        """Lift coefficient at alpha = 0 (interpolated).

        Returns:
            float or dict: CL at zero angle of attack.
        """
        out = {}
        for r in self:
            a, CL, _, _ = self._arrays(r)
            out[r] = float(np.interp(0.0, a, CL))
        return self._scalar_or_dict(out)

    @property
    def alpha_L0(self):
        """Zero-lift angle of attack (interpolated).

        Finds alpha where CL = 0 by sorting the polar by
        CL and interpolating.

        Returns:
            float or dict: Alpha in degrees where CL = 0.
        """
        out = {}
        for r in self:
            a, CL, _, _ = self._arrays(r)
            order = np.argsort(CL)
            out[r] = float(
                np.interp(0.0, CL[order], a[order])
            )
        return self._scalar_or_dict(out)

    @property
    def LD_max(self):
        r"""Maximum lift-to-drag ratio.

        Math:
            .. math::

                (L/D)_{\max} =
                    \max\!\left(\frac{C_L}{C_D}\right)

        Returns:
            float or dict: Peak L/D.
        """
        out = {}
        for r in self:
            _, CL, CD, _ = self._arrays(r)
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):
                LD = np.where(CD > 0, CL / CD, np.nan)
            out[r] = float(np.nanmax(LD))
        return self._scalar_or_dict(out)

    @property
    def CL_opt(self):
        """CL at maximum lift-to-drag ratio.

        Returns:
            float or dict: CL at (L/D)_max.
        """
        out = {}
        for r in self:
            _, CL, CD, _ = self._arrays(r)
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):
                LD = np.where(CD > 0, CL / CD, np.nan)
            out[r] = float(CL[np.nanargmax(LD)])
        return self._scalar_or_dict(out)

    @property
    def alpha_opt(self):
        """Angle of attack at maximum L/D.

        Returns:
            float or dict: Alpha in degrees at (L/D)_max.
        """
        out = {}
        for r in self:
            a, CL, CD, _ = self._arrays(r)
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):
                LD = np.where(CD > 0, CL / CD, np.nan)
            out[r] = float(a[np.nanargmax(LD)])
        return self._scalar_or_dict(out)

    @property
    def CD_min(self):
        """Minimum drag coefficient.

        Returns:
            float or dict: Minimum CD from the polar.
        """
        out = {}
        for r in self:
            _, _, CD, _ = self._arrays(r)
            out[r] = float(np.nanmin(CD))
        return self._scalar_or_dict(out)

    @property
    def CL_alpha(self):
        r"""Lift-curve slope dCL/d\u03b1 in the linear region.

        Estimated via least-squares fit to data where
        alpha < alpha_CL_max - 2 degrees.

        Returns:
            float or dict: dCL/d\u03b1 in per-degree units.

        Math:
            .. math::

                C_L(\alpha) \approx
                    \frac{dC_L}{d\alpha}\,\alpha
                    + C_{L,0}
        """
        out = {}
        for r in self:
            a, CL, _, _ = self._arrays(r)
            a_stall = a[np.nanargmax(CL)]
            mask = a < (a_stall - 2.0)
            if np.sum(mask) < 2:
                mask = a <= a_stall
            if np.sum(mask) < 2:
                out[r] = float("nan")
            else:
                coeffs = np.polyfit(
                    a[mask], CL[mask], 1
                )
                out[r] = float(coeffs[0])
        return self._scalar_or_dict(out)

    @property
    def CM0(self):
        """Pitching-moment coefficient at alpha = 0
        (interpolated).

        Returns:
            float or dict: CM at zero angle of attack.
        """
        out = {}
        for r in self:
            a, _, _, CM = self._arrays(r)
            if len(CM) == 0:
                out[r] = float("nan")
            else:
                out[r] = float(
                    np.interp(0.0, a, CM)
                )
        return self._scalar_or_dict(out)

    # -- summary ----------------------------------------------

    def summary(self):
        """Print a tabulated summary of key aero metrics.

        Displays CL_max, alpha_stall, CL0, alpha_L=0,
        (L/D)_max, CL_opt, alpha_opt, CD_min, and
        dCL/d-alpha for every reynolds number.

        Returns:
            str: The formatted summary text.

        Example::

            >>> res.polar.summary()
        """
        Re_list = self.reynolds

        header = (
            f"{'Airfoil':<16} {'Re':>10}  "
            f"{'CL_max':>7} {'\u03b1_st':>5} "
            f"{'CL0':>7} {'\u03b1_L0':>6} "
            f"{'L/D_mx':>7} {'CL_opt':>7} "
            f"{'\u03b1_opt':>5} "
            f"{'CD_min':>9} {'dCL/d\u03b1':>7}"
        )
        sep = "=" * len(header)
        lines = [sep, header, sep]

        for Re in Re_list:
            a, CL, CD, CM = self._arrays(Re)

            cl_max = float(np.nanmax(CL))
            a_stall = float(a[np.nanargmax(CL)])
            cl0 = float(np.interp(0.0, a, CL))

            order = np.argsort(CL)
            a_l0 = float(
                np.interp(0.0, CL[order], a[order])
            )

            with np.errstate(
                divide="ignore", invalid="ignore"
            ):
                LD = np.where(CD > 0, CL / CD, np.nan)
            ld_max = float(np.nanmax(LD))
            cl_opt = float(CL[np.nanargmax(LD)])
            a_opt = float(a[np.nanargmax(LD)])
            cd_min = float(np.nanmin(CD))

            mask = a < (a_stall - 2.0)
            if np.sum(mask) < 2:
                mask = a <= a_stall
            if np.sum(mask) < 2:
                cl_a = float("nan")
            else:
                cl_a = float(
                    np.polyfit(a[mask], CL[mask], 1)[0]
                )

            lines.append(
                f"{self.label:<16} "
                f"{Re:>10.0f}  "
                f"{cl_max:>7.4f} "
                f"{a_stall:>5.1f} "
                f"{cl0:>7.4f} "
                f"{a_l0:>6.2f} "
                f"{ld_max:>7.2f} "
                f"{cl_opt:>7.4f} "
                f"{a_opt:>5.1f} "
                f"{cd_min:>9.6f} "
                f"{cl_a:>7.4f}"
            )

        lines.append(sep)
        text = "\n".join(lines)
        print(text)
        return text

    # -- helpers ----------------------------------------------

    def _scalar_or_dict(self, mapping):
        """Return scalar if one Re, else dict.

        Args:
            mapping (dict): ``{Re: value}`` mapping.

        Returns:
            float or dict: Single value or full mapping.
        """
        if len(mapping) == 1:
            return next(iter(mapping.values()))
        return mapping

    # -- plotting ---------------------------------------------

    def plot(self, Re=None, show=True):
        """Plot polar comparison (2\u00d72 grid).

        Panels: CL vs \u03b1, CD vs \u03b1, CM vs \u03b1,
        CL vs CD (drag polar).  Each reynolds number is a
        separate curve.

        Args:
            Re (float, list, or None): Filter by reynolds
                number.  ``None`` plots all.
            show (bool): Call ``plt.show()``.

        Returns:
            tuple: ``(fig, axes)``
        """
        import matplotlib.pyplot as plt

        Re_list = self._select_Re(Re)
        if not Re_list:
            raise RuntimeError("No polar data to plot.")

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        for r in Re_list:
            alpha, CL, CD, CM = self._arrays(r)
            kw = dict(
                label=f"Re={r:.2e}",
                marker=".", markersize=3,
            )
            axes[0, 0].plot(alpha, CL, **kw)
            axes[0, 1].plot(alpha, CD, **kw)
            if len(CM):
                axes[1, 0].plot(alpha, CM, **kw)
            axes[1, 1].plot(CD, CL, **kw)

        for ax, (xl, yl, t) in zip(
            axes.flat,
            [
                ("\u03b1 (\u00b0)", "CL", "Lift curve"),
                ("\u03b1 (\u00b0)", "CD", "Drag curve"),
                ("\u03b1 (\u00b0)", "CM", "Moment curve"),
                ("CD", "CL", "Drag polar"),
            ],
        ):
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(t)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        fig.suptitle("Polar comparison", fontsize=13)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_drag(self, Re=None, ax=None, show=True):
        """Plot drag polar (CL vs CD).

        Args:
            Re (float, list, or None): Filter by Re.
            ax: Optional Matplotlib Axes.
            show (bool): Call ``plt.show()``.

        Returns:
            tuple: ``(fig, ax)``
        """
        import matplotlib.pyplot as plt

        Re_list = self._select_Re(Re)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()

        for r in Re_list:
            _, CL, CD, _ = self._arrays(r)
            ax.plot(
                CD, CL,
                label=f"Re={r:.2e}",
                marker=".", markersize=3,
            )

        ax.set_xlabel("CD")
        ax.set_ylabel("CL")
        ax.set_title("Drag polar")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_LD(self, Re=None, ax=None, show=True):
        """Plot L/D vs alpha.

        Args:
            Re (float, list, or None): Filter by Re.
            ax: Optional Matplotlib Axes.
            show (bool): Call ``plt.show()``.

        Returns:
            tuple: ``(fig, ax)``
        """
        import matplotlib.pyplot as plt

        Re_list = self._select_Re(Re)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.get_figure()

        for r in Re_list:
            alpha, CL, CD, _ = self._arrays(r)
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):
                LD = np.where(CD > 0, CL / CD, np.nan)
            ax.plot(
                alpha, LD,
                label=f"Re={r:.2e}",
                marker=".", markersize=3,
            )

        ax.set_xlabel("\u03b1 (\u00b0)")
        ax.set_ylabel("L / D")
        ax.set_title("Lift-to-drag ratio")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # -- data export ------------------------------------------
    def to_dataframe(self, Re=None):
        """Export polar data as a pandas DataFrame.

        Returns a tidy DataFrame with one row per
        operating point.  A ``Re`` column is prepended
        so that data from multiple reynolds numbers can
        be distinguished.

        Args:
            Re (float, list, or None): Filter by reynolds
                number.  ``None`` exports all.

        Returns:
            pandas.DataFrame: Columns include ``Re``,
                ``alpha``, ``CL``, ``CD``, and any other
                fields present in the polar.

        Raises:
            ImportError: If pandas is not installed.

        Example::

            >>> df = res.polar.to_dataframe()
            >>> df = res.polar.to_dataframe(Re=1e6)
        """
        import pandas as pd

        Re_list = self._select_Re(Re)
        frames = []
        for r in Re_list:
            d = self._data[r]
            n = len(next(iter(d.values())))
            frame = {"Re": np.full(n, r)}
            frame.update(d)
            frames.append(pd.DataFrame(frame))
        if not frames:
            return pd.DataFrame()
        return pd.concat(
            frames, ignore_index=True
        )

    def to_tensor(self, Re=None, columns=None):
        """Export polar data as a dense ``[R, A, C]`` tensor.

        This is a fast-path export for vectorized workflows where the
        full Reynolds-by-alpha grid is needed as one array.

        Args:
            Re (float, list, or None): Optional Reynolds filter.
            columns (str, list[str], or None): Polar columns to include.
                ``None`` uses all columns except ``alpha``.

        Returns:
            tuple: ``(values, re_axis, alpha_axis, columns)`` where:
                - ``values`` has shape ``[R, A, C]``
                - ``re_axis`` has shape ``[R]``
                - ``alpha_axis`` has shape ``[A]``
                - ``columns`` is the resolved list of column names

        Raises:
            ValueError: If selected Reynolds blocks do not share the same
                alpha grid, or if a selected column has an inconsistent
                length.
        """
        re_list = self._select_Re(Re)

        if columns is None:
            resolved_columns = [c for c in self.columns if c != "alpha"]
        elif isinstance(columns, str):
            resolved_columns = [columns]
        else:
            resolved_columns = list(columns)

        if len(resolved_columns) == 0:
            raise ValueError("columns must contain at least one field")

        if not re_list:
            empty = np.empty((0, 0, len(resolved_columns)), dtype=float)
            return empty, np.asarray([], dtype=float), np.asarray([], dtype=float), resolved_columns

        alpha_axis = None
        blocks = []
        for re_value in re_list:
            data = self._data[re_value]
            alpha = np.asarray(data["alpha"], dtype=float).reshape(-1)

            if alpha_axis is None:
                alpha_axis = alpha
            else:
                same_shape = alpha.shape == alpha_axis.shape
                same_values = np.allclose(alpha, alpha_axis, equal_nan=True)
                if not (same_shape and same_values):
                    raise ValueError(
                        "Cannot build dense tensor: alpha grids differ across Reynolds values. "
                        "Use indexed access or to_dataframe() for ragged data."
                    )

            column_arrays = []
            for column in resolved_columns:
                if column in data:
                    values = np.asarray(data[column], dtype=float).reshape(-1)
                else:
                    values = np.full(alpha.shape, np.nan, dtype=float)

                if values.shape != alpha.shape:
                    raise ValueError(
                        f"Column '{column}' has shape {values.shape}, expected {alpha.shape}."
                    )
                column_arrays.append(values)

            blocks.append(np.column_stack(column_arrays))

        values = np.stack(blocks, axis=0)
        return values, np.asarray(re_list, dtype=float), alpha_axis.copy(), resolved_columns


class CpData:
    """Pressure-distribution data keyed by (alpha, Re).

    Stores Cp distributions from XFOIL's ``CPWR`` command.
    Each entry is a dict with keys ``'x'`` and ``'Cp'``
    (numpy arrays).

    Plotting supports two overlay modes:

    * Fix alpha, overlay reynolds numbers.
    * Fix reynolds, overlay angles of attack.

    Not intended to be instantiated directly; populated by
    :class:`XFoil` methods.

    Example::

        >>> cp = res.cp
        >>> cp(alpha=5.0, Re=1e6)["Cp"]      # 1D ndarray
        >>> cp(Re=1e6)[["x", "Cp"]]           # {(a,Re): (N,2)}
        >>> cp["Cp"]                           # {(a,Re): ndarray}
        >>> cp[0]                             # dict for first entry
        >>> cp.columns                        # ['x', 'Cp']
        >>> cp.to_dataframe(Re=1e6)           # pandas DataFrame
    """

    def __init__(self):
        self._data = {}

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        if not self._data:
            return "CpData(empty)"
        re = sorted({k[1] for k in self._data})
        na = len({k[0] for k in self._data})
        return (
            f"CpData({len(self._data)} entries, "
            f"Re=[{_fmt_re(re)}], {na} \u03b1)"
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        """Access data by positional index or column name.

        Args:
            key (int): Positional index into sorted
                ``(alpha, Re)`` keys.
            key (str): Column name (e.g. ``'Cp'``).
                Returns ndarray if one entry, else
                ``{(alpha, Re): ndarray}`` dict.
            key (list[str]): Multiple column names.
                Returns ``(N, cols)`` ndarray if one
                entry, else
                ``{(alpha, Re): (N, cols)}`` dict.

        Returns:
            dict or ndarray: Depends on key type.
        """
        if isinstance(key, (int, np.integer)):
            sorted_keys = sorted(self._data)
            return self._data[sorted_keys[key]]
        if isinstance(key, str):
            out = {
                k: self._data[k][key]
                for k in sorted(self._data)
            }
            if len(out) == 1:
                return next(iter(out.values()))
            return out
        if isinstance(key, list):
            out = {
                k: np.column_stack(
                    [self._data[k][c] for c in key]
                )
                for k in sorted(self._data)
            }
            if len(out) == 1:
                return next(iter(out.values()))
            return out
        raise TypeError(
            f"key must be int, str, or list[str], "
            f"got {type(key).__name__}"
        )

    def __contains__(self, item):
        """Check column name or (alpha, Re) key."""
        if isinstance(item, str):
            if not self._data:
                return False
            return item in next(
                iter(self._data.values())
            )
        return item in self._data

    def __iter__(self):
        return iter(sorted(self._data))

    def __call__(self, alpha=None, Re=None):
        """Filter by alpha and/or Re.

        Returns a shallow copy containing only matching
        entries (arrays are shared, not copied).

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).

        Returns:
            CpData: Filtered subset.
        """
        new = CpData()
        for k in self._select(alpha, Re):
            new._data[k] = self._data[k]
        return new

    def _add(self, alpha, Re, raw):
        """Store a Cp distribution for (alpha, Re).

        Values are converted to numpy arrays on insertion.
        """
        self._data[(alpha, Re)] = {
            k: np.asarray(v) for k, v in raw.items()
        }

    @property
    def alphas(self):
        """Sorted unique alpha values with data."""
        return sorted({k[0] for k in self._data})

    @property
    def reynolds(self):
        """Sorted unique Re values with data."""
        return sorted({k[1] for k in self._data})

    Re = reynolds

    @property
    def columns(self):
        """Column names in the data."""
        if not self._data:
            return []
        return list(
            next(iter(self._data.values())).keys()
        )

    @staticmethod
    def _to_set(val):
        """Normalise a filter arg to a set or None."""
        if val is None:
            return None
        if isinstance(val, (int, float, np.floating)):
            return {float(val)}
        return {float(v) for v in val}

    def _select(self, alpha=None, Re=None):
        """Return filtered (alpha, Re) keys.

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).

        Returns:
            list: Matching ``(alpha, Re)`` keys, sorted.
        """
        keys = list(self._data.keys())
        a_set = self._to_set(alpha)
        r_set = self._to_set(Re)
        if a_set is not None:
            keys = [
                k for k in keys if k[0] in a_set
            ]
        if r_set is not None:
            keys = [
                k for k in keys if k[1] in r_set
            ]
        return sorted(keys)

    def plot(
        self, alpha=None, Re=None, ax=None, show=True
    ):
        """Plot Cp distributions.

        Flexible overlay logic:

        * ``plot(alpha=5.0)`` @ fix alpha, overlay Re.
        * ``plot(Re=1e6)`` @ fix Re, overlay alpha.
        * ``plot(alpha=[0, 5, 10])`` @ subset of alphas.
        * ``plot(alpha=5.0, Re=1e6)`` @ single curve.
        * ``plot()`` @ all curves (can be crowded).

        The y-axis is inverted (aerodynamic convention).
        When there are many curves (> 8), a colorbar replaces
        the legend to keep the plot readable.

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).
            ax: Optional Matplotlib Axes.
            show (bool): Call ``plt.show()``.

        Returns:
            tuple: ``(fig, ax)``
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if alpha is None and Re is None:
            warning(
                "At least one of alpha or Re must be specified."
            )

        keys = self._select(alpha, Re)
        if not keys:
            raise RuntimeError(
                "No Cp data matching the given filters."
            )

        use_colorbar = len(keys) > 8

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        if use_colorbar:
            # Build a continuous colormap per Re group
            re_vals = sorted(
                {r for _, r in keys}
            )
            # Map Re -> matplotlib colormap name
            _cmaps = [
                "Blues", "Oranges", "Greens", "Reds",
                "Purples", "YlOrBr", "PuRd", "BuGn",
                "YlGnBu", "RdPu",
            ]
            re_cmap = {
                r: plt.get_cmap(
                    _cmaps[i % len(_cmaps)]
                )
                for i, r in enumerate(re_vals)
            }

            # Per-Re alpha ranges for normalization
            alphas_per_re = {}
            for a, r in keys:
                alphas_per_re.setdefault(
                    r, []
                ).append(a)

            re_norm = {}
            for r, alist in alphas_per_re.items():
                lo, hi = min(alist), max(alist)
                if lo == hi:
                    lo, hi = lo - 1, hi + 1
                re_norm[r] = mcolors.Normalize(
                    vmin=lo, vmax=hi
                )

            for a, r in keys:
                cp = self._data[(a, r)]
                frac = re_norm[r](a)
                # Map 0.25..0.95 to avoid white/black ends
                color = re_cmap[r](0.25 + 0.7 * frac)
                ax.plot(
                    cp["x"], cp["Cp"],
                    color=color, linewidth=0.8,
                )

            # Add one colorbar per Re
            for r in re_vals:
                sm = plt.cm.ScalarMappable(
                    cmap=re_cmap[r], norm=re_norm[r],
                )
                sm.set_array([])
                cbar = fig.colorbar(
                    sm, ax=ax, pad=0.02,
                    fraction=0.04, shrink=0.8,
                )
                cbar.set_label(
                    f"Re={r:.2e}  \u03b1 (\u00b0)",
                    fontsize=8,
                )
                cbar.ax.tick_params(labelsize=7)
        else:
            styles = _keyed_styles(keys)
            unique_a = sorted({k[0] for k in keys})
            unique_r = sorted({k[1] for k in keys})
            for a, r in keys:
                cp = self._data[(a, r)]
                if len(unique_a) == 1:
                    lbl = f"Re={r:.2e}"
                elif len(unique_r) == 1:
                    lbl = f"\u03b1={a:.1f}\u00b0"
                else:
                    lbl = (
                        f"\u03b1={a:.1f}\u00b0, "
                        f"Re={r:.2e}"
                    )
                ax.plot(
                    cp["x"], cp["Cp"],
                    color=styles[(a, r)],
                    label=lbl,
                    marker=".", markersize=2,
                )
            ax.legend(fontsize=7)

        ax.invert_yaxis()
        ax.set_xlabel("x / c")
        ax.set_ylabel("Cp")

        unique_a = sorted({k[0] for k in keys})
        unique_r = sorted({k[1] for k in keys})
        if len(unique_a) == 1:
            title = (
                f"Pressure distribution \u2014 "
                f"\u03b1 = {unique_a[0]:.1f}\u00b0"
            )
        elif len(unique_r) == 1:
            title = (
                f"Pressure distribution \u2014 "
                f"Re = {unique_r[0]:.2e}"
            )
        else:
            title = "Pressure distribution"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # -- data export ------------------------------------------

    def to_dataframe(self, alpha=None, Re=None):
        """Export Cp data as a pandas DataFrame.

        Returns a tidy DataFrame with one row per surface
        point.  ``alpha`` and ``Re`` columns identify each
        distribution.

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).

        Returns:
            pandas.DataFrame: Columns ``alpha``, ``Re``,
                ``x``, ``Cp``.

        Raises:
            ImportError: If pandas is not installed.

        Example::

            >>> df = res.cp.to_dataframe(Re=1e6)
        """
        import pandas as pd

        keys = self._select(alpha, Re)
        frames = []
        for a, r in keys:
            d = self._data[(a, r)]
            n = len(next(iter(d.values())))
            frame = {
                "alpha": np.full(n, a),
                "Re": np.full(n, r),
            }
            frame.update(d)
            frames.append(pd.DataFrame(frame))
        if not frames:
            return pd.DataFrame()
        return pd.concat(
            frames, ignore_index=True
        )


class DumpData:
    r"""Boundary-layer dump data keyed by (alpha, Re).

    Stores XFOIL ``DUMP`` output: arc-length *s*, coordinates
    *x* and *y*, edge velocity ratio *Ue/Vinf*, displacement
    thickness *Dstar*, momentum thickness *Theta*,
    skin-friction coefficient *Cf*, and shape factor *H*.

    The shape factor is computed automatically on insertion:

    .. math::

        H = \frac{\delta^*}{\theta}

    Plotting supports Fix-\u03b1 / Fix-Re overlay like
    :class:`CpData`.

    Not intended to be instantiated directly; populated by
    :class:`XFoil` methods.

    Example::

        >>> dump = res.dump
        >>> dump(alpha=5.0, Re=1e6)["Dstar"]  # 1D ndarray
        >>> dump(Re=1e6)[["x", "Cf"]]         # {(a,Re): (N,2)}
        >>> dump["H"]                          # {(a,Re): ndarray}
        >>> dump[0]                            # dict for first entry
        >>> dump.columns                       # ['s','x','y',...,'H']
        >>> dump.to_dataframe(alpha=5.0)       # pandas DataFrame
    """

    def __init__(self):
        self._data = {}

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        if not self._data:
            return "DumpData(empty)"
        re = sorted({k[1] for k in self._data})
        na = len({k[0] for k in self._data})
        return (
            f"DumpData({len(self._data)} entries, "
            f"Re=[{_fmt_re(re)}], {na} \u03b1)"
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        """Access data by positional index or column name.

        Args:
            key (int): Positional index into sorted
                ``(alpha, Re)`` keys.
            key (str): Column name (e.g. ``'Dstar'``).
                Returns ndarray if one entry, else
                ``{(alpha, Re): ndarray}`` dict.
            key (list[str]): Multiple column names.
                Returns ``(N, cols)`` ndarray if one
                entry, else
                ``{(alpha, Re): (N, cols)}`` dict.

        Returns:
            dict or ndarray: Depends on key type.
        """
        if isinstance(key, (int, np.integer)):
            sorted_keys = sorted(self._data)
            return self._data[sorted_keys[key]]
        if isinstance(key, str):
            out = {
                k: self._data[k][key]
                for k in sorted(self._data)
            }
            if len(out) == 1:
                return next(iter(out.values()))
            return out
        if isinstance(key, list):
            out = {
                k: np.column_stack(
                    [self._data[k][c] for c in key]
                )
                for k in sorted(self._data)
            }
            if len(out) == 1:
                return next(iter(out.values()))
            return out
        raise TypeError(
            f"key must be int, str, or list[str], "
            f"got {type(key).__name__}"
        )

    def __contains__(self, item):
        """Check column name or (alpha, Re) key."""
        if isinstance(item, str):
            if not self._data:
                return False
            return item in next(
                iter(self._data.values())
            )
        return item in self._data

    def __iter__(self):
        return iter(sorted(self._data))

    def __call__(self, alpha=None, Re=None):
        """Filter by alpha and/or Re.

        Returns a shallow copy containing only matching
        entries (arrays are shared, not copied).

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).

        Returns:
            DumpData: Filtered subset.
        """
        new = DumpData()
        for k in self._select(alpha, Re):
            new._data[k] = self._data[k]
        return new

    def _add(self, alpha, Re, raw):
        r"""Store dump data for (alpha, Re).

        Values are converted to numpy arrays on insertion.
        Computes derived shape factor
        :math:`H = \delta^* / \theta` if both
        ``Dstar`` and ``Theta`` are present.
        """
        arrays = {
            k: np.asarray(v) for k, v in raw.items()
        }
        if "Dstar" in arrays and "Theta" in arrays:
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):
                arrays["H"] = np.where(
                    arrays["Theta"] > 0,
                    arrays["Dstar"] / arrays["Theta"],
                    np.nan,
                )
        self._data[(alpha, Re)] = arrays

    @property
    def alphas(self):
        """Sorted unique alpha values with data."""
        return sorted({k[0] for k in self._data})

    @property
    def reynolds(self):
        """Sorted unique Re values with data."""
        return sorted({k[1] for k in self._data})

    Re = reynolds

    @property
    def columns(self):
        """Column names in the data."""
        if not self._data:
            return []
        return list(
            next(iter(self._data.values())).keys()
        )

    @staticmethod
    def _to_set(val):
        """Normalise a filter arg to a set or None."""
        if val is None:
            return None
        if isinstance(val, (int, float, np.floating)):
            return {float(val)}
        return {float(v) for v in val}

    def _select(self, alpha=None, Re=None):
        """Return filtered ``(alpha, Re)`` keys.

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).

        Returns:
            list: Matching ``(alpha, Re)`` keys, sorted.
        """
        keys = list(self._data.keys())
        a_set = self._to_set(alpha)
        r_set = self._to_set(Re)
        if a_set is not None:
            keys = [
                k for k in keys if k[0] in a_set
            ]
        if r_set is not None:
            keys = [
                k for k in keys if k[1] in r_set
            ]
        return sorted(keys)

    @staticmethod
    def _label(a, r, unique_a, unique_r):
        """Build a legend label for one curve."""
        if len(unique_a) == 1:
            return f"Re={r:.2e}"
        if len(unique_r) == 1:
            return f"\u03b1={a:.1f}\u00b0"
        return (
            f"\u03b1={a:.1f}\u00b0, Re={r:.2e}"
        )

    def plot(
        self, alpha=None, Re=None, show=True
    ):
        r"""Plot boundary-layer quantities (2\u00d72 grid).

        Panels: \u03b4* (Dstar), \u03b8 (Theta), Cf, H
        vs x/c.

        Overlay logic matches :meth:`CpData.plot`:
        fix alpha to overlay Re, or fix Re to overlay alpha.

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).
            show (bool): Call ``plt.show()``.

        Returns:
            tuple: ``(fig, axes)``
        """
        import matplotlib.pyplot as plt

        keys = self._select(alpha, Re)
        if not keys:
            raise RuntimeError(
                "No dump data matching the given "
                "filters."
            )

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        panels = [
            ("Dstar", "\u03b4* (displacement thickness)"),
            ("Theta", "\u03b8 (momentum thickness)"),
            ("Cf", "Cf (skin friction)"),
            ("H", "H (shape factor)"),
        ]

        styles = _keyed_styles(keys)
        unique_a = sorted({k[0] for k in keys})
        unique_r = sorted({k[1] for k in keys})

        for a, r in keys:
            bl = self._data[(a, r)]
            lbl = self._label(a, r, unique_a, unique_r)
            x = np.asarray(bl.get("x", []))
            if len(x) == 0:
                continue
            for ax, (col, _) in zip(axes.flat, panels):
                if col in bl:
                    ax.plot(
                        x, np.asarray(bl[col]),
                        color=styles[(a, r)],
                        label=lbl,
                        marker=".", markersize=1,
                    )

        for ax, (col, title) in zip(
            axes.flat, panels
        ):
            ax.set_xlabel("x / c")
            ax.set_ylabel(col)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        if len(unique_a) == 1:
            sup = (
                f"Boundary layer \u2014 "
                f"\u03b1 = {unique_a[0]:.1f}\u00b0"
            )
        elif len(unique_r) == 1:
            sup = (
                f"Boundary layer \u2014 "
                f"Re = {unique_r[0]:.2e}"
            )
        else:
            sup = "Boundary layer"
        fig.suptitle(sup, fontsize=13)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_velocity(
        self, alpha=None, Re=None, ax=None, show=True
    ):
        """Plot edge-velocity distribution (Ue/Vinf vs x/c).

        Overlay logic matches :meth:`CpData.plot`.

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).
            ax: Optional Matplotlib Axes.
            show (bool): Call ``plt.show()``.

        Returns:
            tuple: ``(fig, ax)``
        """
        import matplotlib.pyplot as plt

        keys = self._select(alpha, Re)
        if not keys:
            raise RuntimeError(
                "No dump data matching the given "
                "filters."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        styles = _keyed_styles(keys)
        unique_a = sorted({k[0] for k in keys})
        unique_r = sorted({k[1] for k in keys})

        for a, r in keys:
            bl = self._data[(a, r)]
            lbl = self._label(a, r, unique_a, unique_r)
            # Try common column names for edge velocity
            vel_key = None
            for k in ("Ue/Vinf", "Ue"):
                if k in bl:
                    vel_key = k
                    break
            if vel_key is None or "x" not in bl:
                continue
            ax.plot(
                bl["x"], bl[vel_key],
                color=styles[(a, r)],
                label=lbl,
                marker=".", markersize=2,
            )

        ax.set_xlabel("x / c")
        ax.set_ylabel("Ue / V\u221e")
        if len(unique_a) == 1:
            title = (
                f"Velocity distribution \u2014 "
                f"\u03b1 = {unique_a[0]:.1f}\u00b0"
            )
        elif len(unique_r) == 1:
            title = (
                f"Velocity distribution \u2014 "
                f"Re = {unique_r[0]:.2e}"
            )
        else:
            title = "Velocity distribution"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # -- data export ------------------------------------------

    def to_dataframe(self, alpha=None, Re=None):
        r"""Export boundary-layer data as a pandas DataFrame.

        Returns a tidy DataFrame with one row per surface
        point.  ``alpha`` and ``Re`` columns identify each
        distribution.  Includes all stored fields
        (s, x, y, Ue/Vinf, Dstar, Theta, Cf, H).

        Args:
            alpha (float, array-like, or None):
                Filter by angle(s) of attack.
            Re (float, array-like, or None):
                Filter by reynolds number(s).

        Returns:
            pandas.DataFrame: Columns ``alpha``, ``Re``,
                plus all boundary-layer fields.

        Raises:
            ImportError: If pandas is not installed.

        Example::

            >>> df = res.dump.to_dataframe(alpha=5.0)
        """
        import pandas as pd

        keys = self._select(alpha, Re)
        frames = []
        for a, r in keys:
            d = self._data[(a, r)]
            n = len(next(iter(d.values())))
            frame = {
                "alpha": np.full(n, a),
                "Re": np.full(n, r),
            }
            frame.update(d)
            frames.append(pd.DataFrame(frame))
        if not frames:
            return pd.DataFrame()
        return pd.concat(
            frames, ignore_index=True
        )


class AeroResults:
    """Result container returned by :class:`XFoil` methods
    (and other solvers to be implemented).

    Holds polar, Cp, and boundary-layer dump data for a
    single airfoil at a fixed Mach number, potentially across
    multiple reynolds numbers. It can also hold a collection
    of per-airfoil :class:`AeroResults` objects. Sub-data is
    stored in
    :class:`PolarData`, :class:`CpData`, and
    :class:`DumpData` objects which have their own ``.plot()``
    methods.

    Scalar aerodynamic properties (``CL_max``, ``CD_min``,
    etc.) return a float when there is exactly one reynolds
    number, or a ``{Re: float}`` dict when there are
    several.  Interpolation shortcuts ``CL()``, ``CD()``,
    ``CM()`` follow the same convention.

    Args:
        label (str): Human-readable airfoil identifier (name).
        Mach (float): Freestream Mach number.
        source (str): Source of the aerodynamic data.

    Attributes:
        label (str): Airfoil name / identifier.
        Mach (float): Mach number.
        source (str): Source of the aerodynamic data.
        polar (PolarData): Polar sweep data.
        cp (CpData): Pressure distributions.
        dump (DumpData): Boundary-layer dump data.

    Example::

        >>> res = xf.analyze(foil, alphas, reynolds=1e6)
        >>> res.CL_max
        1.42
        >>> res.polar.plot()
        >>> res.CL(5.0)
        0.78
    """

    def __init__(self, label: str, Mach: float, source: str = "XFoil"):
        self.label = label
        self.Mach = Mach
        self.source = source
        self.polar = PolarData()
        self.cp = CpData()
        self.dump = DumpData()
        self._airfoils = {}

    @classmethod
    def from_airfoils(
        cls,
        results,
        label: str = "multi_airfoil",
        Mach: float | None = None,
        source: str | None = None,
    ):
        """Build a multi-airfoil container from child results.

        Args:
            results: Mapping ``{label: AeroResults}`` or iterable
                of :class:`AeroResults` objects.
            label: Label for the parent container.
            Mach: Optional parent Mach override. Defaults to first
                child Mach.
            source: Optional parent source override. Defaults to
                first child source.

        Returns:
            AeroResults: Multi-airfoil container.
        """
        if hasattr(results, "items"):
            items = list(results.items())
        else:
            items = [(None, r) for r in results]

        if len(items) == 0:
            raise ValueError("results must contain at least one airfoil")

        first = items[0][1]
        if not isinstance(first, AeroResults):
            raise TypeError("results entries must be AeroResults objects")

        parent = cls(
            label=label,
            Mach=first.Mach if Mach is None else Mach,
            source=first.source if source is None else source,
        )
        for key, child in items:
            parent.add_airfoil(child, label=key)
        return parent

    @property
    def is_multi_airfoil(self) -> bool:
        """Whether this object stores multiple airfoils."""
        return len(self._airfoils) > 0

    @property
    def airfoil_labels(self):
        """Ordered list of airfoil labels in a multi container."""
        return list(self._airfoils.keys())

    @property
    def airfoils(self):
        """Shallow copy of contained airfoil results by label."""
        return dict(self._airfoils)

    def add_airfoil(
        self,
        result: "AeroResults",
        label: str | None = None,
    ) -> str:
        """Add a single-airfoil result to this container.

        Args:
            result: Child result to store.
            label: Optional label override.

        Returns:
            str: Final stored label (deduplicated if needed).
        """
        if not isinstance(result, AeroResults):
            raise TypeError("result must be an AeroResults instance")
        if result.is_multi_airfoil:
            raise ValueError(
                "result must be single-airfoil; flatten before adding"
            )

        if self.polar or self.cp or self.dump:
            raise ValueError(
                "cannot add airfoils to an AeroResults object that already "
                "contains single-airfoil data"
            )

        base = str(result.label if label is None else label).strip()
        if base == "":
            base = "airfoil"

        final = base
        i = 2
        while final in self._airfoils:
            final = f"{base}_{i}"
            i += 1

        self._airfoils[final] = result
        return final

    def _resolve_airfoil_keys(self, airfoil=None):
        """Resolve airfoil selector(s) to stored labels."""
        if not self.is_multi_airfoil:
            if airfoil is not None:
                raise ValueError(
                    "airfoil selector is only valid for multi-airfoil results"
                )
            return []

        labels = self.airfoil_labels
        if airfoil is None:
            return labels

        def _one(val):
            if isinstance(val, str):
                if val not in self._airfoils:
                    raise KeyError(f"Unknown airfoil label: {val}")
                return val
            if isinstance(val, (int, np.integer)):
                return labels[int(val)]
            raise TypeError(
                "airfoil selector must be label, index, or iterable of those"
            )

        if isinstance(airfoil, (str, int, np.integer)):
            return [_one(airfoil)]

        out = []
        for item in airfoil:
            out.append(_one(item))
        return out

    def __call__(self, Re=None, airfoil=None):
        """Filter by reynolds and/or airfoil selector."""
        if self.is_multi_airfoil:
            keys = self._resolve_airfoil_keys(airfoil)
            if len(keys) == 1 and isinstance(
                airfoil, (str, int, np.integer)
            ):
                child = self._airfoils[keys[0]]
                return child(Re=Re) if Re is not None else child

            new = AeroResults(self.label, self.Mach, self.source)
            for key in keys:
                child = self._airfoils[key]
                new.add_airfoil(
                    child(Re=Re) if Re is not None else child,
                    label=key,
                )
            return new

        new = AeroResults(self.label, self.Mach, self.source)
        new.polar = self.polar(Re=Re)
        new.cp = self.cp(Re=Re)
        new.dump = self.dump(Re=Re)
        return new

    def __getitem__(self, idx):
        if self.is_multi_airfoil:
            if isinstance(idx, str):
                return self._airfoils[idx]
            if isinstance(idx, (int, np.integer)):
                key = self.airfoil_labels[int(idx)]
                return self._airfoils[key]
            raise TypeError(
                "multi-airfoil indexing expects label or integer index"
            )
        re = self.reynolds[idx]  # already sorted
        return self(Re=re)

    def __repr__(self):
        if self.is_multi_airfoil:
            labels = self.airfoil_labels
            preview = ", ".join(labels[:3])
            if len(labels) > 3:
                preview += ", ..."
            return (
                f"AeroResults('{self.label}', "
                f"M={self.Mach}, "
                f"airfoils={len(labels)} [{preview}])"
            )

        re = self.reynolds or []
        parts = []
        if self.polar:
            parts.append("polar")
        if self.cp:
            parts.append(f"cp({len(self.cp)})")
        if self.dump:
            parts.append(f"dump({len(self.dump)})")
        info = ", ".join(parts) if parts else "empty"
        return (
            f"AeroResults('{self.label}', "
            f"M={self.Mach}, "
            f"Re=[{_fmt_re(re)}], {info})"
        )

    @property
    def reynolds(self):
        """Sorted list of reynolds numbers with data."""
        if self.is_multi_airfoil:
            out = set()
            for child in self._airfoils.values():
                out.update(child.reynolds)
            return sorted(out)

        out = set()
        if self.polar:
            out.update(self.polar.reynolds)
        if self.cp:
            out.update(self.cp.reynolds)
        if self.dump:
            out.update(self.dump.reynolds)
        return sorted(out)

    Re = reynolds

    def __getattr__(self, name):
        """Delegate polar metrics to single or multi containers.

        For single-airfoil results this mirrors ``self.polar``.
        For multi-airfoil results, returns ``{label: value}`` for
        properties and ``{label: value}`` from callables.
        """
        if name.startswith("_") or not hasattr(self.polar, name):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute '{name}'"
            )

        if not self.is_multi_airfoil:
            return getattr(self.polar, name)

        sample = getattr(next(iter(self._airfoils.values())).polar, name)

        if callable(sample):

            def wrapped(*args, airfoil=None, **kwargs):
                keys = self._resolve_airfoil_keys(airfoil)
                return {
                    key: getattr(self._airfoils[key].polar, name)(
                        *args, **kwargs
                    )
                    for key in keys
                }

            return wrapped

        return {
            key: getattr(self._airfoils[key].polar, name)
            for key in self.airfoil_labels
        }

    def metric(
        self,
        name: str,
        *args,
        airfoil=None,
        as_dataframe: bool = False,
        **kwargs,
    ):
        """Evaluate a polar metric for one or many airfoils.

        Args:
            name: PolarData property or method name (e.g. ``CL_max``
                or ``CL``).
            *args: Positional args for method metrics.
            airfoil: Airfoil selector for multi-airfoil containers.
            as_dataframe: Export as tidy table with columns
                ``airfoil``, ``Re``, and metric value.
            **kwargs: Keyword args for method metrics.

        Returns:
            Scalar/dict result, or a pandas DataFrame when
            ``as_dataframe=True``.
        """
        if self.is_multi_airfoil:
            keys = self._resolve_airfoil_keys(airfoil)
            values = {}
            single_re = {}
            for key in keys:
                child = self._airfoils[key]
                attr = getattr(child.polar, name)
                value = attr(*args, **kwargs) if callable(attr) else attr
                values[key] = value

                if not isinstance(value, dict):
                    re_vals = child.reynolds
                    if len(re_vals) == 1:
                        single_re[key] = float(re_vals[0])

            if as_dataframe:
                return self._metric_dataframe(
                    values,
                    name,
                    single_re=single_re,
                )
            return values

        attr = getattr(self.polar, name)
        value = attr(*args, **kwargs) if callable(attr) else attr
        if as_dataframe:
            single_re = None
            if not isinstance(value, dict) and len(self.reynolds) == 1:
                single_re = {self.label: float(self.reynolds[0])}
            return self._metric_dataframe(
                {self.label: value},
                name,
                single_re=single_re,
            )
        return value

    @staticmethod
    def _metric_dataframe(
        values: dict,
        metric_name: str,
        single_re: dict | None = None,
    ):
        """Convert nested metric values to a tidy DataFrame."""
        import pandas as pd

        if single_re is None:
            single_re = {}

        rows = []
        for label, value in values.items():
            if isinstance(value, dict):
                for re_value, item in sorted(value.items()):
                    rows.append(
                        {
                            "airfoil": label,
                            "Re": float(re_value),
                            metric_name: item,
                        }
                    )
            else:
                rows.append(
                    {
                        "airfoil": label,
                        "Re": single_re.get(label, np.nan),
                        metric_name: value,
                    }
                )
        return pd.DataFrame(rows)

    def to_dataframe(self, Re=None, airfoil=None):
        """Export polar data as a pandas DataFrame.

        Convenience shortcut for
        ``self.polar.to_dataframe()``.

        For multi-airfoil containers, returns concatenated
        polar rows with an ``airfoil`` column.

        Returns:
            pandas.DataFrame: Polar data for all reynolds
                numbers.

        Raises:
            ImportError: If pandas is not installed.
            RuntimeError: If no polar data is available.

        Example::

            >>> df = res.to_dataframe()
        """
        if self.is_multi_airfoil:
            import pandas as pd

            frames = []
            for key in self._resolve_airfoil_keys(airfoil):
                child = self._airfoils[key]
                df = child(Re=Re).to_dataframe()
                if df.empty:
                    continue
                df.insert(0, "airfoil", key)
                frames.append(df)
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        if not self.polar:
            raise RuntimeError(
                f"No polar data for '{self.label}'. "
                "Run get_polar() or analyze() first."
            )
        return self.polar.to_dataframe(Re=Re)

    def to_polar_tensor(
        self,
        Re=None,
        airfoil=None,
        columns=None,
    ):
        """Export polar data as a dense tensor for training workflows.

        This method preserves the existing structured/indexed API while
        also exposing the complete polar grid as one array.

        Args:
            Re (float, list, or None): Optional Reynolds filter.
            airfoil: Optional airfoil selector for multi-airfoil results.
            columns (str, list[str], or None): Polar columns to include.
                ``None`` includes all non-alpha polar columns.

        Returns:
            dict: Tensor bundle with keys:
                - ``values``: ``[B, R, A, C]`` array
                - ``airfoils``: list of airfoil labels (len ``B``)
                - ``Re``: Reynolds axis ``[R]``
                - ``alpha``: alpha axis ``[A]``
                - ``columns``: column names (len ``C``)

        Raises:
            RuntimeError: If no polar data is available.
            ValueError: If selected airfoils do not share the same dense
                Reynolds/alpha grid.
        """
        if self.is_multi_airfoil:
            labels = self._resolve_airfoil_keys(airfoil)
            if len(labels) == 0:
                raise RuntimeError("No airfoils selected for tensor export.")

            stacked = []
            re_axis = None
            alpha_axis = None
            resolved_columns = None

            for label_key in labels:
                child = self._airfoils[label_key]
                if not child.polar:
                    raise RuntimeError(
                        f"No polar data for airfoil '{label_key}'."
                    )

                values, child_re, child_alpha, child_columns = child.polar.to_tensor(
                    Re=Re,
                    columns=columns,
                )

                if re_axis is None:
                    re_axis = child_re
                    alpha_axis = child_alpha
                    resolved_columns = child_columns
                else:
                    if (
                        child_re.shape != re_axis.shape
                        or not np.allclose(child_re, re_axis, equal_nan=True)
                    ):
                        raise ValueError(
                            "Cannot build multi-airfoil tensor: Reynolds grids differ."
                        )
                    if (
                        child_alpha.shape != alpha_axis.shape
                        or not np.allclose(child_alpha, alpha_axis, equal_nan=True)
                    ):
                        raise ValueError(
                            "Cannot build multi-airfoil tensor: alpha grids differ."
                        )
                    if child_columns != resolved_columns:
                        raise ValueError(
                            "Cannot build multi-airfoil tensor: column sets differ."
                        )

                stacked.append(values)

            return {
                "values": np.stack(stacked, axis=0),
                "airfoils": labels,
                "Re": re_axis,
                "alpha": alpha_axis,
                "columns": resolved_columns,
            }

        if not self.polar:
            raise RuntimeError(
                f"No polar data for '{self.label}'. "
                "Run get_polar() or analyze() first."
            )

        values, re_axis, alpha_axis, resolved_columns = self.polar.to_tensor(
            Re=Re,
            columns=columns,
        )
        return {
            "values": values[None, ...],
            "airfoils": [self.label],
            "Re": re_axis,
            "alpha": alpha_axis,
            "columns": resolved_columns,
        }
