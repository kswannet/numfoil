from functools import cached_property
import os
import re
import shutil
import subprocess as sp
import time

import numpy as np

from numfoil.aero.aeroresults import (
    AeroResults,
    PolarData,
    CpData,
    DumpData,
)


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_STDOUT_ALPHA_CL_RE = re.compile(
    rf"a\s*=\s*({_FLOAT_RE}).*?CL\s*=\s*({_FLOAT_RE})"
)
_STDOUT_CM_CD_RE = re.compile(
    rf"Cm\s*=\s*({_FLOAT_RE}).*?CD\s*=\s*({_FLOAT_RE})(?:.*?CDp\s*=\s*({_FLOAT_RE}))?"
)


def _as_Re_list(reynolds):
    """Normalise a reynolds input to a list of floats.

    Args:
        reynolds: Scalar or iterable of reynolds numbers.

    Returns:
        list: List of float reynolds values.
    """
    if isinstance(reynolds, (int, float, np.floating)):
        return [float(reynolds)]
    return [float(r) for r in reynolds]


def _get_xfoil_timeout_seconds(default: float = 120.0) -> float | None:
    """Return the XFOIL process timeout in seconds.

    Read from ``NUMFOIL_XFOIL_TIMEOUT_SECONDS`` when present. Values of
    ``0``, ``none``, ``off``, or ``false`` disable the timeout.
    """
    raw = os.getenv("NUMFOIL_XFOIL_TIMEOUT_SECONDS", "").strip().lower()
    if raw == "":
        return default
    if raw in {"0", "none", "off", "false"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return default
    return None if value <= 0 else value


def _get_xfoil_debug_dir() -> str | None:
    """Return optional directory for XFOIL debug artifacts.

    When ``NUMFOIL_XFOIL_DEBUG_DIR`` is set, command input and captured process
    logs are written there for each XFOIL run.
    """
    raw = os.getenv("NUMFOIL_XFOIL_DEBUG_DIR", "").strip()
    if raw == "":
        return None
    os.makedirs(raw, exist_ok=True)
    return raw


# Following class is based on original code by Pedro Leal, adapted for use in
# numfoil. Original code subject to the following License:

# MIT License
#
# Copyright (c) 2020 Pedro Leal
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# Created on Mar  9 14:58:25 2014
# Last update Jul 20 16:26:40 2015
# @author: Pedro Leal


class XFoil:
    """Solver-centric XFOIL interface.

    A self-contained wrapper around the XFOIL panel-method
    executable.  The solver is instantiated once with global
    defaults (iteration count, geometry options).  Mach number
    is specified per analysis call, not at construction time.

    Airfoils are accepted as:

    * A ``(N, 2)`` NumPy array of (x, y) coordinates in
      **Selig format** (upper surface TE\u2192LE, then lower
      surface LE\u2192TE).
    * Any object with a ``.points`` attribute that returns
      such an array (e.g. ``numfoil.geometry.BsplineAirfoil``).

    Each public method returns an :class:`AeroResults` that
    owns :class:`PolarData`, :class:`CpData`, and
    :class:`DumpData` sub-objects with their own ``.plot()``
    methods.

    Typical workflow::

        >>> xf = XFoil()
        >>> alphas = np.arange(-5, 15, 0.5)

        >>> res = xf.analyze(
        >>>     my_airfoil, alphas,
        >>>     reynolds=[1e6, 5e5], Mach=0.2,
        >>> )

        >>> res.summary()
        >>> res.polar.plot()
        >>> res.cp.plot(alpha=5.0)
        >>> res.dump.plot_velocity(Re=1e6)

        >>> print(res.CL_max)
        >>> print(res.CL(5.0))

    Args:
        iter (int): Max convergence iterations per operating
            point.
        GDES (bool): Run XFOIL geometry-design improvement.
        PANE (bool): Re-panel geometry (needed for > 495 pts).
        NORM (bool): Normalize coordinates to unit chord.
    """

    _XFOIL_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(
        self,
        iter=100,
        GDES=False,
        PANE=False,
        NORM=True,
    ):
        self.iter = iter
        self.GDES = GDES
        self.PANE = PANE
        self.NORM = NORM

    def __repr__(self):
        return (
            f"XFoil(iter={self.iter}, "
            f"GDES={self.GDES}, PANE={self.PANE})"
        )

    @staticmethod
    def _progress(Re_list, n_alpha):
        """Wrap *Re_list* with a progress indicator.

        Prints the total number of XFOIL sessions up front,
        then logs each session with elapsed time.

        Args:
            Re_list (list): reynolds numbers to iterate over.
            n_alpha (int): Number of alphas per session
                (shown for context only).

        Yields:
            float: Each reynolds number in *Re_list*.
        """
        n = len(Re_list)
        print(
            f"XFOIL: {n} session"
            f"{'s' if n != 1 else ''}"
            f", {n_alpha} alpha each"
        )
        t0 = time.perf_counter()
        for i, Re in enumerate(Re_list, 1):
            print(
                f"  [{i}/{n}] Re = {Re:.2e} ...",
                end="",
                flush=True,
            )
            yield Re
            dt = time.perf_counter() - t0
            print(f" done  ({dt:.1f}s elapsed)")

    @staticmethod
    def _get_executable():
        """Locate the XFOIL executable bundled alongside
        this module.

        Returns:
            str: Full path to the xfoil executable.
        """
        override = os.getenv("NUMFOIL_XFOIL_EXECUTABLE", "").strip()
        if override:
            resolved = shutil.which(override) if os.path.basename(override) == override else override
            if resolved and os.path.exists(resolved):
                return str(resolved)
            raise FileNotFoundError(
                "NUMFOIL_XFOIL_EXECUTABLE is set but does not resolve to an existing executable: "
                f"{override}"
            )

        d = XFoil._XFOIL_DIR
        if os.name == "nt":
            for name in os.listdir(d):
                if "xfoil" in name.lower() and name.endswith(
                    ".exe"
                ):
                    return os.path.join(d, name)

            system_xfoil = shutil.which("xfoil")
            if system_xfoil is not None:
                return system_xfoil

            raise FileNotFoundError(
                f"No xfoil executable found in {d} or on PATH"
            )

        system_xfoil = shutil.which("xfoil")
        if system_xfoil is not None:
            return system_xfoil

        raise FileNotFoundError(
            "No xfoil executable found on PATH for this platform. "
            "Install xfoil (for example `apt install xfoil`) or set "
            "NUMFOIL_XFOIL_EXECUTABLE to the desired binary path."
        )

    @staticmethod
    def _get_environment(executable: str | None = None):
        """Build an environment dict with xfoil's directory
        on PATH.

        Returns:
            dict: Copy of ``os.environ`` with the xfoil bin
                dir appended.
        """
        env = os.environ.copy()
        path_entries = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []

        extra_dirs: list[str] = []
        if os.name == "nt":
            extra_dirs.append(XFoil._XFOIL_DIR)
        if executable:
            extra_dirs.append(os.path.dirname(os.path.abspath(executable)))

        for extra_dir in extra_dirs:
            if extra_dir and extra_dir not in path_entries:
                path_entries.append(extra_dir)

        env["PATH"] = os.pathsep.join(path_entries)
        return env

    @staticmethod
    def _resolve_airfoil(airfoil):
        """Resolve an airfoil input to a (label, points)
        tuple.

        Accepts a ``(N, 2)`` NumPy array in Selig format or
        any object with a ``.points`` property (e.g. a
        numfoil ``BsplineAirfoil``).  If the object has a
        ``.name`` attribute it is used as the label;
        otherwise a hash-based label is generated.

        Args:
            airfoil: Coordinate array or airfoil object.

        Returns:
            tuple: ``(label, points)`` where *points* is a
                ``(N, 2)`` ndarray.

        Raises:
            TypeError: If the input is not recognised.
        """
        if hasattr(airfoil, "points"):
            pts = np.asarray(
                airfoil.points
            ).reshape(-1, 2)
            label = getattr(
                airfoil, "name", None
            ) or getattr(
                airfoil, "description", None
            )
            if not label:
                label = (
                    f"airfoil_"
                    f"{hash(pts.tobytes()) % 0xFFFF:04x}"
                )
            return label.strip().replace(" ", "_"), pts
        pts = np.asarray(airfoil, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise TypeError(
                "airfoil must be an (N, 2) coordinate "
                "array or an object with a .points "
                "property."
            )
        label = (
            f"airfoil_"
            f"{hash(pts.tobytes()) % 0xFFFF:04x}"
        )
        return label, pts

    @staticmethod
    def _write_dat(points, filepath):
        """Write Selig-format coordinates to an XFOIL .dat
        file.

        Args:
            points (ndarray): (N, 2) coordinate array.
            filepath (str): Destination file path.
        """
        pts = np.asarray(points).reshape(-1, 2)
        with open(filepath, "w") as f:
            f.write("airfoil\n")
            for x, y in pts:
                f.write(f"  {x: .7f}  {y: .7f}\n")

    @staticmethod
    def _polar_point_count(raw):
        """Return the number of polar operating points in a parsed output."""
        if not raw:
            return 0
        return int(np.asarray(raw.get("alpha", [])).size)

    @staticmethod
    def _parse_polar_stdout(stdout_text):
        """Recover converged polar points from XFOIL stdout.

        Some Linux XFOIL builds compute the operating point correctly but
        crash before flushing the accumulated polar file. In that case the
        final converged coefficients are still present in stdout.
        """
        if not stdout_text:
            return {}

        points = []
        current = None

        def _flush_current():
            nonlocal current
            if current is None:
                return
            if {"alpha", "CL", "CD"}.issubset(current):
                points.append(current)
            current = None

        for line in stdout_text.splitlines():
            match = _STDOUT_ALPHA_CL_RE.search(line)
            if match:
                alpha = float(match.group(1))
                cl = float(match.group(2))

                if (
                    current is not None
                    and "alpha" in current
                    and not np.isclose(
                        alpha,
                        current["alpha"],
                        atol=1e-9,
                        rtol=0.0,
                    )
                ):
                    _flush_current()

                if current is None:
                    current = {}
                current["alpha"] = alpha
                current["CL"] = cl
                continue

            match = _STDOUT_CM_CD_RE.search(line)
            if match and current is not None:
                current["CM"] = float(match.group(1))
                current["CD"] = float(match.group(2))
                if match.group(3) is not None:
                    current["CDp"] = float(match.group(3))

        _flush_current()
        if not points:
            return {}

        return {
            "alpha": [point["alpha"] for point in points],
            "CL": [point["CL"] for point in points],
            "CD": [point["CD"] for point in points],
            "CM": [point.get("CM", float("nan")) for point in points],
            "CDp": [point.get("CDp", float("nan")) for point in points],
        }

    def _run_xfoil(
        self,
        dat_path,
        alphas,
        reynolds,
        Mach=0,
        polar=False,
        cp=False,
        dump=False,
        flap=None,
        timeout_seconds=None,
    ):
        """Run a single XFOIL session for one reynolds number.

        Supports simultaneous polar accumulation and per-alpha
        Cp / Dump output in the same session, minimizing the
        number of XFOIL process invocations.

        Args:
            dat_path (str): Path to the .dat coordinate file.
            alphas (list of float): Angles of attack.
            reynolds (float): reynolds number.
            Mach (float): Freestream Mach number.
            polar (bool): Accumulate polar data.
            cp (bool): Write Cp per alpha.
            dump (bool): Write boundary-layer dump per alpha.
            flap (list or None):
                ``[x_hinge, y_hinge, defl_deg]``.

        Returns:
            dict: ``{"polar": path_or_None,
            "cp": {alpha: path}, "dump": {alpha: path}}``
        """
        if timeout_seconds is None:
            timeout_seconds = _get_xfoil_timeout_seconds()
        elif timeout_seconds <= 0:
            timeout_seconds = None

        exe = self._get_executable()
        env = self._get_environment(executable=exe)
        debug_dir = _get_xfoil_debug_dir()

        dat_basename = os.path.basename(dat_path)
        tag = os.path.splitext(dat_basename)[0]
        re_tag = f"{float(reynolds):.3e}".replace("+", "").replace("-", "m").replace(".", "p")
        debug_prefix = (
            os.path.join(debug_dir, f"{tag}_Re{re_tag}_{int(time.time() * 1000)}")
            if debug_dir is not None
            else None
        )

        # Generate output basenames
        polar_bn = (
            f"_xf_P_{tag}.txt" if polar else None
        )
        cp_bns = {}
        dump_bns = {}
        if cp:
            for i, a in enumerate(alphas):
                cp_bns[a] = f"_xf_C_{tag}_{i:03d}.txt"
        if dump:
            for i, a in enumerate(alphas):
                dump_bns[a] = f"_xf_D_{tag}_{i:03d}.txt"

        # Remove stale files
        all_bns = list(cp_bns.values()) + list(
            dump_bns.values()
        )
        if polar_bn:
            all_bns.append(polar_bn)
        for bn in all_bns:
            fp = os.path.join(self._XFOIL_DIR, bn)
            if os.path.isfile(fp):
                os.remove(fp)

        # -- build command sequence --
        cmds: list[str] = []
        if os.name != "nt":
            cmds.extend(["PLOP", "G F", ""])

        cmds.extend([f"load {dat_basename}", ""])

        if self.NORM:
            cmds.append("NORM")
        if self.PANE:
            cmds.append("PANE")
        if self.GDES:
            cmds += [
                "GDES", "CADD",
                "", "", "", "",
                "PANEL",
            ]
        if flap is not None:
            cmds += [
                "GDES", "FLAP",
                f"{flap[0]:f}",
                f"{flap[1]:f}",
                f"{flap[2]:f}",
                "eXec", "",
            ]

        cmds.append("OPER")
        cmds.append("iter")
        cmds.append(f"{self.iter:d}")

        if reynolds > 0:
            cmds.append("v")
            cmds.append(f"{reynolds:f}")
        if Mach > 0:
            cmds.append(f"MACH {Mach}")

        if polar:
            cmds.append("PACC")
            cmds.append(polar_bn)
            cmds.append("")

        for a in alphas:
            cmds.append(f"ALFA {a:.4f}")
            if cp and a in cp_bns:
                cmds.append(f"CPWR {cp_bns[a]}")
            if dump and a in dump_bns:
                cmds.append(f"DUMP {dump_bns[a]}")

        if polar:
            cmds.append("PACC")

        cmds.append("")
        cmds.append("QUIT")

        stdin_text = "\n".join(cmds) + "\n"

        if debug_prefix is not None:
            with open(f"{debug_prefix}.stdin.txt", "w", encoding="utf8", errors="replace") as f:
                f.write(stdin_text)

        startupinfo = None
        if hasattr(sp, "STARTUPINFO") and hasattr(sp, "STARTF_USESHOWWINDOW"):
            startupinfo = sp.STARTUPINFO()
            startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW
        capture_logs = debug_prefix is not None or polar
        popen_kwargs = dict(
            args=[exe],
            executable=exe,
            stdin=sp.PIPE,
            stdout=sp.PIPE if capture_logs else sp.DEVNULL,
            stderr=sp.PIPE if capture_logs else sp.DEVNULL,
            env=env,
            cwd=self._XFOIL_DIR,
            encoding="utf8",
        )
        if startupinfo is not None:
            popen_kwargs["startupinfo"] = startupinfo

        ps = sp.Popen(**popen_kwargs)

        def _write_debug_outputs(status: str, stdout_text: str | None, stderr_text: str | None) -> None:
            if debug_prefix is None:
                return
            with open(f"{debug_prefix}.stdout.txt", "w", encoding="utf8", errors="replace") as f:
                f.write("" if stdout_text is None else stdout_text)
            with open(f"{debug_prefix}.stderr.txt", "w", encoding="utf8", errors="replace") as f:
                f.write("" if stderr_text is None else stderr_text)
            with open(f"{debug_prefix}.meta.txt", "w", encoding="utf8", errors="replace") as f:
                f.write(f"status={status}\n")
                f.write(f"returncode={ps.returncode}\n")
                f.write(f"timeout_seconds={timeout_seconds}\n")
                f.write(f"reynolds={reynolds}\n")
                f.write(f"n_alpha={len(alphas)}\n")
                f.write(f"polar={polar} cp={cp} dump={dump}\n")
                f.write(f"executable={exe}\n")
                f.write(f"cwd={self._XFOIL_DIR}\n")

        stdout_text = None
        stderr_text = None
        try:
            if timeout_seconds is None:
                stdout_text, stderr_text = ps.communicate(stdin_text)
            else:
                stdout_text, stderr_text = ps.communicate(
                    stdin_text, timeout=timeout_seconds
                )
        except sp.TimeoutExpired as exc:
            ps.kill()
            stdout_text, stderr_text = ps.communicate()
            _write_debug_outputs("timeout", stdout_text, stderr_text)
            raise TimeoutError(
                "XFOIL session timed out after "
                f"{timeout_seconds:.1f}s "
                f"(Re={reynolds:.2e}, n_alpha={len(alphas)}, "
                f"polar={polar}, cp={cp}, dump={dump})."
                + (
                    f" Debug logs written to '{debug_prefix}.*'."
                    if debug_prefix is not None
                    else ""
                )
            ) from exc

        _write_debug_outputs("ok", stdout_text, stderr_text)

        # Assemble full paths
        result = {"polar": None, "cp": {}, "dump": {}}
        if polar_bn:
            result["polar"] = os.path.join(
                self._XFOIL_DIR, polar_bn
            )
        result["stdout"] = stdout_text
        result["stderr"] = stderr_text
        result["returncode"] = ps.returncode
        for a, bn in cp_bns.items():
            result["cp"][a] = os.path.join(
                self._XFOIL_DIR, bn
            )
        for a, bn in dump_bns.items():
            result["dump"][a] = os.path.join(
                self._XFOIL_DIR, bn
            )
        return result

    @staticmethod
    def _read_output(filepath, output_type, delete=True):
        """Parse an XFOIL output file into a dict of lists.

        Handles Polar, Cp, and Dump file formats.  Lines
        containing ``*********`` (XFOIL overflow indicators)
        are converted to ``nan``.  Negative numbers that are
        concatenated without whitespace separators (a common
        XFOIL formatting quirk) are split correctly.

        Args:
            filepath (str): Path to the XFOIL output file.
            output_type (str): ``'Polar'``, ``'Cp'``, or
                ``'Dump'``.
            delete (bool): Remove the file after reading.

        Returns:
            dict: Column-name \u2192 list-of-float mapping.
                Empty dict if the file does not exist.
        """
        if not os.path.isfile(filepath):
            return {}

        rows_to_skip = {
            "Polar": 10,
            "Cp": 2,
            "Dump": 0,
        }.get(output_type, 0)

        data = {}
        header = None
        count_skip = 0

        with open(filepath, "r") as f:
            for line in f:
                if count_skip < rows_to_skip:
                    count_skip += 1
                    continue

                line = (
                    line.replace("\t", " ")
                    .replace("\n", "")
                    .replace("#", " ")
                    .replace("*********", " nan")
                    .replace("---------", "")
                    .replace("--------", "")
                    .replace("-------", "")
                    .replace("------", "")
                    .replace("-", " -")
                )
                parts = line.split()
                if not parts:
                    continue

                if header is None:
                    header = parts
                    for h in header:
                        data[h] = []
                    continue

                for j, h in enumerate(header):
                    if j < len(parts):
                        try:
                            data[h].append(
                                float(parts[j])
                            )
                        except ValueError:
                            data[h].append(
                                float("nan")
                            )

        if delete and os.path.isfile(filepath):
            os.remove(filepath)
        return data

    # ==========================================================
    #                       PUBLIC API
    # ==========================================================

    def get_polar(
        self,
        airfoil,
        alphas,
        reynolds=0,
        Mach=0,
        flap=None,
        label=None,
    ):
        """Run a polar sweep and return the result.

        Runs XFOIL once per reynolds number, sweeping all
        supplied angles of attack.  Only polar data is
        collected (no Cp or boundary-layer dump).

        Args:
            airfoil: ``(N, 2)`` coordinate array in Selig
                format, or an object with a ``.points``
                property.
            alphas (array-like): Angles of attack in degrees.
            reynolds (float or list): reynolds number(s).
            Mach (float): Freestream Mach number.
            flap (list or None):
                ``[x_hinge, y_hinge, defl_deg]``.
            label (str or None): Override the auto-detected
                airfoil label.

        Returns:
            AeroResults: Result with ``polar`` populated.

        Example::

            >>> xf = XFoil()
            >>> res = xf.get_polar(
            ...     foil, np.arange(-5, 15, 0.5),
            ...     reynolds=1e6,
            ... )
            >>> res.polar.plot()
        """
        auto_label, pts = self._resolve_airfoil(airfoil)
        if label is not None:
            auto_label = label

        Re_list = _as_Re_list(reynolds)
        alphas = [float(a) for a in alphas]
        result = AeroResults(auto_label, Mach, source="XFoil_polar")

        dat_path = os.path.join(
            self._XFOIL_DIR,
            f"_xf_{auto_label}.dat",
        )
        self._write_dat(pts, dat_path)

        try:
            for Re in self._progress(
                Re_list, len(alphas)
            ):
                paths = self._run_xfoil(
                    dat_path, alphas, Re, Mach,
                    polar=True, flap=flap,
                )
                raw = self._read_output(
                    paths["polar"], "Polar"
                )
                if self._polar_point_count(raw) < len(alphas):
                    recovered = self._parse_polar_stdout(
                        paths.get("stdout")
                    )
                    if self._polar_point_count(recovered) < len(alphas):
                        recovered_points = []
                        for alpha in alphas:
                            single_paths = self._run_xfoil(
                                dat_path,
                                [alpha],
                                Re,
                                Mach,
                                polar=True,
                                flap=flap,
                            )
                            single_raw = self._read_output(
                                single_paths["polar"], "Polar"
                            )
                            if self._polar_point_count(single_raw) == 0:
                                single_raw = self._parse_polar_stdout(
                                    single_paths.get("stdout")
                                )
                            if self._polar_point_count(single_raw) > 0:
                                recovered_points.append(single_raw)

                        if recovered_points:
                            merged = {}
                            keys = []
                            for point in recovered_points:
                                for key in point:
                                    if key not in merged:
                                        merged[key] = []
                                        keys.append(key)

                            for point in recovered_points:
                                point_count = self._polar_point_count(point)
                                if point_count == 0:
                                    continue
                                for key in keys:
                                    values = point.get(key)
                                    if values is None:
                                        merged[key].extend(
                                            [float("nan")] * point_count
                                        )
                                        continue
                                    merged[key].extend(
                                        np.asarray(values).tolist()
                                    )

                            if self._polar_point_count(merged) > 0:
                                order = np.argsort(
                                    np.asarray(merged["alpha"])
                                )
                                recovered = {
                                    key: np.asarray(values)[order].tolist()
                                    for key, values in merged.items()
                                }

                    if self._polar_point_count(recovered) > 0:
                        raw = recovered
                if raw:
                    result.polar._add(Re, raw)
        finally:
            if os.path.isfile(dat_path):
                os.remove(dat_path)

        return result

    def get_cp(
        self,
        airfoil,
        alphas,
        reynolds=0,
        Mach=0,
        flap=None,
        label=None,
    ):
        """Compute Cp distributions and return the result.

        Runs one XFOIL session per reynolds number.  Within
        each session every alpha is evaluated sequentially
        and a Cp file is written.

        Args:
            airfoil: Coordinate array or airfoil object.
            alphas (array-like): Angles of attack in degrees.
            reynolds (float or list): reynolds number(s).
            Mach (float): Freestream Mach number.
            flap (list or None): Flap definition.
            label (str or None): Override label.

        Returns:
            AeroResults: Result with ``cp`` populated.

        Example::

            >>> res = xf.get_cp(foil, [0, 5, 10], Re=1e6)
            >>> res.cp.plot(alpha=5.0)
        """
        auto_label, pts = self._resolve_airfoil(airfoil)
        if label is not None:
            auto_label = label

        Re_list = _as_Re_list(reynolds)
        alphas = [float(a) for a in alphas]
        result = AeroResults(auto_label, Mach, source="XFoil_cp")

        dat_path = os.path.join(
            self._XFOIL_DIR,
            f"_xf_{auto_label}.dat",
        )
        self._write_dat(pts, dat_path)

        try:
            for Re in self._progress(
                Re_list, len(alphas)
            ):
                paths = self._run_xfoil(
                    dat_path, alphas, Re, Mach,
                    cp=True, flap=flap,
                )
                for a, fp in paths["cp"].items():
                    raw = self._read_output(fp, "Cp")
                    if raw:
                        result.cp._add(a, Re, raw)
        finally:
            if os.path.isfile(dat_path):
                os.remove(dat_path)

        return result

    def get_dump(
        self,
        airfoil,
        alphas,
        reynolds=0,
        Mach=0,
        flap=None,
        label=None,
    ):
        r"""Compute boundary-layer dump and return the result.

        Runs one XFOIL session per reynolds number.  Within
        each session every alpha is evaluated and a DUMP file
        is written.  The shape factor
        :math:`H = \delta^* / \theta` is computed
        automatically.

        Args:
            airfoil: Coordinate array or airfoil object.
            alphas (array-like): Angles of attack in degrees.
            reynolds (float or list): reynolds number(s).
            Mach (float): Freestream Mach number.
            flap (list or None): Flap definition.
            label (str or None): Override label.

        Returns:
            AeroResults: Result with ``dump`` populated.

        Example::

            >>> res = xf.get_dump(
            ...     foil, [0, 5, 10], Re=1e6
            ... )
            >>> res.dump.plot(alpha=5.0)
        """
        auto_label, pts = self._resolve_airfoil(airfoil)
        if label is not None:
            auto_label = label

        Re_list = _as_Re_list(reynolds)
        alphas = [float(a) for a in alphas]
        result = AeroResults(auto_label, Mach, source="XFoil_dump")

        dat_path = os.path.join(
            self._XFOIL_DIR,
            f"_xf_{auto_label}.dat",
        )
        self._write_dat(pts, dat_path)

        try:
            for Re in self._progress(
                Re_list, len(alphas)
            ):
                paths = self._run_xfoil(
                    dat_path, alphas, Re, Mach,
                    dump=True, flap=flap,
                )
                for a, fp in paths["dump"].items():
                    raw = self._read_output(
                        fp, "Dump"
                    )
                    if raw:
                        result.dump._add(a, Re, raw)
        finally:
            if os.path.isfile(dat_path):
                os.remove(dat_path)

        return result

    def analyze(
        self,
        airfoil: np.ndarray | object,
        alphas: np.ndarray = np.arange(-5, 15, 0.5),
        reynolds: float | np.ndarray = 0,
        Mach: float = 0,
        flap: list | None = None,
        label: str | None = None,
    ):
        """Full analysis: polar + Cp + boundary-layer dump.

        Runs one efficient XFOIL session per reynolds number.
        Within each session, polar accumulation is active and
        after each alpha the Cp and Dump files are written.
        This is much faster than calling :meth:`get_polar`,
        :meth:`get_cp`, and :meth:`get_dump` separately
        because it avoids redundant XFOIL startups and
        geometry loading.

        Args:
            airfoil: ``(N, 2)`` coordinate array in Selig
                format, or an object with a ``.points``
                property.
            alphas (array-like): Angles of attack in degrees.
                Used for all three data types.
            reynolds (float or list): reynolds number(s).
            Mach (float): Freestream Mach number.
            flap (list or None):
                ``[x_hinge, y_hinge, defl_deg]``.
            label (str or None): Override the auto-detected
                airfoil label.

        Returns:
            AeroResults: Result with ``polar``, ``cp``,
                and ``dump`` all populated.

        Example::

            >>> xf = XFoil()
            >>> res = xf.analyze(
            ...     foil, np.arange(-5, 15, 0.5),
            ...     reynolds=[1e6, 5e5], Mach=0.2,
            ... )
            >>> res.summary()
            >>> res.polar.plot()
            >>> res.cp.plot(alpha=5.0)
            >>> res.dump.plot_velocity(Re=1e6)
            >>> print(res.CL_max)
            >>> print(res.CL(5.0))
        """
        auto_label, pts = self._resolve_airfoil(airfoil)
        if label is not None:
            auto_label = label

        Re_list = _as_Re_list(reynolds)
        alphas = [float(a) for a in alphas]
        result = AeroResults(auto_label, Mach, source="XFoil")

        dat_path = os.path.join(
            self._XFOIL_DIR,
            f"_xf_{auto_label}.dat",
        )
        self._write_dat(pts, dat_path)

        try:
            for Re in self._progress(
                Re_list, len(alphas)
            ):
                paths = self._run_xfoil(
                    dat_path, alphas, Re, Mach,
                    polar=True, cp=True, dump=True,
                    flap=flap,
                )

                # Polar
                raw = self._read_output(
                    paths["polar"], "Polar"
                )
                if raw:
                    result.polar._add(Re, raw)

                # Cp per alpha
                for a, fp in paths["cp"].items():
                    raw = self._read_output(fp, "Cp")
                    if raw:
                        result.cp._add(a, Re, raw)

                # Dump per alpha
                for a, fp in paths["dump"].items():
                    raw = self._read_output(
                        fp, "Dump"
                    )
                    if raw:
                        result.dump._add(a, Re, raw)
        finally:
            if os.path.isfile(dat_path):
                os.remove(dat_path)

        return result


# ============================================================
#                      EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # -- 1. Load airfoil coordinates -------------------------
    #
    # Option A: from a numfoil BsplineAirfoil
    #   from numfoil.geometry.airfoil import BsplineAirfoil
    #   foil = BsplineAirfoil("naca4412")
    #   coords = foil.points       # (199, 2) Selig ndarray
    #
    # Option B: from a .dat file on disk
    #   coords = np.loadtxt("my_airfoil.dat", skiprows=1)
    #
    # For this demo we read two UIUC .dat files directly.

    _repo = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(
            os.path.join(_repo, "UIUC_airfoils")
        ):
            break
        _repo = os.path.dirname(_repo)

    naca2412 = np.loadtxt(
        os.path.join(
            _repo,
            "UIUC_airfoils",
            "bezier_closed",
            "naca2412.dat",
        )
    )
    a18 = np.loadtxt(
        os.path.join(
            _repo,
            "UIUC_airfoils",
            "bezier_closed",
            "a18.dat",
        )
    )

    # -- 2. Create the solver --------------------------------
    xf = XFoil(iter=100)

    # -- 3. Full analysis (polar + Cp + BL) ------------------
    alphas = np.arange(-5, 15, 0.5)

    # NACA 2412 at two reynolds numbers
    res_naca = xf.analyze(
        naca2412, alphas,
        reynolds=[1e6, 5e5],
        label="NACA2412",
    )

    # A18 at one reynolds number
    res_a18 = xf.analyze(
        a18, alphas,
        reynolds=1e6,
        label="A18",
    )

    # -- 4. Inspect scalar properties ------------------------
    print("\nNACA 2412 (two Re):")
    print(f"  CL_max      = {res_naca.CL_max}")
    print(f"  \u03b1 @ CL_max  = {res_naca.alpha_CL_max}")
    print(f"  (L/D)_max   = {res_naca.LD_max}")
    print(f"  CD_min      = {res_naca.CD_min}")

    print(f"\nA18 @ Re=1e6:")
    print(f"  CL_max      = {res_a18.CL_max:.4f}")
    print(f"  CL(5.0)     = {res_a18.CL(5.0):.4f}")
    print(f"  CD(5.0)     = {res_a18.CD(5.0):.6f}")

    # -- 5. Summary table ------------------------------------
    print("\n")
    res_naca.summary()
    res_a18.summary()

    # -- 6. Polar plots --------------------------------------
    # All Re overlaid automatically
    res_naca.polar.plot()

    # Filter to one Re
    res_naca.polar.plot(Re=1e6)

    # Drag polar
    res_naca.polar.plot_drag()

    # L/D
    res_naca.polar.plot_LD()

    # -- 7. Cp plots -----------------------------------------
    # Fix alpha, overlay Re
    res_naca.cp.plot(alpha=5.0)

    # Fix Re, overlay alpha
    res_naca.cp.plot(Re=1e6)

    # -- 8. Boundary-layer plots -----------------------------
    res_naca.dump.plot(alpha=5.0)
    res_naca.dump.plot_velocity(Re=1e6)

    # -- 9. Raw data access ----------------------------------
    print(
        f"\nPolar Re values: "
        f"{res_naca.polar.reynolds}"
    )
    print(
        f"Polar columns: "
        f"{res_naca.polar.columns}"
    )
    print(
        f"Cp entries: "
        f"{len(res_naca.cp)}"
    )
    print(
        f"Dump entries: "
        f"{len(res_naca.dump)}"
    )

    # -- 10. New data access API ----------------------------
    # Filter + column access
    cl = res_naca.polar(Re=1e6)["CL"]
    print(f"\nCL at Re=1e6: {cl[:5]}...")

    # Multi-column paired array
    paired = res_naca.polar(Re=1e6)[["alpha", "CL"]]
    print(f"alpha-CL shape: {paired.shape}")

    # Column across multiple Re => dict
    cl_all = res_naca.polar["CL"]
    print(f"CL keys: {list(cl_all.keys())}")

    # Positional access
    first_re = res_naca.polar[0]
    print(f"First Re columns: {list(first_re.keys())}")

    # Cp filter + chain
    cp_filtered = res_naca.cp(Re=1e6)(alpha=5.0)
    print(f"Cp filtered entries: {len(cp_filtered)}")

    # Dump column access
    print(f"Dump columns: {res_naca.dump.columns}")
