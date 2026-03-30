import numpy as np
import scipy.interpolate as si
import scipy.optimize as opt
import os

from typing import Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from ..geometry.spline import BSpline2D, SplevCBezier
from ..geometry.geom2d import Point2D, Geom2D


class AirfoilNormalizer:
    """
    Normalizes airfoil data — either raw points or a BSpline2D — and returns a
    spline with transformed control points containing transformation metadata.

    Airfoil data, especially from the UIUC database, sucks. It is messy,
    inconsistent (varying number of points, trailing edges from closed to
    gigantic gaps), contains errors (misplaced points), missing leading and
    trailing edge points, and is a pain to work with. Instead of filtering out
    any airfoil which causes issues (which wouldn't leave many usable ones),
    this class tries to normalize the data as best as possible. Like the data,
    this process is messy, trying to deal with all the different edge cases, and
    is far from flawless. But it kinda works. There are probably better ways to
    deal with this, but those are outside my capabilities.
    """
    @classmethod
    def normalize_points(cls, points: np.ndarray) -> np.ndarray:
        """Normalizes raw airfoil coordinate data points.
        Similar to ``normalized_bspline``, but returns the normalized points
        instead of a spline object.

        !!! Note: DO NOT USE THIS unless you really want the original points.
        Instead, use ``normalized_bspline`` to get a normalized spline object
        with transformation metadata and complete normalized surface bspline.

        Args:
            points (np.ndarray): The airfoil coordinate data points.

        Returns:
            np.ndarray: The normalized airfoil coordinate data points.
        """
        # breakpoint() # ! reconsider... please...
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError("Input points must be a 2D array with shape (N, 2).")
        points = cls._fill_data_gaps(points)
        spline = cls._fix_trailing_edge(BSpline2D(points))
        leading_edge, trailing_edge, _ = cls._find_leading_trailing_edges(spline)
        scale, translation, rotation = cls._compute_transformation(leading_edge, trailing_edge)
        return cls._apply_transformation(points, scale, translation, rotation).T

    @classmethod
    def normalized_bspline(cls, data: np.ndarray | BSpline2D) -> BSpline2D:
        """Normalizes airfoil data and returns it as a BSpline2D object.

        Args:
            data (np.ndarray | BSpline2D):
                The airfoil data to be normalized, either as raw coordinate points
                or as a BSpline2D object.

        Returns:
            BSpline2D:
                The normalized airfoil as a BSpline2D object.
        """
        # create an initial spline from the given points
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[-1] != 2:
                raise ValueError(
                    "Input points must be a 2D array with shape (N, 2)"
                )
            points = cls._remove_consecutive_duplicates(data)
            points = cls._fill_data_gaps(points)
            data = BSpline2D(points)

        if not isinstance(data, BSpline2D):
            raise TypeError("Input must be a numpy array or BSpline2D.")

        return cls._normalize_spline(data)

    @classmethod
    def _normalize_spline(cls, spline: BSpline2D) -> BSpline2D:
        """Normalizes a BSpline2D airfoil spline.

        Args:
            spline (BSpline2D): The airfoil spline to be normalized.
        Returns:
            BSpline2D: The normalized airfoil spline with transformation metadata.
        """
        # first fix potential trailing edge issues
        spline = cls._fix_trailing_edge(spline)

        # get the transformation parameters
        leading_edge, trailing_edge, u_leading_edge = cls._find_leading_trailing_edges(spline)
        scale, translation, rotation = cls._compute_transformation(leading_edge, trailing_edge)

        # apply the transformation to the spline control points
        spline.spline[1] = cls._apply_transformation(
            spline.spline[1], scale, translation, rotation
        )

        # ensure the trailing edge is correct after rotation
        spline = cls._force_trailing_edge_at_x1(spline, target="1")

        # recompute
        leading_edge, trailing_edge, u_leading_edge = cls._find_leading_trailing_edges(spline)

        # append additional information
        spline.leading_edge = leading_edge
        spline.trailing_edge = trailing_edge
        spline.u_leading_edge = u_leading_edge
        return spline

    @staticmethod
    def _apply_transformation(points: np.ndarray, scale: float, translation: np.ndarray, rotation: np.ndarray | float) -> np.ndarray:
        """Applies the transformation to an array of points.
        Args:
            points (np.ndarray): The points to be transformed.
            scale (float): The scaling factor.
            translation (np.ndarray): The translation vector.
            rotation (np.ndarray | float): The rotation matrix or angle in radians.
        Returns:
            np.ndarray(2, N): The transformed points.
        """
        points = np.asarray(points)
        # return (rotation @ ((points + translation) * scale).T).T.view(Point2D)
        if isinstance(rotation, float):
            rotation = np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation),  np.cos(rotation)]
            ])
        elif rotation.shape != (2, 2):
            raise ValueError("Rotation must be a 2x2 matrix or a rotation angle in radians.")

        if points.ndim == 2 and points.shape[0] != 2:
            points = points.T

        if translation.ndim == 1:
            translation = translation[:, np.newaxis]

        return TransformedArray(
            rotation @ ((points + translation) * scale),
            (scale, translation, rotation)
        )

    @staticmethod
    def _remove_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
        """Removes consecutive duplicate points from the array."""
        # # this one checks for exact duplicates
        # diff = np.diff(points, axis=0)
        # idx = np.where(np.any(diff != 0, axis=1))[0] + 1
        # # this one checks for duplicates in x, regardless of y
        # x_diff = np.diff(points[:, 0])
        # idx = np.where(x_diff != 0)[0] + 1
        # return np.vstack([points[0], points[idx]])
        # # new version that should also fix manually closed trailing edges
        x, y = points[:, 0], points[:, 1]
        mask = np.ones(len(points), dtype=bool)
        for i in np.where(np.isclose(np.diff(x), 0))[0]:
            mask[i + (abs(y[i + 1]) > abs(y[i]))] = False
        return points[mask]

    @staticmethod
    def _remove_overshoots(points: np.ndarray) -> np.ndarray:
        """Removes points that are outside the range x=[0, 1]."""
        return points[(points[:, 0] >= 0) & (points[:, 0] <= 1)]

    @classmethod
    def _fill_data_gaps(cls, points: np.ndarray, gap_threshold: float = 0.15) -> np.ndarray:
        """Detect and fill large gaps in airfoil coordinate data with
        interpolated points.

        This method identifies gaps in x-coordinate data that are larger than the
        threshold and inserts a single interpolated point at the midpoint of each gap.
        This is useful for airfoils like b707b.dat that have missing data sections.

        Args:
            points: Airfoil coordinate array with shape [N, 2]
            gap_threshold: Minimum gap size in x-coordinate to trigger filling (default: 0.15)

        Returns:
            np.ndarray: Points with gaps filled by interpolated midpoints
        """
        if len(points) < 4:
            return points

        # The number of points to insert and interpolation points now depend on gap size
        # // just some settings to mess with. currently not accessible from outside
        # // n_fill = 3 # number of points to insert per gap
        # // n_interp = 3 # number of points to use for interpolation on each side of the gap

        # Calculate gaps between consecutive x-coordinates
        x_diffs = np.abs(np.diff(points[:, 0]))
        gap_indices = np.where(x_diffs > gap_threshold)[0]

        if len(gap_indices) == 0:
            return points  # No gaps to fill

        # Work backwards through gaps to avoid index shifting when inserting
        filled_points = points.copy()
        for gap_idx in reversed(gap_indices):
            # number of infill points depends on the gap size
            gap_size = x_diffs[gap_idx]

            # add in one point per ~0.11 gap size
            n_fill = int(gap_size // 0.11)

            # use at least 3 points on either side for interpolation, but add more for larger gaps
            n_interp = max(3, int(gap_size // 0.11))

            # Points on either side of the gap
            pt1 = filled_points[gap_idx]
            pt2 = filled_points[gap_idx + 1]
            filler_x = np.linspace(pt1[0], pt2[0], num=n_fill + 2)[1:-1]  # 3 points in between

            flank_pts = filled_points[
                min(gap_idx, max(0, gap_idx - n_interp + 1)) : min(len(filled_points), gap_idx + n_interp)
            ]  # n_interp points on either side of the gap
            if np.all(np.diff(flank_pts[:, 0])<0):
                flank_pts = flank_pts[::-1]

            # Create interpolated midpoint
            # mid_x = 0.5 * (pt1[0] + pt2[0])
            # mid_y = 0.5 * (pt1[1] + pt2[1])  # Linear interpolation for y
            # mid_y = float(f_interp(mid_x))  # Use interpolator for y-coordinate
            # midpoint = np.array([mid_x, mid_y])

            # Insert midpoint after the first point of the gap
            # filled_points = np.insert(filled_points, gap_idx + 1, midpoint,
            # axis=0)

            filler_y = si.pchip_interpolate(flank_pts[:, 0], flank_pts[:, 1], filler_x)
            filler_points = np.column_stack((filler_x, filler_y))
            filled_points = np.insert(filled_points, gap_idx + 1, filler_points, axis=0)
        return filled_points

    @classmethod
    def _get_key_locations(cls, spline: BSpline2D) -> Tuple[Point2D, Point2D]:
        """find the endpoints and most aft points on the airfoil spline.

        Args:
            spline (BSpline2D): The spline object representing the airfoil.

        Returns:
            tuple[Point2D, Point2D, Point2D, Point2D]:
                The start and end control points of the spline, and the most
                aft points on the upper and lower surfaces.
        """
        upper_search_result = opt.minimize(
            lambda u: -spline.evaluate_at(u).x,
            # lambda u: -np.linalg.norm(spline.evaluate_at(u).x),
            0,  # search on upper surface (starting from the start of the spline)
            bounds=[(0, 1)],
            method="SLSQP",
        )
        lower_search_result = opt.minimize(
            lambda u: -spline.evaluate_at(u).x,
            # lambda u: -np.linalg.norm(spline.evaluate_at(u).x),
            1,  # search on lower surface (starting form the end of the spline)
            bounds=[(0, 1)],
            method="SLSQP",
        )

        if not upper_search_result.success or not lower_search_result.success:
            raise RuntimeError(
                "Failed to find aft most spline points. \n" +
                "Search on upper surface: \n" +
                f"{str(upper_search_result)} \n" +
                "Search on lower surface: \n" +
                f"{str(lower_search_result)}"
            )

        # Verify spline endpoint consistency (control points should match evaluation)
        assert np.allclose(spline.evaluate_at(0), spline.control_points[0], rtol=0), \
            "The first control point does not coincide with u=0."
        assert np.allclose(spline.evaluate_at(1), spline.control_points[-1], rtol=0), \
            "The last control point does not coincide with u=1."

        # extract key points
        spline_start = spline.control_points[0]#.round(7).view(Point2D)
        spline_end = spline.control_points[-1]#.round(7).view(Point2D)
        upper_aft_pt = spline.evaluate_at(upper_search_result.x[0])#.round(7).view(Point2D)
        lower_aft_pt = spline.evaluate_at(lower_search_result.x[0])#.round(7).view(Point2D)
        return spline_start, spline_end, upper_aft_pt, lower_aft_pt

    @classmethod
    def _trailing_edge(cls, spline: BSpline2D,) -> tuple[np.ndarray, np.ndarray]:
        """The simplest possible trailing edge finder: midpoint of endpoints.
        This assumes `_fix_trailing_edge()` has already been called."""
        return 0.5 * (spline.evaluate_at(0) + spline.evaluate_at(1))

    @classmethod
    def _fix_trailing_edge(cls, spline: BSpline2D,) -> tuple[np.ndarray, np.ndarray]:
        """Finds the trailing edge of the airfoil splines with problematic data.

        This method accounts for airfoil data in the following cases:

        1. Well-defined closed TE: Upper and lower surfaces meet at same point
        2. Incomplete data: One surface extends further aft than the other
        3. Overshoot with symmetric TE: Spline overshoots then doubles back
        4. Complex malformation: Asymmetric overshoots or other geometric issues

        Detection Strategy:
        - Use optimization to find maximum x-coordinates from both spline ends
        - Compare optimization results with spline endpoints
        - Classify case based on geometric relationships
        - Apply appropriate correction strategy

        Args:
            spline (BSpline2D): The spline object representing the airfoil.
            find_trailing_edge (bool): Whether to solve for the trailing edge.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                The trailing edge point and the parameter ``u`` at which it
                occurs.

        Note:
            Use np.isclose to avoid floating point issues.
            This is some janky a** code. It works (for now, I think, hard to
            tell whether there is some other stupid edgecase), but at what cost.
            Set method to slsqp because default one fails too often.
            Edit: it did in fact not work 100% of the time. Tried to fix again.

        """
        # # **Simple Case**: Skip detection
        # if not find_trailing_edge:
        #     # endpoints define trailing edge after forcing x-alignment
        #     spline = cls._force_trailing_edge_at_x1(spline, target="xmax")
        #     trailing_edge = 0.5 * (spline.evaluate_at(0) + spline.evaluate_at(1))
        #     return trailing_edge

        # **Core Detection**: Find maximum x-coordinates from both spline ends
        (
            spline_start,
            spline_end,
            upper_aft_pt,
            lower_aft_pt,
        ) = cls._get_key_locations(spline)

        # **Case 1: Well-defined or incomplete trailing edge**
        """If the found max x locations coincide with the spline end points,
        no funky stuff going on, at most the trailing edge is missing data:"""
        if np.allclose(upper_aft_pt, spline_start, rtol=0) and np.allclose(lower_aft_pt, spline_end, rtol=0):

            # **Sub-case 1a: Endpoints have same x-coordinate**
            if np.isclose(spline_start.x, spline_end.x, rtol=0):
                """If the spline endpoints have the same x-coordinate, all is
                good in the world, the trailing edge is assumed to be at the
                midpoint between the start and end points of the spline."""
                pass

            # **Sub-case 1b: Endpoints have different x-coordinates**
            else:
                """If the spline endpoints don't have matching x-coordinates,
                the data is incomplete, and the trailing edge is assumed to
                be at the maximum x-value found, while the other side is
                missing (a) datapoint(s). As a dirty fix, the spline is
                extended to x=1 by translating the respective endpoint."""
                # # first fix the trailing edge
                # spline = cls._force_trailing_edge_at_x1(spline, target="xmax")
                # # at this point, the trailing edge should be fine, but imma
                # # check anyway because trust isssues
                # assert np.allclose(spline.evaluate_at(0)[0], spline.evaluate_at(1)[0], rtol=0), \
                #     "trailing edge does not match for upper and lower surface."
                pass  # ! _force_trailing_edge_at_x1 always called before final return

        # **Case 2: Overshoot with symmetric trailing edge**
        # ! this case might be redundant now, already covered in case 3
        # Maximum x-points are found interior to the spline (not at endpoints)
        # but show symmetric geometric properties indicating a well-defined TE
        elif (upper_aft_pt.x > 1.0 and lower_aft_pt.x > 1.0                         # Both sides overshoot x=1
            and np.isclose(upper_aft_pt.x, lower_aft_pt.x, rtol=1e-3, atol=1e-3)    # Approx same x for most aft points
            and np.isclose(upper_aft_pt.y, -lower_aft_pt.y, rtol=1e-3, atol=1e-3)   # Approx symmetric y-coordinates
            and np.allclose(spline_start, spline_end, rtol=1e-4, atol=1e-5)  # Spline Endpoints match
            and np.allclose(spline_start, np.array([1.0, 0.0]), rtol=1e-4, atol=1e-4)  # spline ends at (1,0)
            ):
            """This pattern occurs (or should) when:
            - Original airfoil had open trailing edge [x=1, y_upper] and [x=1, y_lower]
            - Manual point was added in between: [x=1, y=0] (duplicated on both sides)
            - This creates a triple point at the trailing edge, data looks like:
                [ [1, 0], [1, y_upper], ..., [1, y_lower], [1, 0] ]
            - Spline tries to pass through both points on either side,
                causing the spline to overshoot x=1
            - Spline travels past x=1, then doubles back to reach the (suspect
              manually) defined endpoint

            The symmetric max points indicate the overshoot peak, but the true
            trailing edge is at the spline endpoints where the surfaces
            converge.

            Example airfoil: fx69274
            """
            # In theory this should always be the case now,
            # but again, trust issues, so still checking again...
            if np.isclose(spline.points.x[0], spline.points.x[1], rtol=1e-3, atol=1e-3) \
                and np.isclose(spline.points.x[-1], spline.points.x[-2], rtol=1e-3, atol=1e-3):
                # Now get rid of the problematic point that should never have
                # been there in the first place...
                spline = BSpline2D(spline.points[1:-1])
            else:
                breakpoint() # investigate further if this happens
                raise RuntimeError(
                    "Unexpected spline endpoint configuration detected."
                )

        # **Case 3: Asymmetric overshoot, or anything else**
        else:
            """This handles:
            1. Asymmetric overshoot: One side overshoots (more than the other)
                - example: ah81k144wfKlappe
            2. Symmetric overshoot that didn't match the tolerances in Case 2
                This step might make the previous case redundant, but oh well,
                better safe than sorry
            3. Everything else

            Strategy: Remove overshooting endpoints and potentially insert
            cut-off points where the spline begins to overshoot beyond the
            natural trailing edge.

            Note:
                This is a destructive operation and may remove valid data!
                Initially, only points on the overshooting side were removed
                (commented out), but switched to removing points on both sides
                as the yields cleaner results overall.
            """
            # * First try a gentle nudge is the overshoot is small
            if max(upper_aft_pt.x, lower_aft_pt.x) - 1.0 < 1e-4:
                ctrl_pts = spline.control_points.copy()
                ctrl_pts[ctrl_pts.x - 1e-8 > 1.0, 0] = 0.9995
                spline.control_points = ctrl_pts

            (  # recalculate key locations
                spline_start,
                spline_end,
                upper_aft_pt,
                lower_aft_pt,
            ) = cls._get_key_locations(spline)

            # * if not fixed yet, start removing points
            while not (
                np.allclose(upper_aft_pt, spline_start, rtol=0)
                and np.allclose(lower_aft_pt, spline_end, rtol=0)
            ):
                points = spline.points.copy()
                # * Remove points on the overshooting side only
                # if not np.all(upper_aft_pt == spline_start):    # upper aft point not at start (overshoot)
                #     points = points[1:]                         # remove first point
                # if not np.all(lower_aft_pt == spline_end):      # lower aft point not at end (overshoot)
                #     points = points[:-1]                        # remove last point
                # * remove points on both sides
                spline = BSpline2D(points[1:-1])                  # rebuild spline
                (
                    spline_start,
                    spline_end,
                    upper_aft_pt,
                    lower_aft_pt,
                ) = cls._get_key_locations(spline)

        spline = cls._force_trailing_edge_at_x1(spline, target="xmax")
        return spline


    @staticmethod
    def plot_te_debug(spline: BSpline2D, save_path: str = "te_debug") -> plt.Figure:
        res1 = opt.minimize(lambda u: -np.linalg.norm(spline.evaluate_at(u).x), 0, bounds=[(0, 1)], method="SLSQP")
        res2 = opt.minimize(lambda u: -np.linalg.norm(spline.evaluate_at(u).x), 1, bounds=[(0, 1)], method="SLSQP")
        # res1 = opt.minimize(lambda u: -spline.evaluate_at(u)[0][0], 0, bounds=[(0, 1)])
        # res2 = opt.minimize(lambda u: -spline.evaluate_at(u)[0][0], 1, bounds=[(0, 1)])
        if not res1.success or not res2.success:
            raise RuntimeError(
                "Failed to find trailing edge. \n" +
                "Search on upper surface: \n" +
                f"{str(res1)} \n" +
                "Search on lower surface: \n" +
                f"{str(res2)}"
            )
        # spline_start, spline_end = spline.evaluate_at(0), spline.evaluate_at(1)
        spline_start, spline_end = spline.control_points[0], spline.control_points[-1]
        upper_aft_pt, lower_aft_pt = spline.evaluate_at(res1.x[0]), spline.evaluate_at(res2.x[0])
        trailing_edge = AirfoilNormalizer._trailing_edge(spline)
        leading_edge = AirfoilNormalizer._find_leading_edge(spline, trailing_edge=trailing_edge)[0]

        fig, (ax_le_zoom, ax, ax_te_zoom) = plt.subplots(
            1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [4, 7, 4]},
            sharey=False, #constrained_layout=True
        )
        fig.subplots_adjust(wspace=-0.1)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax_te_zoom.set_facecolor("white")
        ax_le_zoom.set_facecolor("white")

        # --- Full Airfoil Plot ---
        xx, yy = spline.evaluate_at(np.linspace(0, 1, 10000)).T

        ax.plot(xx, yy, 'k', label="spline")
        ax.plot(*spline.points.T, 'bo', markersize=4, label="points")
        ax.plot(*spline_start, 'o', markersize=3)#, label="spline start")
        ax.plot(*spline_end, 'o', markersize=3)#, label="spline end")
        ax.plot(*upper_aft_pt, '*')#, label="res1 TE")
        ax.plot(*lower_aft_pt, '*')#, label="res2 TE")
        ax.plot(*trailing_edge, 'rx')#, label="trailing edge")
        # ax.plot(*spline.control_points.T, 'go-', markersize=4, label="control points")

        ax.set_title("Trailing Edge Debugging")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.axis("equal")
        ax.legend(fontsize="small")

        # --- Zoomed TE Region ---
        x_min = min(upper_aft_pt.x, lower_aft_pt.x, trailing_edge.x, spline.points.x[0], spline.points.x[-1], spline.control_points.x[1], spline.control_points.x[-2]) - 0.005
        x_max = max(upper_aft_pt.x, lower_aft_pt.x, trailing_edge.x, spline.points.x[0], spline.points.x[-1], spline.control_points.x[1], spline.control_points.x[-2]) + 0.0005
        y_min = min(upper_aft_pt.y, lower_aft_pt.y, trailing_edge.y, spline.points.y[0], spline.points.y[-1], spline.control_points.y[1], spline.control_points.y[-2])*1.3 - 0.001
        y_max = max(upper_aft_pt.y, lower_aft_pt.y, trailing_edge.y, spline.points.y[0], spline.points.y[-1], spline.control_points.y[1], spline.control_points.y[-2])*1.3 + 0.001

        ax_te_zoom.plot(xx, yy, 'k', linewidth=1,)
        ax_te_zoom.plot(*spline_start, 'o', markersize=6, label="spline start")
        ax_te_zoom.plot(*spline_end, 'o', markersize=5, label="spline end")
        ax_te_zoom.plot(*spline.points.T, 'bo', markersize=4, label="points")
        ax_te_zoom.plot(*upper_aft_pt, '*', markersize=6, label="upper TE solution")
        ax_te_zoom.plot(*lower_aft_pt, '*', markersize=5, label="lower TE solution")
        ax_te_zoom.plot(*trailing_edge, '*', color="orange", label="Selected trailing edge")
        ax_te_zoom.plot(*spline.control_points.T, 'go-', markersize=1, linewidth=0.5, label="control points")

        ax_te_zoom.set_xlim(x_min, x_max)
        ax_te_zoom.set_ylim(y_min, y_max)
        ax_te_zoom.set_title("Trailing Edge Zoom")
        # ax_te_zoom.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax_te_zoom.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))#, useOffset=True)

        # Clean look
        ax_te_zoom.grid(True, linestyle="--", alpha=0.4)
        ax_te_zoom.yaxis.tick_right()
        ax_te_zoom.yaxis.set_label_position("right")
        ax_te_zoom.legend(loc="center left", fontsize="small")

        mark_inset(ax, ax_te_zoom, loc1=2, loc2=3, fc="none", ec="0.5")

        # --- Zoomed LE Region ---
        le_x_min = min(0.0, leading_edge.x, min(xx)) - 0.0002
        le_x_max = max(0.0, leading_edge.x) + 0.0005
        le_y_min = min(0.0, leading_edge.y) - 0.007
        le_y_max = max(0.0, leading_edge.y) + 0.007

        ax_le_zoom.plot(xx, yy, 'k', linewidth=1)
        ax_le_zoom.plot(0,0, 'ro', markersize=6, label="Origin")
        ax_le_zoom.plot(*spline.points.T, 'bo', markersize=4, label="points")

        ax_le_zoom.plot(*leading_edge, '*', color="orange", markersize=8, label="leading edge")

        ax_le_zoom.set_xlim(le_x_min, le_x_max)
        ax_le_zoom.set_ylim(le_y_min, le_y_max)
        ax_le_zoom.set_title("Leading Edge Zoom")
        # ax_le_zoom.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
        ax_le_zoom.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax_le_zoom.grid(True, linestyle="--", alpha=0.4)

        ax_le_zoom.legend(loc="lower left", fontsize="small")

        mark_inset(ax, ax_le_zoom, loc1=1, loc2=4, fc="none", ec="0.5")

        # --- Save ---
        fig.tight_layout()
        fig.savefig(save_path, transparent=False, format="pdf")
        return fig, (ax_le_zoom, ax, ax_te_zoom)

    @staticmethod
    def _force_trailing_edge_at_x1(spline: BSpline2D, target: str = "xmax"):
        """
        Forces the trailing edge of the spline to be at x=1. This is done by
        adjusting the last control point to match the x-coordinate of the
        trailing edge point, which is assumed to be at the end of the spline.
        Args:
            curve (BSpline2D): The spline object representing the airfoil.
            target (str): The target x-coordinate for the trailing edge.
                - "xmax" to force the trailing edge at the maximum x found
                - "1" to force trailing edge at x=1.0
        Returns:
            BSpline2D: The adjusted spline with the trailing edge at the target
                x location.
        """
        if not target in ["xmax", "1"]:
            raise ValueError(
                f"Unknown target for trailing edge: {target}. " +
                "Must be either 'xmax' or '1'."
            )
        if not np.allclose(spline.evaluate_at(0), spline.control_points[0], rtol=0):
            raise ValueError(
                f"The first control point, ({spline.control_points[0]}), does not coincide with u=0."
            )
        if not np.allclose(spline.evaluate_at(1), spline.control_points[-1], rtol=0):
            raise ValueError(
                f"The last control point, ({spline.control_points[-1]}) does not coincide with u=1."
            )

        spline_start, spline_end = spline.control_points[0], spline.control_points[-1]
        # get the target x-coordinate for the trailing edge:
        # match target:
        #     case "xmax":
        #         x_trailing_edge = max(spline_start.x, spline_end.x)
        #     case "1":
        #         x_trailing_edge = 1.0

        # can't use match-case because of autoformatter incompatibility
        # change back later, but for now autoformatter is more useful
        if target == "xmax":
            x_trailing_edge = max(spline_start.x, spline_end.x)
        elif target == "1":
            x_trailing_edge = 1.0
        else:
            raise ValueError(
                f"Unknown target for trailing edge: {target}. " +
                "Must be either 'xmax' or '1'."
            )

        # if the spline represents either the top or bottom side of an airfoil,
        # the first control point is at the leading edge, and should never be
        # adjusted. Setting it to match the target will ensure it is skipped.
        # - might be redundant
        # if spline_start.x == 0.0:
        #     spline_start = Point2D([x_trailing_edge, spline_start.y])
        #     assert spline_start.x == x_trailing_edge, \
        #         "Failed to overwrite start point"

        if spline_end.x == x_trailing_edge:                                 # last point already at target
            if spline_start.x == x_trailing_edge or spline_start.x == 0.0:  # first point also at target
                # no adjustments needed
                return spline

        # if the spline endpoints dont have matching x-coordinates,
        # the spline is incomplete, and the trailing edge is assumed to be
        # at the maximum x-value found, while the other side is missing data.
        # if abs(spline_start.x - spline_end.x) > 1e-6:

        adjusted_control_points = spline.control_points.copy()
        # check if first control point needs adjustment
        if spline_start.x != x_trailing_edge and spline_start.x != 0.0:
            # move the first control point along direction from point 1 to point
            # 0 to match the target x-coordinate.
            # IMPORTANT!!! on the top side, the curve is defined
            # backwards in selig format (from trailing to leading edge)

            direction = adjusted_control_points[0] - adjusted_control_points[1]             # vector from point 1 to point 0
            magnitude = (x_trailing_edge - adjusted_control_points[0].x) / direction[0]     # scaling factor to reach target x
            adjusted_control_points[0] += magnitude * direction

            # if the surfaces overlap after adjustment, meaning the upper point
            # was moved down too much, it is moved back up to match the lower
            # surface, keeping the x-coordinate the same.
            if adjusted_control_points[0].y < adjusted_control_points[-1].y:
                adjusted_control_points[0][1] = adjusted_control_points[-1][1]

        # check if last control point needs adjustment
        if spline_end.x != x_trailing_edge:
            # move the last control point along direction from point -2 to
            # point -1 to match the target x-coordinate.
            direction = adjusted_control_points[-1] - adjusted_control_points[-2]
            magnitude = (x_trailing_edge - adjusted_control_points[-1].x) / direction[0]
            adjusted_control_points[-1] += magnitude * direction

            # if the surfaces overlap after adjustment, the adjusted control
            # point is moved to the trailing edge point.
            if adjusted_control_points[-1].y > adjusted_control_points[0].y:
                adjusted_control_points[-1][1] = adjusted_control_points[0][1]

        # Finally, if the output should be a normalized spline, it is assumed
        # tne provided spline was already normalized. Following the potential
        # adjustemts above, the trailing edge is forced to be mirrored.
        # This is only done if the target is 1, and if the spline does not
        # start at the leading edge.
        if x_trailing_edge == 1.0 and target == "1" and spline_start.x != 0.0:
            gap = adjusted_control_points[0].y - adjusted_control_points[-1].y
            adjusted_control_points[0][1] = 0.5 * gap
            adjusted_control_points[-1][1] = - 0.5 * gap
            #! if any control points exceed x=1 after adjustment, set them to 1
            #! this is a really dirty fix, only as last resort
            adjusted_control_points[adjusted_control_points.x > 1.0, 0] = 0.9995

        spline.control_points = adjusted_control_points

        # verify that the adjustement worked
        assert spline.control_points[-1].x == x_trailing_edge, \
            f"The last control points do not have the required x-coordinate after adjustment:\n" + \
            f"Last: {spline.control_points[-1]} instead of x={x_trailing_edge}."

        if spline_start.x != 0.0:
            assert spline.control_points[0].x == x_trailing_edge, \
                f"The first control point does not have have the required x-coordinate after adjustment:\n" + \
                f"Last: {spline.control_points[0]} instead of x={x_trailing_edge}."
            if target == "1":
                assert spline.control_points[0].y == -spline.control_points[-1].y, \
                    "The first and last control points do not have the same y-coordinate after adjustment. "

        # Also verify that the adjusted control points are valid:
        assert np.allclose(spline.evaluate_at(0), spline.control_points[0], rtol=1e-4), \
            "The first control point does not coincide with u=0 after adjustment."
        assert np.allclose(spline.evaluate_at(1), spline.control_points[-1], rtol=1e-4), \
            "The last control point does not coincide with u=1 after adjustment."

        return spline

    @staticmethod
    def _find_leading_edge(spline: BSpline2D, trailing_edge: Point2D) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the leading edge of the airfoil by maximizing the distance from
        the trailing edge. This is done by minimizing the negative L2 norm of
        the distance between the trailing edge and the spline points.

        Args:
            spline (BSpline2D):
                spline object representing the airfoil.
            trailing_edge (Point2D):
                The trailing edge point coordinates.

        Returns:
            tuple[np.ndarray, float]:
                The leading edge point and the parameter ``u`` at which it occurs.
        """
        def objective(u):
            return -np.linalg.norm(trailing_edge - spline.evaluate_at(u))

        res = opt.minimize(objective, 0.5, bounds=[(0, 1)], method="SLSQP")
        if not res.success:
            raise RuntimeError("Failed to find leading edge. \n" + str(res))
        return spline.evaluate_at(res.x[0]), res.x[0]

    @classmethod
    def _find_leading_trailing_edges(cls, spline: BSpline2D,) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Simply combines the two methods for leading and trailing edge in
        a single function. just for convenience.

        Args:
            spline (BSpline2D):
                The spline object representing the airfoil.
            find_trailing_edge (bool):
                Whether to solve for the trailing edge.

        Returns:
            tuple[np.ndarray, np.ndarray, float]:
                leading edge point, trailing edge point, and the parameter ``u``
                at which the leading edge occurs.
        """
        trailing_edge = cls._trailing_edge(spline)
        leading_edge, u_leading_edge = cls._find_leading_edge(spline, trailing_edge)
        return leading_edge, trailing_edge, u_leading_edge

    @staticmethod
    def _compute_transformation(leading_edge, trailing_edge):
        chord = trailing_edge - leading_edge
        scale = 1.0 / np.linalg.norm(chord)
        translation = -leading_edge[:, np.newaxis]

        angle = -np.arctan2(chord[1], chord[0])
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        return (scale, translation, rotation)


class TransformedArray(Point2D):
    """
    Numpy Array subclass to store transformed points in a 2D array and retain
    the transformation applied during normalization.
    """
    def __new__(cls, input_array, transformation):
        obj = np.asarray(input_array).view(cls)
        obj.transformation = transformation
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.transformation = getattr(obj, 'transformation', None)

    @property
    def scale(self):
        """Returns the scaling factor which was applied."""
        return self.transformation[0]

    @property
    def translation(self):
        """Returns the applied translation vector."""
        return self.transformation[1]

    @property
    def rotation(self) -> float:
        """Returns the applied rotation angle in radians."""
        return self.transformation[2]


def normalize_airfoil_dir(original_dir: str = "UIUC_airfoils/original", output_dir: str = "UIUC_airfoils/smoothed"):
    from numfoil.geometry.airfoil import BezierAirfoil
    """Smoothens and normalizes given airfoil coordinate data.
    Nornalizes the data using an interpolating Bspline, then smoothens by
    fitting Bezier curves with as high a number of control points as
    possible. if fitting falls, the number of control points is reduced until
    a fit is found or the number of control points is reduced to 6, after which
    point this function (and I) gives up.

    Args:
        original_dir (str): Directory containing the original airfoil data files.
        output_dir (str): Directory where the converted Bezier airfoil files will be saved.
    """
    if not os.path.exists(original_dir):
        raise FileNotFoundError(f"Original directory {original_dir} does not exist.")
    if not os.path.isdir(original_dir):
        raise NotADirectoryError(f"Original directory {original_dir} is not a directory.")
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(original_dir)

    for idx, file in enumerate(tqdm(files, total=len(files), ncols=100, unit="file", desc="Loading Airfoils")):
        success = False
        curvefit_kwargs = {
            "n_control_points": 20,
        }
        while not success:
            try:
                airfoil = BezierAirfoil.from_file(
                    os.path.join(original_dir, file),
                    curvefit_kwargs=curvefit_kwargs
                )
                filename = os.path.join(output_dir, f"{airfoil.name}.dat")
                np.savetxt(filename, airfoil.points, fmt="%.6f", header=airfoil.description, comments="")
                success = True

            except Exception:
                print(f"\nFailed to fit {file} with n={curvefit_kwargs['n_control_points']}, trying with n={curvefit_kwargs['n_control_points'] - 1}")
                if curvefit_kwargs["n_control_points"] < 6:
                    print(f"\nFailed to fit {file} with n={curvefit_kwargs['n_control_points']}, giving up.")
                    break
                curvefit_kwargs = {
                    "n_control_points": curvefit_kwargs["n_control_points"] - 1,
                }
