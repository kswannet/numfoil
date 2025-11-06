from functools import cached_property
import numpy as np
from typing import Tuple
import scipy.interpolate as si
import scipy.optimize as opt
from ..geometry.spline import BSpline2D, SplevCBezier
from ..geometry.geom2d import Point2D, Geom2D
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    def normalized_bspline(cls, data: np.ndarray | BSpline2D, find_trailing_edge: bool = True) -> BSpline2D:
        if isinstance(data, BSpline2D):
            return cls._normalize_spline(data, find_trailing_edge=find_trailing_edge)
        elif isinstance(data, np.ndarray):
            points = cls._remove_consecutive_duplicates(data)
            spline = BSpline2D(points)
        else:
            raise TypeError("Input must be a numpy array or BSpline2D.")

        return cls._normalize_spline(spline, find_trailing_edge=find_trailing_edge)

    @classmethod
    def normalize_points(cls, points: np.ndarray) -> np.ndarray:
        """Normalizes raw airfoil coordinate data points.

        Args:
            points (np.ndarray): The airfoil coordinate data points.

        Returns:
            np.ndarray: The normalized airfoil coordinate data points.
        """
        spline = cls.normalized_bspline(points)
        leading_edge, trailing_edge, _ = cls._find_leading_trailing_edges(spline)
        scale, translation, rotation = cls._compute_transformation(leading_edge, trailing_edge)
        return cls._apply_transformation(points, scale, translation, rotation)


    @classmethod
    def _normalize_spline(cls, spline: BSpline2D,  find_trailing_edge: bool = False) -> BSpline2D:
        leading_edge, trailing_edge, u_leading_edge = cls._find_leading_trailing_edges(spline,  find_trailing_edge=find_trailing_edge)
        scale, translation, rotation = cls._compute_transformation(leading_edge, trailing_edge)

        # Apply transformation to control points
        # ctrl_transformed = rotation @ ((spline.spline[1] + translation) * scale)
        # ctrl_transformed = TransformedArray(ctrl_transformed, scale=scale, translation=translation, rotation=rotation)

        # Replace spline control points
        # spline.spline[1] = ctrl_transformed

        spline.spline[1] = cls._apply_transformation(
            spline.spline[1], scale, translation, rotation
        )

        # ensure the trailing edge is correct after rotation
        spline = cls._force_trailing_edge_at_x1(spline, target="1")

        # recompute
        leading_edge, trailing_edge, u_leading_edge = cls._find_leading_trailing_edges(spline,  find_trailing_edge=False)

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
            mask[i + (abs(y[i + 1]) < abs(y[i]))] = False
        return points[mask]


    @staticmethod
    def _remove_overshoots(points: np.ndarray) -> np.ndarray:
        """Removes points that are outside the range x=[0, 1]."""
        return points[(points[:, 0] >= 0) & (points[:, 0] <= 1)]

    @classmethod
    def _find_trailing_edge(cls, spline: BSpline2D, find_trailing_edge=False, verbose=False) -> tuple[np.ndarray, np.ndarray]:
        """Finds the trailing edge of the airfoil. To account for cases where
        the trailing edge is ill-defined, or missing, there are two methods.
        First, the trailing edge is found by maximizing the distance from the
        leading edge. This is done by finding the point with the minimum
        negative L2 norm of the x-coordinate at the start and end of the spline
        separately.

        TODO : using the norm is redundant? just maximumize x-value?

        If the maximum x-values found for both sides of the spline are close
        enough together (wihtin some error tolerance), the trailing edge is
        likely well enough defined. In that case, the trailing edge is set to
        the midpoint of the start and end points of the spline.

        Args:
            spline (BSpline2D): The spline object representing the airfoil.
            find_trailing_edge (bool): Whether to solve for the trailing edge.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                The trailing edge point and the parameter ``u`` at which it
                occurs.

        Note:
            this is some janky a** code. It works (for now, I think, hard to
            tell whether there is some other stupid edgecase), but at what cost.
        """
        if find_trailing_edge:
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

            # some verification
            assert np.allclose(spline.evaluate_at(0), spline.control_points[0], rtol=0), \
                "The first control point does not coincide with u=0."
            assert np.allclose(spline.evaluate_at(1), spline.control_points[-1], rtol=0), \
                "The last control point does not coincide with u=1."

            # spline_start, spline_end = spline.evaluate_at(0), spline.evaluate_at(1)
            spline_start, spline_end = spline.control_points[0], spline.control_points[-1]
            res1_TE, res2_TE = spline.evaluate_at(res1.x[0]), spline.evaluate_at(res2.x[0])

            # if the found max x locations coincide with the spline end points,
            # at least no funky stuff is going on, at most some missing points:
            if np.allclose(res1_TE, spline_start, rtol=1e-6) and np.allclose(res2_TE, spline_end, rtol=1e-6):
                # if the spline endpoints have the same x-coordinate,
                # the trailing edge is assumed to be at the midpoint of
                # the start and end points
                if spline_start.x == spline_end.x:
                    trailing_edge = 0.5 * (spline_start + spline_end)
                    return trailing_edge

                # if the spline endpoints dont have matching x-coordinates,
                # the spline is incomplete, and the trailing edge is assumed to be
                # at the maximum x-value found, while the other side is missing
                # data.
                else:
                    # first fix the trailing edge
                    spline = cls._force_trailing_edge_at_x1(spline, target="xmax")
                    # at this point, the trailing edge should be fine, but imma
                    # check anyway because trust isssues
                    assert np.allclose(spline.evaluate_at(0)[0], spline.evaluate_at(1)[0], rtol=0), \
                        "trailing edge does not match for upper and lower surface."
                    spline_start, spline_end = spline.control_points[0], spline.control_points[-1]
                    trailing_edge = 0.5 * (spline_start + spline_end)
                    return trailing_edge

            # if the x-locations are the close,
            # yet we end up here, the spline likely overshoots the trailing
            # edge, but the trailing edge is can still be well defined.
            # checks: found x locations are close, y-coordinates are mirrored,
            # and the spline start and end points have the same x-coordinate.
            # Should be a good indicator that the spline overshoots the trailing
            # edge in the points, and then doubles back to it with both upper
            # and lower parts of the surface
            elif np.isclose(res1_TE.x, res2_TE.x, rtol=1e-4, atol=1e-4) \
                    and np.isclose(res1_TE.y, -res2_TE.y, rtol=1e-4, atol=1e-3) \
                    and np.isclose(spline_start.x, spline_end.x, rtol=1e-4, atol=1e-4):
                trailing_edge = 0.5 * (spline_start + spline_end)
                return trailing_edge

            # if x-locations are not the same, trailing edge is assumed to be
            # at the maximum x-value found, while the other side is missing data
            # and the data is causing the spline to overshoot the trailing edge.
            # elif not np.isclose(res1_TE.x, res2_TE.x, rtol=1e-4, atol=1e-4):
            else:
                # idk anymore man... is this then the final stop which
                # accounts for all the other bullshit data out there?
                u_te = res1.x[0] if -res1.fun > -res2.fun else res2.x[0]
                trailing_edge = spline.evaluate_at(u_te)
                # ! This must be verified, might be some points that need deleting
                # ! verify the trailing edge point makes sense
                spline.plot()
                plt.plot(*spline.points.T, 'bo', markersize=5, label="points")
                plt.plot(*spline_start, 'o', markersize=3, label="spline start")
                plt.plot(*spline_end, 'o', markersize=3, label="spline end")
                plt.plot(*res1_TE, '*', label="res1 TE")
                plt.plot(*res2_TE, '*', label="res2 TE")
                plt.plot(*trailing_edge, 'rx', label="trailing edge")
                plt.legend()
                breakpoint()
                return trailing_edge
            # else:
            #     # this is probably never reached. please let it never be reached.
            #     raise ValueError(
            #         f"Unable to determine trailing edge. \n" +
            #         f"Start: {spline_start}, End: {spline_end}, u1: {res1.x[0]}, u2: {res2.x[0]} \n" +
            #         f"{res1} \n {res2}"
            #         )


            # # if x-locations are not the same, trailing edge is assumed to be
            # # at the maximum x-value found, while the other side is missing data
            # if abs(res1.fun - res2.fun) > 1e-5:
            #     u_te = res1.x[0] if -res1.fun > -res2.fun else res2.x[0]
            #     trailing_edge = spline.evaluate_at(u_te)
            #     return trailing_edge #, u_te
            # else:
            #     # if the maximum x values found are the same, the trailing edge
            #     # is assumed to be at the midpoint of the start and end points
            #     start, end = spline.evaluate_at(0), spline.evaluate_at(1)
            #     if abs(start[0] - end[0]) < 1e-5:
            #         trailing_edge = 0.5 * (start + end)
            #     else:
            #         raise ValueError(
            #             f"Unable to determine trailing edge. \n" +
            #             f"Start: {start}, End: {end}, u1: {res1.x[0]}, u2: {res2.x[0]} \n" +
            #             f"{res1} \n {res2}"
            #             )
        else:
            # if argument says not to find trailing edge, this whole ordeal is
            # skipped and the trailing edge is assumed (hoped) to be the
            # midpoint of the defined end points.
            spline = cls._force_trailing_edge_at_x1(spline, target="xmax")
            trailing_edge = 0.5 * (spline.evaluate_at(0) + spline.evaluate_at(1))
            trailing_edge[0] = np.max([spline.evaluate_at(0)[0], spline.evaluate_at(1)[0]])
        return trailing_edge

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
        assert target in ["xmax", "1"], \
            f"Unknown target for trailing edge: {target}. " + \
            "Must be either 'xmax' or '1'."
        assert np.allclose(spline.evaluate_at(0), spline.control_points[0], rtol=0), \
            "The first control point does not coincide with u=0."
        assert np.allclose(spline.evaluate_at(1), spline.control_points[-1], rtol=0), \
            "The last control point does not coincide with u=1."

        spline_start, spline_end = spline.control_points[0], spline.control_points[-1]
        # get the target x-coordinate for the trailing edge:
        match target:
            case "xmax":
                x_trailing_edge = max(spline_start.x, spline_end.x)
            case "1":
                x_trailing_edge = 1.0

        # if the spline represents either the top or bottom side of an airfoil,
        # the first control point is at the leading edge, and should never be
        # adjusted. Setting it to match the target will ensure it is skipped.
        # - might be redundant
        # if spline_start.x == 0.0:
        #     spline_start = Point2D([x_trailing_edge, spline_start.y])
        #     assert spline_start.x == x_trailing_edge, \
        #         "Failed to overwrite start point"

        if spline_end.x == x_trailing_edge:
            if spline_start.x == x_trailing_edge or spline_start.x == 0.0:
                # no adjustments needed
                return spline

        # if the spline endpoints dont have matching x-coordinates,
        # the spline is incomplete, and the trailing edge is assumed to be
        # at the maximum x-value found, while the other side is missing data.
        # if abs(spline_start.x - spline_end.x) > 1e-6:

        adjusted_control_points = spline.control_points
        # check if first control point needs adjustment
        if spline_start.x != x_trailing_edge and spline_start.x != 0.0:
            # if the start point has a smaller x-value, adjust the
            # first control point to match the x-value of the last.
            # IMPORTANT!!! on the top side, the curve is defined
            # backwards in selig format (from trailing to leading edge)

            direction = adjusted_control_points[0] - adjusted_control_points[1]
            magnitude = (x_trailing_edge - adjusted_control_points[0].x) / direction[0]
            adjusted_control_points[0] += magnitude * direction

            # if the surfaces overlap after adjustment, meaning the upper point
            # was moved down too much, it is moved back up to match the lower
            # surface, keeping the x-coordinate the same.
            if adjusted_control_points[0].y < adjusted_control_points[-1].y:
                adjusted_control_points[0][1] = adjusted_control_points[-1][1]

        # check if last control point needs adjustment
        if spline_end.x != x_trailing_edge:
            # if the end point has a smaller x-value, the trailing edge
            # is on the start point side, so we adjust the last control
            # point to match the x-value of the first.
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
        if x_trailing_edge == 1.0 and target == "1" and spline_start.x != 0.0 and \
                adjusted_control_points[0].y != -adjusted_control_points[-1].y:
            # difference = adjusted_control_points[0].y + adjusted_control_points[-1].y
            # adjusted_control_points[0][1] -= difference / 2
            # adjusted_control_points[-1][1] += difference / 2
            adjusted_control_points[-1][1] = -adjusted_control_points[0][1]

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
        assert np.allclose(spline.evaluate_at(0), spline.control_points[0], rtol=0), \
            "The first control point does not coincide with u=0 after adjustment."
        assert np.allclose(spline.evaluate_at(1), spline.control_points[-1], rtol=0), \
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
    def _find_leading_trailing_edges(cls, spline: BSpline2D, find_trailing_edge=True) -> tuple[np.ndarray, np.ndarray, float]:
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
        trailing_edge = cls._find_trailing_edge(spline, find_trailing_edge)
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
