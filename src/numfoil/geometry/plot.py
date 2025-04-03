from functools import cached_property
from typing import Tuple, Union, Literal

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arc

from .airfoil import AirfoilBase as Airfoil
from ..util import Container, cosine_spacing


class AirfoilPlot:
    """ Create a plotter object with ready functions to plot different
    properties of the airfoil. calling the respective methods more than once
    will toggle the property plot.

    plot methods toggle properties shown. Methods are decorated as properties
    just save the extra 2 characters to type.
    """
    def __init__(self, airfoil: Airfoil, n_points: int = 200) -> None:
        self.airfoil = airfoil
        self.n_points = n_points
        self.elements = Container(
            surface_line=None,
            upper_surface_line=None,
            lower_surface_line=None,
            camber_line=None,
            leading_edge_point=None,
            leading_edge_radius=None,
            leading_edge_angle=None,
            points=None,
            unprocessed_points=None,
            max_camber=None,
            max_thickness=None,
            max_curvature=None,
            upper_crest=None,
            lower_crest=None,
            upper_crest_curvature=None,
            lower_crest_curvature=None,
            surface_curvature=[],
            upper_curvature=[],
            lower_curvature=[],
            camber_curvature=[],
            trailing_edge_wedgedge=[],
            trailing_edge_angle=None,
            control_points=None,
            upper_control_points=None,
            lower_control_points=None,
        )

        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.draw()

        # self.fig, self.ax = plt.subplots()
        # self.ax.set_xlabel("Normalized Location Along Chordline (x/c)")
        # self.ax.set_ylabel("Normalized Thickness (t/c)")
        # self.ax.set_title(airfoil.fullname)
        # plt.axis("equal")
        # self.upper_surface, self.lower_surface, self.camber_line
        # self.ax.legend(loc="best")
        # plt.show()

    def draw(self):
        """
        Draw the airfoil geometry.

        This method creates a plot of the airfoil geometry using matplotlib.
        It sets the x and y labels, the title, and ensures that the plot is
        displayed with equal aspect ratio. It also adds a legend to the plot.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Normalized Location Along Chordline (x/c)")
        self.ax.set_ylabel("Normalized Thickness (t/c)")
        self.ax.set_title(self.airfoil.fullname if hasattr(self.airfoil, "fullname") else "Airfoil")
        plt.axis("equal")
        self.upper_surface, self.lower_surface, self.camber_line
        self.ax.legend(loc="best")
        plt.show()

# TODO: the coordinates should come from the airfoil, keeping the x-locations
# TODO:   separate might be dangerous.
    # @cached_property
    @property
    def x_cos(self) -> np.ndarray:
        """ X-locations for the airfoil coordinates with cosine spacing
        Returns:
            list:  100 pts cosine spacing
        """
        # return 0.5 * (1 - np.cos(np.linspace(0, np.pi, num=self.n_points)))
        return cosine_spacing(0, 1, self.n_points)

    # @cached_property
    @property
    def x_coor(self) -> np.ndarray:
        """ X-coordinates from trailing edge to leading edge over the upper
        surface, then back to trailing edge along the lower surface.

        Returns:
            list:  199 pts along entire airfoil surface with cosine spacing
        """
        return np.append(self.x_cos[::-1], self.x_cos[1:])

    @property
    def upper_surface(self):
        if self.elements.upper_surface_line:
            self.elements.upper_surface_line.remove()
            self.elements.upper_surface_line = None
        else:
            self.elements.upper_surface_line = self.ax.plot(
                self.x_cos, self.airfoil.upper_surface_at(self.x_cos),
                label="Upper Surface",
                color=self.colors[0]
                )[0]
            # self.ax.add_line(self.elements.upper_surface_line)
        self.ax.legend()

    @property
    def lower_surface(self):
        if self.elements.lower_surface_line:
            self.elements.lower_surface_line.remove()
            self.elements.lower_surface_line = None
        else:
            self.elements.lower_surface_line = plt.plot(
                self.x_cos, self.airfoil.lower_surface_at(self.x_cos),
                label="Lower Surface",
                color=self.colors[1]
                )[0]
            # self.ax.add_patch(self.elements.lower_surface_line)
        self.ax.legend()

    @property
    def camber_line(self):
        if self.elements.camber_line:
            self.elements.camber_line.remove()
            self.elements.camber_line = None
        else:
            self.elements.camber_line = plt.plot(
                self.x_cos, self.airfoil.camberline_at(self.x_cos),
                label="Camber Line",
                color=self.colors[2]
                )[0]
            # self.ax.add_line(self.elements.camber_line)
        self.ax.legend()

    @property
    def surface(self):
        if self.elements.surface_line:
            self.elements.surface_line.remove()
            self.elements.surface_line = None
        else:
            u1 = cosine_spacing(0, self.airfoil.u_leading_edge, self.n_points)
            u2 = cosine_spacing(self.airfoil.u_leading_edge, 1, self.n_points)
            u = np.append(u1[:-1], u2)
            x, y = self.airfoil.surface.evaluate_at(u).T
            self.elements.surface_line = self.ax.plot(
                x, y, label="Airfoil Surface",
                color=self.colors[3]
                )[0]
            # self.ax.add_line(self.elements.surface_line)
        self.ax.legend()

    @property
    def points(self):
        if self.elements.points:
            self.elements.points.remove()
            self.elements.points = None
        else:
            pts = self.airfoil.points
            self.elements.points = plt.plot(
                pts[:, 0], pts[:, 1], 'x', label="Airfoil Coordinate Points",
                color=self.colors[4]
                )[0]
            # self.ax.add_line(self.elements.points)
        self.ax.legend()

    @property
    def unprocessed_points(self):
        if self.elements.unprocessed_points:
            self.elements.unprocessed_points.remove()
            self.elements.unprocessed_points = None
        elif hasattr(self.airfoil, "unprocessed_points"):
            pts = self.airfoil.unprocessed_points
            self.elements.unprocessed_points = plt.plot(
                pts[:, 0], pts[:, 1], 'x', label="Unprocessed Airfoil Coordinate",
                color=self.colors[4]
                )[0]
            # self.ax.add_line(self.elements.unprocessed_points)
        else:
            print("No unprocessed points found")
        self.ax.legend()

    @property
    def control_points(self):
        if self.elements.control_points:
            self.elements.control_points.remove()
            self.elements.control_points = None
        elif hasattr(self.airfoil.surface, "control_points"):
            pts = self.airfoil.surface.control_points
            self.elements.control_points = plt.plot(
                pts[:, 0], pts[:, 1], 'bo-', label="Bezier Control Points",
                )[0]
            # self.ax.add_line(self.elements.control_points)
        else:
            print("No control points found")
        self.ax.legend()

    @property
    def upper_control_points(self):
        if self.elements.upper_control_points:
            self.elements.upper_control_points.remove()
            self.elements.upper_control_points = None
        elif hasattr(self.airfoil.upper_surface, "control_points"):
            pts = self.airfoil.upper_surface.control_points
            self.elements.upper_control_points = plt.plot(
                pts[:, 0], pts[:, 1], 'bo-', label="Bezier Control Points",
                )[0]
            # self.ax.add_line(self.elements.control_points)
        else:
            print("No control points found")
        self.ax.legend()

    @property
    def lower_control_points(self):
        if self.elements.lower_control_points:
            self.elements.lower_control_points.remove()
            self.elements.lower_control_points = None
        elif hasattr(self.airfoil.lower_surface, "control_points"):
            pts = self.airfoil.lower_surface.control_points
            self.elements.lower_control_points = plt.plot(
                pts[:, 0], pts[:, 1], 'bo-', label="Bezier Control Points",
                )[0]
            # self.ax.add_line(self.elements.control_points)
        else:
            print("No control points found")
        self.ax.legend()


    @property
    def leading_edge_point(self):
        if self.elements.leading_edge_point:
            self.elements.leading_edge_point.remove()
            self.elements.leading_edge_point = None
        else:
            x, y = self.airfoil.leading_edge_point
            self.elements.leading_edge_point = plt.plot(
                x, y, 'x', label="Airfoil leading edge point",
                color='blue' #self.colors[4]
                )[0]
            # self.ax.add_line(self.elements.points)
        self.ax.legend()

    @property
    def leading_edge_radius(self):
        """Leading edge radius plot consists of several elements:
            - the leading edge itself
            - the leading edge radius circle
            - leading edge circle centroid
            - leading edge radius itself connecting leading edge with centroid
        """
        if self.elements.leading_edge_radius:
            # self.elements.LE_point.remove()
            self.elements.leading_edge_radius_circle.remove()
            # self.elements.leading_edge_radius_c.remove()
            self.elements.leading_edge_radius.remove()
            self.elements.leading_edge_radius = None
        else:
            le = self.airfoil.surface.evaluate_at(self.airfoil.u_leading_edge)
            radius = self.airfoil.surface.radius_at(self.airfoil.u_leading_edge)
            center = le + radius * self.airfoil.surface.normal_at(self.airfoil.u_leading_edge)[0]
            self.elements.leading_edge_radius_circle = plt.Circle(center, radius,
                color="r", linestyle="--", linewidth=1.5, fill=False,
                label=f"Leading Edge Radius {radius:.2e}",
                )
            self.ax.add_patch(self.elements.leading_edge_radius_circle)
            xs, ys = np.column_stack((le, center))
            self.elements.leading_edge_radius = self.ax.plot(xs, ys, 'r--o')[0]#, label="Leading Edge Radius")[0]
            # self.elements.leading_edge_radius_c = self.ax.plot(center[0], center[1], 'go')[0]#, label="Leading Edge Radius")[0]
            # self.elements.leading_edge_radius = self.ax.plot(np.column_stack((le, center))[0], np.column_stack((le, center))[1], 'g--', label=f"Leading Edge Radius {radius:.2e}")[0]
            # self.elements.LE_point = self.ax.plot(le[0], le[1], 'ro', label="Leading Edge")[0]
        self.ax.legend()

    @property
    def max_curvature(self):
        """Leading edge radius plot consists of several elements:
            - the leading edge itself
            - the leading edge radius circle
            - leading edge circle centroid
            - leading edge radius itself connecting leading edge with centroid
        """
        if self.elements.max_curvature:
            self.elements.max_curvature_circle.remove()
            self.elements.max_curvature.remove()
            self.elements.max_curvature = None
        else:
            u, c = self.airfoil.surface.max_curvature
            radius = 1/c
            point = self.airfoil.surface.evaluate_at(u)
            center = point + radius * self.airfoil.surface.normal_at(u)[0]
            self.elements.max_curvature_circle = plt.Circle(center, radius,
                color="grey", linestyle="--", linewidth=1.5, fill=False,
                # label="Maximum Surface Curvature",
                )
            self.ax.add_patch(self.elements.max_curvature_circle)
            xs, ys = np.column_stack((point, center))
            self.elements.max_curvature = self.ax.plot(
                xs, ys,
                color='grey', marker='o',
                label=f"Maximum Curvature - R={radius:.2e}"
            )[0]
        self.ax.legend()

    @property
    def max_camber(self):
        if self.elements.max_camber:
            self.elements.max_camber.remove()
            self.elements.max_camber = None
        else:
            u, c = self.airfoil.max_camber
            x, y = self.airfoil.camber_line.evaluate_at(u)
            self.elements.max_camber = plt.plot(
                [x, x], [0,y], '-*', label=f"Maximum Camber {c:.2e}",
                color=self.colors[5]
                )[0]
            # self.ax.add_line(self.elements.max_camber)
        self.ax.legend()

    @property
    def max_thickness(self):
        if self.elements.max_thickness:
            self.elements.max_thickness.remove()
            self.elements.max_thickness = None
        else:
            u, t = self.airfoil.max_thickness_spline
            x, y = self.airfoil.surface.evaluate_at(u).T
            # x, t = self.airfoil.max_thickness
            # y = self.airfoil.lower_surface_at(x)
            self.elements.max_thickness = plt.plot(
                x, y, '-*', label=f"Maximum Thickness {t:.2e}",
                # [x, x], [y, y+t], '-*', label=f"Maximum Thickness {t:.2e}",
                color=self.colors[6]
                )[0]
        self.ax.legend()

    @property
    def upper_crest(self):
        if self.elements.upper_crest:
            self.elements.upper_crest.remove()
            self.elements.upper_crest = None
        else:
            # x, y = self.airfoil.upper_crest(output="x")
            _, [x, y] = self.airfoil.upper_surface.crest
            self.elements.upper_crest = plt.plot(
                [x, x], [0,y], '-_', label=f"Upper Crest {y:.2e}",
                color="k"
                )[0]
        self.ax.legend()

    @property
    def lower_crest(self):
        if self.elements.lower_crest:
            self.elements.lower_crest.remove()
            self.elements.lower_crest = None
        else:
            # x, y = self.airfoil.lower_crest(output="x")
            _, [x, y] = self.airfoil.lower_surface.crest
            self.elements.lower_crest = plt.plot(
                [x, x], [0,y], '-_', label=f"Lower Crest {y:.2e}",
                color="k"
                )[0]
        self.ax.legend()

    @property
    def upper_crest_curvature(self):
        """Upper crest curvature plot consists of several elements:
            - the upper crest itself
            - the upper crest curvature circle
            - upper crest circle centroid
            - upper crest curvature itself connecting upper crest with centroid
        """
        if self.elements.upper_crest_curvature:
            self.elements.upper_crest_curvature.remove()
            self.elements.upper_crest_curvature = None
        else:
            u, xy = self.airfoil.upper_surface.crest
            curvature = abs(self.airfoil.upper_surface.curvature_at(u))
            radius = 1/curvature
            self.elements.upper_crest_curvature = Arc(xy - [0, radius], 2*radius, 2*radius, angle=90, theta1=-15, theta2=15,
                color="r", linestyle="--", linewidth=1.5,
                label=f"Upper Crest Curvature {curvature:.2e}",
                )
            self.ax.add_patch(self.elements.upper_crest_curvature)
        self.ax.legend()

    @property
    def lower_crest_curvature(self):
        """lower crest curvature plot consists of several elements:
            - the lower crest itself
            - the lower crest curvature circle
            - lower crest circle centroid
            - lower crest curvature itself connecting lower crest with centroid
        """
        if self.elements.lower_crest_curvature:
            self.elements.lower_crest_curvature.remove()
            self.elements.lower_crest_curvature = None
        else:
            u, xy = self.airfoil.lower_surface.crest
            curvature = abs(self.airfoil.lower_surface.curvature_at(u))
            radius = 1/curvature
            self.elements.lower_crest_curvature = Arc(xy + [0, radius], 2*radius, 2*radius, angle=-90, theta1=-15, theta2=15,
                color="r", linestyle="--", linewidth=1.5,
                label=f"Lower Crest Curvature {curvature:.2e}",
                )
            self.ax.add_patch(self.elements.lower_crest_curvature)
        self.ax.legend()

    @property
    def leading_edge_angle(self):
        if self.elements.leading_edge_angle:
            self.elements.leading_edge_angle.remove()
            self.elements.leading_edge_angle = None
        else:
            vect = self.airfoil.leading_edge_vect
            angle = self.airfoil.leading_edge_angle
            a = np.array([0, 0])
            b = (a + vect) * 20
            x, y = np.column_stack([a, b])
            self.elements.leading_edge_angle = plt.plot(
                x, y, label=f"Leading Edge Angle {angle:.2e}deg",
                color=self.colors[7]
                )[0]
        self.ax.legend()

    @property
    def trailing_edge_angle(self):
        if self.elements.trailing_edge_angle:
            self.elements.trailing_edge_angle.remove()
            self.elements.trailing_edge_angle = None
        else:
            vect = self.airfoil.trailing_edge_vect
            angle = self.airfoil.trailing_edge_deflection_angle
            a = np.array([1, 0])
            b = a - vect * 10
            x, y = np.column_stack([a, b])
            self.elements.trailing_edge_angle = plt.plot(
                x, y, label=f"Trailing Edge Angle {angle:.2e}deg",
                color='k'
                )[0]
        self.ax.legend()

    @property
    def trailing_edge_wedgedge(self):
        if self.elements.trailing_edge_wedgedge:
            for line in self.elements.trailing_edge_wedgedge:
                line.remove()
            self.elements.trailing_edge_wedgedge = []
        else:
            vect_u = self.airfoil.trailing_edge_upper_vect
            vect_l = self.airfoil.trailing_edge_lower_vect
            angle = self.airfoil.trailing_edge_wedge_angle
            a = np.array([1, 0])
            b = a - vect_u * 6
            x, y = np.column_stack([a, b])
            self.elements.trailing_edge_wedgedge.append(
                plt.plot(
                    x, y, label=f"Trailing Edge Wedge Angle {angle:.2e}deg",
                    color=self.colors[9]
                )[0]
            )
            b = a - vect_l * 6
            x, y = np.column_stack([a, b])
            self.elements.trailing_edge_wedgedge.append(
                plt.plot(
                    x, y,
                    color=self.colors[9]
                )[0]
            )
        self.ax.legend()

    @property
    def upper_curvature(self):
        if self.elements.upper_curvature:
            for line in self.elements.upper_curvature:
                line.remove()
            self.elements.upper_curvature = []
        else:
            for u in np.linspace(0, 1, num=100):
                point = self.airfoil.upper_surface.evaluate_at(u)
                c = self.airfoil.upper_surface.curvature_at(u)
                radius = c/200#1/c
                center = point + radius * self.airfoil.upper_surface.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.upper_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Upper Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    @property
    def lower_curvature(self):
        if self.elements.lower_curvature:
            for line in self.elements.lower_curvature:
                line.remove()
            self.elements.lower_curvature = []
        else:
            for u in np.linspace(0, 1, num=100):
                point = self.airfoil.lower_surface.evaluate_at(u)
                c = self.airfoil.lower_surface.curvature_at(u)
                radius = c/200#1/c
                center = point + radius * self.airfoil.lower_surface.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.lower_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Lower Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    @property
    def surface_curvature(self):
        if self.elements.surface_curvature:
            for line in self.elements.surface_curvature:
                line.remove()
            self.elements.surface_curvature = []
        else:
            for u in np.linspace(0, 1, num=200):
                point = self.airfoil.surface.evaluate_at(u)
                c = self.airfoil.surface.curvature_at(u)
                radius = c/200#1/c
                center = point + radius * self.airfoil.surface.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.surface_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Camber Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    @property
    def camber_curvature(self):
        if self.elements.camber_curvature:
            for line in self.elements.camber_curvature:
                line.remove()
            self.elements.camber_curvature = []
        else:
            for u in np.linspace(0, 1, num=100):
                point = self.airfoil.camber_line.evaluate_at(u)
                c = self.airfoil.camber_line.curvature_at(u)
                radius = c/200#1/c
                center = point + radius * self.airfoil.camber_line.normal_at(u)[0]
                xs, ys = np.column_stack((point, center))
                self.elements.camber_curvature.append(
                    self.ax.plot(
                        xs, ys,
                        color='gray',
                        label=f"Camber Curvature distr." if u==0 else None
                    )[0]
                )
        self.ax.legend()

    def reset(self):
        delattr(self.airfoil, "plot")

    def plot_all(self):
        for prop in self.elements.__dict__.keys():
            getattr(self, prop)
        # self.leading_edge_radius
        # self.leading_edge_angle
        # self.max_camber
        # self.max_thickness
        # self.max_curvature
        # self.trailing_edge_angle
        # self.trailing_edge_wedgedge

