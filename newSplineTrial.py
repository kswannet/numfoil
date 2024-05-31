import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev
import scipy.interpolate as si
import scipy.optimize as so

from src.data.airfoilDataset import UIUCDataset
from numfoil.geometry.airfoil import ProcessedPointsAirfoil



dataset = UIUCDataset("src\\data\\UIUC_airfoils")


airfoil = dataset.airfoils[827]
airfoil = ProcessedPointsAirfoil(airfoil.points)

n_control_points = 4
degree = n_control_points #- 1


def knot_value(combined_control_points):
    distances = np.sqrt(np.sum(np.diff(combined_control_points, axis=0)**2, axis=1))  # Compute distances between control points# Compute distances between control points
    parameters = np.concatenate(([0], np.cumsum(distances) / np.sum(distances)))           # Compute parameter values proportional to distances
    return parameters[len(parameters)//2]

def cost_function(y_control_points):
    y_control_points = np.insert(y_control_points, len(y_control_points)//2, 0)
    y_control_points = np.insert(y_control_points, 0, 0.001)
    y_control_points = np.insert(y_control_points, len(y_control_points), 0.001)
    x_control_points = np.insert(np.linspace(0, 1, n_control_points), 0, 0)
    x_control_points = np.append(x_control_points[::-1], x_control_points[1:])
    control_points = np.column_stack((x_control_points, y_control_points))
    combined_knots = np.concatenate(([0]*(degree+1), [knot_value(control_points)]*degree, [1]*(degree+1)))
    tck = np.array([combined_knots, control_points.T, degree])
    u_fine = np.linspace(0, 1, 400)
    sample_u = np.array(si.splev(np.linspace(1, knot_value(control_points), 100), tck)).T
    target_u = np.array(si.splev(np.linspace(1, airfoil.u_leading_edge, 100), airfoil.surface.spline)).T
    error_u = np.sum(np.linalg.norm(si.pchip_interpolate(sample_u[0], sample_u[1], np.linspace(0, )-target_u))
    return np.sum(np.linalg.norm(sample-target))

result = so.minimize(
    cost_function,
    np.hstack((
        [1]*(n_control_points-1),
        [-1]*(n_control_points-1)
    )),
    # bounds are 0 to 1 for the first half and -1 to 0 for the second half
    bounds=[(0, 1)]*(n_control_points-1) + [(-1, 0)]*(n_control_points-1),
    method="SLSQP"#'L-BFGS-B'
)
y_control_points = result.x
y_control_points = np.insert(y_control_points, len(y_control_points)//2, 0)
y_control_points = np.insert(y_control_points, 0, 0.0005)
y_control_points = np.insert(y_control_points, len(y_control_points), -0.0005)
x_control_points = np.insert(np.linspace(0, 1, n_control_points), 0, 0)
x_control_points = np.append(x_control_points[::-1], x_control_points[1:])
control_points = np.column_stack((x_control_points, y_control_points))
combined_knots = np.concatenate(([0]*(degree+1), [knot_value(control_points)]*degree, [1]*(degree+1)))
tck = np.array([combined_knots, control_points.T, degree])


# Plotting
plt.figure(figsize=(8, 4))
plt.plot(control_points[:, 0], control_points[:, 1], 'bo-', label='Combined Control Points')
plt.plot(splev(np.linspace(0,1,400), tck)[0], splev(np.linspace(0,1,400), tck)[1], '-k', label='Combined B-Spline')
plt.plot(airfoil.unprocessed_points.T[0], airfoil.unprocessed_points.T[1], '*', label='Original points')
plt.title('Combined B-Spline for Airfoil Surfaces using splev')
plt.legend()
plt.grid(True)
plt.show()

1==1
# Control points for the upper and lower surfaces
control_points_upper = np.array([[0, 0], [0, 1], [1, 2], [2, 1.5], [3, 0]])[::-1]
control_points_lower = np.array([[0, 0], [0, -1], [1, -0.5], [2, 0.5], [3, 0]])  # Reversed lower surface


# Degree of the B-spline
p = len(control_points_upper) - 1  # Degree is number of control points minus one, assume cubic (p=3)

# Combine control points
combined_control_points = np.concatenate((control_points_upper, control_points_lower[1:]))
# combined_control_points = np.column_stack((
#     np.concatenate((np.linspace(0,1,10)[::-1],
#                     [0],
#                     np.linspace(0,1,10)
#     )),

# ))

# Creating a single, continuous knot vector for the combined spline
# Start and end must be clamped, mid must allow continuity
num_ctrl_pts = len(combined_control_points)

"""
Number of knot points should be number of control points + degree + 1 (n+p+1).
The degree of the spline is the number of control points per segment minus 1.
To clamp the endpoints, the first and last p+1 knots must be 0 and 1,
respectively.
The knot point should repeat p times for C2 continuity.
"""
# Compute distances between control points
distances = np.sqrt(np.sum(np.diff(combined_control_points, axis=0)**2, axis=1))
# Compute parameter values proportional to distances
parameters = np.concatenate(([0], np.cumsum(distances) / np.sum(distances)))
# Compute knot value at knot point (middle of parameters)
knot_value = parameters[len(parameters)//2]
# Create knot vector with knot value at knot point
combined_knots = np.concatenate(([0]*(p+1), [knot_value]*p, [1]*(p+1)))

# combined_knots = np.array([0,0,0,0,0,0.5,0.5,0.5,0.5,1,1,1,1,1])



# Prepare the tck tuple equivalent for use with splev
tck = (combined_knots, combined_control_points.T, p)

# Evaluate the combined spline on a fine mesh using splev
t_combined = np.linspace(0, 1, 400)
combined_values = splev(t_combined, tck)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(combined_control_points[:, 0], combined_control_points[:, 1], 'bo-', label='Combined Control Points')
plt.plot(combined_values[0], combined_values[1], 'k-', label='Combined B-Spline via splev')
plt.plot(splev(np.linspace(0,1,25), tck)[0], splev(np.linspace(0,1,25), tck)[1], '*', label='spline evaluations')
plt.title('Combined B-Spline for Airfoil Surfaces using splev')
plt.legend()
plt.grid(True)
plt.show()

1==1




