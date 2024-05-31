import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from scipy.special import comb
import matplotlib.pyplot as plt
from numfoil.geometry.airfoil import ProcessedFileAirfoil




def cosine_spacing(n_points):
    """ Generate cosine-spaced points between 0 and 1. """
    return (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2

def bezier_curve(control_points, num_points=200):
    """ Generate Bezier curve points from fixed x control points and variable y control points. """
    t = np.linspace(0, 1, num_points)
    n = len(control_points) - 1
    curve_points = np.zeros((num_points, 2))
    for i in range(n + 1):
        binomial_coeff = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
        curve_points[:, 0] += binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * control_points[i, 0]
        curve_points[:, 1] += binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * control_points[i, 1]
    return curve_points

def objective_function(y_values, x_control_points, airfoil_points, degree):
    """ Compute the sum of squared distances from airfoil points to the Bezier curve. """
    control_points = np.column_stack((x_control_points, y_values))
    curve_points = bezier_curve(control_points)
    tree = KDTree(curve_points)
    distances, _ = tree.query(airfoil_points)
    return np.sum(distances**2)

def fit_airfoil(airfoil_points, x_control_points, initial_y_values, degree):
    """ Fit Bezier curve to airfoil points using least squares optimization. """
    result = least_squares(objective_function, initial_y_values, args=(x_control_points, airfoil_points, degree))
    optimized_y_values = result.x
    return optimized_y_values

# Example usage
n_control_points = 5
degree = n_control_points - 1
x_control_points = cosine_spacing(n_control_points)  # Change to linear_spacing(n_control_points) if needed

# Mock data for either upper or lower surface of an airfoil
airfoil_points = np.random.rand(100, 2)  # This should be replaced with actual airfoil data points
initial_y_values = np.random.rand(n_control_points)  # Initial y-values for control points

# Fit the airfoil
optimized_y_values = fit_airfoil(airfoil_points, x_control_points, initial_y_values, degree)

# Create control points and plot
control_points = np.column_stack((x_control_points, optimized_y_values))
curve_points = bezier_curve(control_points)

plt.figure(figsize=(10, 5))
plt.scatter(airfoil_points[:, 0], airfoil_points[:, 1], color='blue', label='Airfoil Data Points')
plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-', label='Fitted Bezier Curve')
plt.scatter(control_points[:, 0], control_points[:, 1], color='green', label='Optimized Control Points')
plt.legend()
plt.title('Optimized Bezier Curve for Airfoil Surface')
plt.show()

1==1

# # Define the Bernstein polynomial basis function
# def bernstein_poly(i, n, t):
#     return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

# # Compute the Bezier curve points using Bernstein polynomials
# def bezier_curve(control_points, n_points=200):
#     n = len(control_points) - 1
#     t = np.linspace(0, 1, n_points)
#     curve = np.zeros((n_points, 2))
#     for i in range(n + 1):
#         curve += np.outer(bernstein_poly(i, n, t), control_points[i])
#     return curve

# # Objective function for least-squares optimization
# def objective_function(control_points, target_points):
#     curve_points = bezier_curve(control_points)
#     kdtree = KDTree(curve_points)
#     distances, _ = kdtree.query(target_points)
#     return distances

# # Function to optimize Bezier control points
# def optimize_bezier(control_points, target_points):
#     result = least_squares(objective_function, control_points.ravel(), args=(target_points,))
#     optimized_control_points = result.x.reshape(-1, 2)
#     return optimized_control_points

# # Example usage
# if __name__ == "__main__":
#     # Initial Bezier control points (example)
#     initial_control_points = np.array([[0, 0], [0.5, 1], [1, 1], [1.5, 0], [2, -0.5], [3, 0]])

#     airfoil = ProcessedFileAirfoil("C:\\Projects\\VAE\src\\data\\UIUC_airfoils\\be50sm.dat")
#     airfoil.plotter


#     # Airfoil data points (example)
#     airfoil_data_points = airfoil.points

#     # Optimize the control points
#     optimized_control_points = optimize_bezier(initial_control_points, airfoil_data_points)

#     # Plot the results
#     initial_curve = bezier_curve(initial_control_points)
#     optimized_curve = bezier_curve(optimized_control_points)

#     plt.figure(figsize=(10, 6))
#     plt.plot(initial_curve[:, 0], initial_curve[:, 1], 'r--', label='Initial Bezier Curve')
#     plt.plot(optimized_curve[:, 0], optimized_curve[:, 1], 'b-', label='Optimized Bezier Curve')
#     plt.plot(airfoil_data_points[:, 0], airfoil_data_points[:, 1], 'go', label='Airfoil Data Points')
#     plt.plot(initial_control_points[:, 0], initial_control_points[:, 1], 'ro-', label='Initial Control Points')
#     plt.plot(optimized_control_points[:, 0], optimized_control_points[:, 1], 'bo-', label='Optimized Control Points')
#     plt.legend()
#     plt.title('Bezier Curve Optimization for Airfoil Parameterization')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.grid(True)
#     plt.show()
#     1==1
