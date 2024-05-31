import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from scipy.special import comb
import matplotlib.pyplot as plt
from numfoil.geometry.airfoil import ProcessedFileAirfoil

# Define the Bernstein polynomial basis function
def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

# Compute the Bezier curve points using Bernstein polynomials
def bezier_curve(control_points, n_points=200):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), control_points[i])
    return curve

# Objective function for least-squares optimization
def objective_function(control_points, target_points):
    curve_points = bezier_curve(control_points)
    kdtree = KDTree(curve_points)
    distances, _ = kdtree.query(target_points)
    return distances

# Function to optimize Bezier control points
def optimize_bezier(control_points, target_points):
    result = least_squares(objective_function, control_points.ravel(), args=(target_points,))
    optimized_control_points = result.x.reshape(-1, 2)
    return optimized_control_points

# Example usage
if __name__ == "__main__":
    # Initial Bezier control points (example)
    initial_control_points = np.array([[0, 0], [0.5, 1], [1, 1], [1.5, 0], [2, -0.5], [3, 0]])

    airfoil = ProcessedFileAirfoil("C:\\Projects\\VAE\src\\data\\UIUC_airfoils\\be50sm.dat")
    airfoil.plotter


    # Airfoil data points (example)
    airfoil_data_points = airfoil.points
    
    # Optimize the control points
    optimized_control_points = optimize_bezier(initial_control_points, airfoil_data_points)

    # Plot the results
    initial_curve = bezier_curve(initial_control_points)
    optimized_curve = bezier_curve(optimized_control_points)

    plt.figure(figsize=(10, 6))
    plt.plot(initial_curve[:, 0], initial_curve[:, 1], 'r--', label='Initial Bezier Curve')
    plt.plot(optimized_curve[:, 0], optimized_curve[:, 1], 'b-', label='Optimized Bezier Curve')
    plt.plot(airfoil_data_points[:, 0], airfoil_data_points[:, 1], 'go', label='Airfoil Data Points')
    plt.plot(initial_control_points[:, 0], initial_control_points[:, 1], 'ro-', label='Initial Control Points')
    plt.plot(optimized_control_points[:, 0], optimized_control_points[:, 1], 'bo-', label='Optimized Control Points')
    plt.legend()
    plt.title('Bezier Curve Optimization for Airfoil Parameterization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
    1==1
