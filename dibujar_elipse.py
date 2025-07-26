import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse(center, a, b, angle_deg, num_points=100):
    """
    Plots an ellipse defined by:
      - center: a tuple (x, y) for the ellipse center,
      - a: length of the semi-major axis,
      - b: length of the semi-minor axis,
      - angle_deg: rotation angle in degrees,
      - num_points: number of points to sample for drawing the ellipse.
    """
    # Generate points for an unrotated ellipse centered at the origin
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    
    # Convert the rotation angle from degrees to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Create the rotation matrix
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    
    # Rotate and then shift the ellipse points
    ellipse_points = np.dot(R, np.array([x, y]))
    ellipse_points[0, :] += center[0]
    ellipse_points[1, :] += center[1]
    
    # Plot the ellipse
    plt.figure(figsize=(6, 6))
    plt.plot(ellipse_points[0, :], ellipse_points[1, :], label="Ellipse")
    plt.scatter(*center, color="red", zorder=5, label="Center")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Ellipse Plot")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# Example usage:
plot_ellipse(center=(0, 0), a=5, b=3, angle_deg=90)
