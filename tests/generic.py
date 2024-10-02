import numpy as np
from utils import normalize
import matplotlib.pyplot as plt

if __name__ == "__main__":
  r = 1
  N = 100
  thetha = np.random.uniform(0, np.pi, size=N)
  phi = np.random.uniform(0, 2 * np.pi, size=N)

  x = r * np.sin(thetha) * np.cos(phi)
  y = 3.5 * r * np.sin(thetha) * np.sin(phi)
  z = 0.81 * r * np.cos(thetha)

  data = np.stack([x, y, z], axis=-1)

  # Create a figure
  fig = plt.figure()
  ax = fig.add_subplot(121, projection='3d')

  # Plot the points
  ax.scatter(data[:, 0], data[:, 1], data[:, 2])

  # Set labels
  ax.set_xlabel('X Axis')
  ax.set_ylabel('Y Axis')
  ax.set_zlabel('Z Axis')
  ax.set_aspect("equal")

  data = normalize(data)
  ax = fig.add_subplot(122, projection='3d')

  # Plot the points
  ax.scatter(data[:, 0], data[:, 1], data[:, 2])

  # Set labels
  ax.set_xlabel('X Axis')
  ax.set_ylabel('Y Axis')
  ax.set_zlabel('Z Axis')
  ax.set_aspect("equal")
  # Show the plot
  plt.show()
