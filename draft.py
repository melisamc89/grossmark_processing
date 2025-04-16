import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import viridis

# Example data points in 3D space
points = umap_emb_all
#points = umap_emb_all[1000:9000,:]


# Example flow magnitudes between consecutive points
differences = np.diff(points, axis=0)
flow_intensities = np.linalg.norm(differences, axis=1)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
labels = position


N= points.shape[0]
#ax.scatter(points[:N,0], points[:N,1], points[:N,2], c=labels[:N], s=1, alpha=0.5, cmap='viridis')

# Normalize differences to get unit vectors
unit_vectors = differences#/ np.linalg.norm(differences, axis=1)[:, np.newaxis]
# Normalize these labels to use with the colormap
x = flow_intensities
norm = Normalize(vmin=labels.min(), vmax=labels.max())

color_mapper = viridis

# Adding arrows with color mapped from predefined labels
for i in range(len(flow_intensities)):
    color = color_mapper(norm(labels[i]))
    #color = color_mapper(norm([i]))
    ax.quiver(points[i,0], points[i,1], points[i,2],
              unit_vectors[i,0], unit_vectors[i,1], unit_vectors[i,2],
              length=flow_intensities[i],
              #arrow_length_ratio=0.5,
              color=color)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)

# Add a color bar to show label mapping
sm = plt.cm.ScalarMappable(cmap=color_mapper, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Position')
# Show the plot
plt.show()