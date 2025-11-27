import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist


def create_gaussian_density_map(annotations, image_shape, sigma=15):
    """Create a Gaussian density map"""
    density_map = np.zeros(image_shape)

    for x, y in annotations:
        # Create Gaussian blob for each annotation
        y_grid, x_grid = np.ogrid[0:image_shape[0], 0:image_shape[1]]
        gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        density_map += gaussian

    return density_map


def create_distance_transform(annotations, image_shape):
    """Create Euclidean distance transform"""
    # Create grid of all pixel coordinates
    y_coords, x_coords = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    pixels = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    # Calculate distance from each pixel to nearest annotation
    annotations_array = np.array(annotations)
    distances = cdist(pixels, annotations_array, metric='euclidean')
    min_distances = distances.min(axis=1)

    # Reshape back to image shape
    distance_map = min_distances.reshape(image_shape)

    return distance_map


def create_fidt_map(annotations, image_shape, alpha=0.02, beta=0.75, C=1):
    """Create Focal Inverse Distance Transform map"""
    P = create_distance_transform(annotations, image_shape)

    # FIDT formula: I = 1 / (P^(alpha*P + beta) + C)
    exponent = alpha * P + beta
    fidt_map = 1.0 / (np.power(P, exponent) + C)

    return fidt_map


# Dense region (people close together)
dense_annotations = [(50, 200), (70, 200), (90, 200), (110, 200), (130, 200)]

# Image dimensions
image_shape = (300, 400)

# Generate maps
gaussian_map = create_gaussian_density_map(dense_annotations, image_shape, sigma=15)
fidt_map = create_fidt_map(dense_annotations, image_shape)

# Dense region zoom-in
zoom_y_range = (170, 230)
zoom_x_range = (20, 170)

# Get zoomed regions
gaussian_zoom = gaussian_map[zoom_y_range[0]:zoom_y_range[1], zoom_x_range[0]:zoom_x_range[1]]
fidt_zoom = fidt_map[zoom_y_range[0]:zoom_y_range[1], zoom_x_range[0]:zoom_x_range[1]]

# Adjust annotation coordinates for zoomed view
dense_zoom_x = [x - zoom_x_range[0] for x, y in dense_annotations]
dense_zoom_y = [y - zoom_y_range[0] for x, y in dense_annotations]

# Save Gaussian map
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(gaussian_zoom, cmap='jet', aspect='auto')
ax.scatter(dense_zoom_x, dense_zoom_y, c='white', s=100, marker='x', linewidths=3)
ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig('gaussian_dense_zoom.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

# Save FIDT map
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(fidt_zoom, cmap='jet', aspect='auto')
ax.scatter(dense_zoom_x, dense_zoom_y, c='white', s=100, marker='x', linewidths=3)
ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig('fidt_dense_zoom.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print("Saved: gaussian_dense_zoom.png")
print("Saved: fidt_dense_zoom.png")
print("\nMap statistics:")
print(f"Gaussian - Max: {gaussian_zoom.max():.4f}, Min: {gaussian_zoom.min():.4f}")
print(f"FIDT - Max: {fidt_zoom.max():.4f}, Min: {fidt_zoom.min():.4f}")