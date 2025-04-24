import os

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import cosine
from tqdm import tqdm


def spectral_residual_saliency(image):
    """
    Compute the spectral residual saliency map for an image.
    Based on the paper "Saliency Detection: A Spectral Residual Approach" by Hou and Zhang.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize for faster processing
    height, width = gray.shape
    size = min(width, height)
    gray = cv2.resize(gray, (size, size))

    # FFT transform and get log-spectrum
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log(magnitude + 1e-9)

    # Get spectral residual
    spectral_residual = magnitude - cv2.boxFilter(magnitude, -1, (3, 3))

    # Back to spatial domain
    dft_shift[:, :, 0] = np.cos(spectral_residual) * dft_shift[:, :, 0]
    dft_shift[:, :, 1] = np.sin(spectral_residual) * dft_shift[:, :, 1]
    idft_shift = np.fft.ifftshift(dft_shift)
    inverse_dft = cv2.idft(idft_shift)

    # Get magnitude and normalize
    saliency = cv2.magnitude(inverse_dft[:, :, 0], inverse_dft[:, :, 1])
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)

    # Smooth with Gaussian filter
    saliency = ndimage.gaussian_filter(saliency, sigma=2.5)

    # Resize back to original size
    saliency = cv2.resize(saliency, (width, height))

    return saliency


def itti_koch_saliency(image):
    """
    A simplified implementation of the Itti-Koch-Niebur saliency model.
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Create Gaussian pyramids
    l_pyr = [l]
    a_pyr = [a]
    b_pyr = [b]

    for i in range(5):
        l_pyr.append(cv2.pyrDown(l_pyr[i]))
        a_pyr.append(cv2.pyrDown(a_pyr[i]))
        b_pyr.append(cv2.pyrDown(b_pyr[i]))

    # Compute feature maps
    intensity_map = cv2.absdiff(l_pyr[2], cv2.pyrUp(cv2.pyrUp(l_pyr[4])))
    rg_map = cv2.absdiff(a_pyr[2], cv2.pyrUp(cv2.pyrUp(a_pyr[4])))
    by_map = cv2.absdiff(b_pyr[2], cv2.pyrUp(cv2.pyrUp(b_pyr[4])))

    # Normalize maps
    intensity_map = cv2.normalize(intensity_map, None, 0, 255, cv2.NORM_MINMAX)
    rg_map = cv2.normalize(rg_map, None, 0, 255, cv2.NORM_MINMAX)
    by_map = cv2.normalize(by_map, None, 0, 255, cv2.NORM_MINMAX)

    # Combine maps
    saliency = cv2.addWeighted(intensity_map, 0.4, cv2.addWeighted(rg_map, 0.3, by_map, 0.3, 0), 0.6, 0)

    # Final normalization and Gaussian blur
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)

    # Resize to original image size
    saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))

    return saliency


def measure_camouflage(image, bbox, method='spectral', vis=False):
    """
    Measure how well an object within a bounding box is camouflaged.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image
    bbox : tuple
        Bounding box coordinates (x, y, width, height)
    method : str
        Saliency method to use ('spectral' or 'itti')

    Returns:
    --------
    metrics : dict
        Dictionary containing camouflage metrics
    visualizations : dict
        Dictionary containing visualization images
    """
    # Generate saliency map
    if method == 'spectral':
        saliency_map = spectral_residual_saliency(image)
    elif method == 'itti':
        saliency_map = itti_koch_saliency(image)
    else:
        raise ValueError("Method must be 'spectral' or 'itti'")

    # Extract bounding box coordinates
    x, y, w, h = bbox

    # Create masks for animal and background regions
    animal_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    animal_mask[y:y + h, x:x + w] = 1

    # Define background region (area around bounding box)
    expand_factor = 2.0
    bg_x = max(0, int(x - w * (expand_factor - 1) / 2))
    bg_y = max(0, int(y - h * (expand_factor - 1) / 2))
    bg_w = min(image.shape[1] - bg_x, int(w * expand_factor))
    bg_h = min(image.shape[0] - bg_y, int(h * expand_factor))

    background_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    background_mask[bg_y:bg_y + bg_h, bg_x:bg_x + bg_w] = 1
    background_mask[y:y + h, x:x + w] = 0  # Exclude the animal region

    # Extract image regions
    animal_region = image.copy()
    animal_region[animal_mask == 0] = 0

    background_region = image.copy()
    background_region[background_mask == 0] = 0

    # 1. Saliency-based metrics
    animal_saliency = np.mean(saliency_map[animal_mask == 1])
    background_saliency = np.mean(saliency_map[background_mask == 1])
    max_saliency = np.max(saliency_map)

    # 2. Color histogram comparison
    animal_hist = cv2.calcHist([image], [0, 1, 2], animal_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    animal_hist = cv2.normalize(animal_hist, animal_hist).flatten()

    bg_hist = cv2.calcHist([image], [0, 1, 2], background_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    bg_hist = cv2.normalize(bg_hist, bg_hist).flatten()

    hist_similarity = 1 - cosine(animal_hist, bg_hist)

    # 3. Edge contrast
    if len(image.shape) == 3:
        animal_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        animal_gray = image.copy()

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(animal_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(animal_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    animal_edge_magnitude = np.mean(edge_magnitude[animal_mask == 1])
    background_edge_magnitude = np.mean(edge_magnitude[background_mask == 1])
    edge_ratio = animal_edge_magnitude / (background_edge_magnitude + 1e-10)

    # 4. Texture similarity using LBP
    radius = 1
    n_points = 8 * radius
    animal_gray_region = animal_gray.copy()
    animal_gray_region[animal_mask == 0] = 0
    bg_gray_region = animal_gray.copy()
    bg_gray_region[background_mask == 0] = 0

    animal_lbp = local_binary_pattern(animal_gray, n_points, radius, method='uniform')
    animal_lbp = animal_lbp * animal_mask  # Apply mask

    bg_lbp = local_binary_pattern(animal_gray, n_points, radius, method='uniform')
    bg_lbp = bg_lbp * background_mask  # Apply mask

    # Calculate LBP histograms
    animal_lbp_hist, _ = np.histogram(animal_lbp[animal_mask == 1], bins=n_points + 2, range=(0, n_points + 2),
                                      density=True)
    bg_lbp_hist, _ = np.histogram(bg_lbp[background_mask == 1], bins=n_points + 2, range=(0, n_points + 2),
                                  density=True)

    texture_similarity = 1 - cosine(animal_lbp_hist, bg_lbp_hist)

    # Collect all metrics
    metrics = {
        "saliency_ratio": animal_saliency / (background_saliency + 1e-10),
        "saliency_contrast": abs(animal_saliency - background_saliency),
        "normalized_saliency": animal_saliency / (max_saliency + 1e-10),
        "color_hist_similarity": hist_similarity,
        "edge_ratio": edge_ratio,
        "texture_similarity": texture_similarity,
        # Derived camouflage score (lower = better camouflaged)
        "camouflage_score": (animal_saliency / (background_saliency + 1e-10) +
                             (1 - hist_similarity) +
                             edge_ratio +
                             (1 - texture_similarity)) / 4
    }
    if vis:
        # Create visualizations
        # Highlight bounding box
        vis_bbox = image.copy()
        cv2.rectangle(vis_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Create saliency heatmap visualization
        saliency_colored = cv2.applyColorMap(saliency_map.astype(np.uint8), cv2.COLORMAP_JET)
        saliency_blend = cv2.addWeighted(image, 0.7, saliency_colored, 0.3, 0)

        # Create background mask visualization
        bg_mask_vis = image.copy()
        overlay = np.zeros_like(image)
        overlay[background_mask == 1] = [0, 0, 255]  # Blue for background
        overlay[animal_mask == 1] = [0, 255, 0]  # Green for animal
        bg_mask_vis = cv2.addWeighted(bg_mask_vis, 0.7, overlay, 0.3, 0)

        visualizations = {
            "original_with_bbox": vis_bbox,
            "saliency_map": saliency_map,
            "saliency_heatmap": saliency_blend,
            "region_masks": bg_mask_vis
        }

        return metrics, visualizations

    return metrics, None


def display_results(metrics, visualizations):
    """Display the camouflage analysis results"""
    fig = plt.figure(figsize=(15, 10))

    # Plot images
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(visualizations["original_with_bbox"], cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Bounding Box")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(visualizations["saliency_map"], cmap='jet')
    plt.title("Saliency Map")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(visualizations["saliency_heatmap"], cv2.COLOR_BGR2RGB))
    plt.title("Saliency Heatmap Overlay")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(visualizations["region_masks"], cv2.COLOR_BGR2RGB))
    plt.title("Animal (green) and Background (blue) Regions")
    plt.axis('off')

    # Plot metrics
    plt.tight_layout()

    # Create a new figure for metrics
    fig2 = plt.figure(figsize=(10, 6))
    metrics_to_plot = {k: v for k, v in metrics.items() if k != "camouflage_score"}
    bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=0)

    plt.title(f"Camouflage Metrics (Overall Score: {metrics['camouflage_score']:.2f})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()

    # Print interpretation
    print("\nCamouflage Analysis Results:")
    print(f"Overall Camouflage Score: {metrics['camouflage_score']:.2f} (Lower = Better Camouflaged)")
    print("\nMetric Interpretation:")
    print(f"  - Saliency Ratio: {metrics['saliency_ratio']:.2f} (Lower = Better Camouflaged)")
    print(f"  - Color Similarity: {metrics['color_hist_similarity']:.2f} (Higher = Better Camouflaged)")
    print(f"  - Edge Ratio: {metrics['edge_ratio']:.2f} (Lower = Better Camouflaged)")
    print(f"  - Texture Similarity: {metrics['texture_similarity']:.2f} (Higher = Better Camouflaged)")

    if metrics['camouflage_score'] < 1.0:
        print("\nInterpretation: The animal is well camouflaged in its environment.")
    elif metrics['camouflage_score'] < 1.5:
        print("\nInterpretation: The animal has moderate camouflage in its environment.")
    else:
        print("\nInterpretation: The animal is poorly camouflaged and stands out from its environment.")


def analyze_camouflage_for_dataset(annotations_df, method='spectral', sample_size=None):
    """
    Run camouflage analysis on all annotations in the DataFrame

    Parameters:
    -----------
    annotations_df : pandas.DataFrame
        DataFrame with one row per annotation
    method : str
        Saliency method to use ('spectral' or 'itti')
    sample_size : int or None
        If provided, analyze only a sample of the annotations

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with added camouflage metrics columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    results_df = annotations_df.copy()

    # Add columns for the camouflage metrics
    metrics_columns = [
        'saliency_ratio', 'saliency_contrast', 'normalized_saliency',
        'color_hist_similarity', 'edge_ratio', 'texture_similarity',
        'camouflage_score'
    ]

    for col in metrics_columns:
        results_df[col] = None

    # Add a column for error tracking
    results_df['analysis_error'] = None

    # Sample if requested
    if sample_size and sample_size < len(results_df):
        analyze_df = results_df.sample(sample_size, random_state=42)
        indices = analyze_df.index
    else:
        analyze_df = results_df
        indices = results_df.index

    # Process each annotation
    for idx in tqdm(indices, desc="Analyzing camouflage"):
        row = results_df.loc[idx]

        # Skip if file doesn't exist
        if not os.path.exists(row['image_path']):
            results_df.loc[idx, 'analysis_error'] = "File not found"
            continue

        try:
            # Load image
            image = cv2.imread(row['image_path'])

            if image is None:
                results_df.loc[idx, 'analysis_error'] = "Failed to load image"
                continue

            # Extract bbox
            bbox = (int(row['bbox_x']), int(row['bbox_y']),
                    int(row['bbox_width']), int(row['bbox_height']))

            # Run camouflage analysis
            metrics, _ = measure_camouflage(image, bbox, method=method)

            # Store metrics in DataFrame
            for metric_name, metric_value in metrics.items():
                results_df.loc[idx, metric_name] = metric_value

        except Exception as e:
            results_df.loc[idx, 'analysis_error'] = str(e)

    return results_df


# Example usage
if __name__ == "__main__":

    # Example with a butterfly image
    # image_path = "/Users/christian/data/training_data/2025_02_22_HIT/03_all_other/train/crops_1024/FMO03___DJI_0483_x1024_y2048.jpg"
    bbox = (729, 253, 50, 50)  # Adjust these coordinates to match your butterfly location

    image_path = "/Users/christian/data/training_data/2025_02_22_HIT/03_all_other/train/crops_640/FMO03___DJI_0483_x3840_y2560.jpg"
    bbox = (436, 330, 150, 130)  # Adjust these coordinates to match your butterfly location

    WAID_train = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/waid_coco/train')
    WAID_annotations = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/waid_coco/train/_annotations.coco.json')



    # Load the image
    try:
        image = cv2.imread(image_path)

        # Run the camouflage analysis
        metrics, visualizations = measure_camouflage(image, bbox, method='spectral')
        # Display the results
        display_results(metrics, visualizations)

        # Run the camouflage analysis
        metrics, visualizations = measure_camouflage(image, bbox, method='itti')
        # Display the results
        display_results(metrics, visualizations)

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your image path and bounding box coordinates.")

# Sample input for interactive testing:

