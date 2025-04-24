"""
Read images and check the image quality
"""
import typing

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from active_learning.util.image import get_image_id



# 3. Function to calculate grid metrics for a single image
def calculate_grid_metrics(image: np.ndarray, grid_size=(10, 10), metrics=None):
    """
    Calculate quality metrics for each cell in a grid overlayed on the image.

    Args:
        image (numpy.ndarray): Input image
        grid_size (tuple, optional): Number of grid cells (rows, cols). Default is (10, 10).
        metrics (list, optional): List of metric functions to calculate.
                                Default is None (all metrics).

    Returns:
        dict: Dictionary with metric values for each grid cell
    """
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]

    # Define metrics if not provided
    if metrics is None:
        metrics = {
            'laplacian_variance': calculate_laplacian_variance,
            'gradient_magnitude': calculate_gradient_magnitude,
            'local_contrast': calculate_local_contrast,
            'tenengrad': calculate_tenengrad
        }

    # Initialize results
    results = {metric_name: np.zeros(grid_size) for metric_name in metrics.keys()}

    # Calculate metrics for each grid cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Calculate cell coordinates
            top = i * cell_h
            left = j * cell_w
            bottom = min((i + 1) * cell_h, h)
            right = min((j + 1) * cell_w, w)

            # Extract cell region
            cell = image[top:bottom, left:right]

            # Skip empty cells
            if cell.size == 0:
                continue

            # Calculate metrics
            for metric_name, metric_func in metrics.items():
                results[metric_name][i, j] = metric_func(cell)

    return results


# Helper function for parallel processing - must be at module level for multiprocessing
def process_single_image(args):
    """
    Process a single image to calculate grid metrics.

    Args:
        args (tuple): Tuple containing (image_tuple, grid_size, metrics)
            image_tuple is a tuple of (image_path, image)
            grid_size is a tuple specifying the grid dimensions
            metrics is a dictionary of metric functions

    Returns:
        dict: Dictionary with metric values for each grid cell
    """
    img_path, grid_size, metrics = args
    image_id = get_image_id(img_path)
    try:
        img = cv2.imread(img_path)
        if img is not None:
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.errror(f"Error reading {img_path}: {e}")

    metrics = calculate_grid_metrics(img, grid_size, metrics)
    return image_id, metrics

# 4. Function to process multiple images in parallel
def process_grid(grid_file_path: Path, metrics, grid_size=(10, 10)) -> pd.DataFrame:
    """
    Process multiple images and calculate grid metrics.

    Args:
        images (list): List of image_path Path
        grid_size (tuple, optional): Number of grid cells (rows, cols). Default is (10, 10).
        max_workers (int, optional): Maximum number of workers for parallel processing.
                                   Default is None (uses os.cpu_count()).

    Returns:
        dict: Dictionary with aggregated metrics
    """
    # Define metrics

    metric_dumps = pd.read_parquet(grid_file_path)

    for _, row in metric_dumps.iterrows():

        for metric_name in metrics.keys():

            metric_result = row[metric_name].reshape(grid_size)
            row[metric_name] = metric_result
    return metric_dumps

def aggregate_results(metric_dumps, metrics, grid_size=(10, 10) ) -> dict:
    """
    aggreagate results from multiple images
    :param metric_dumps:
    :param grid_size:
    :param metrics:
    :return:
    """

    # Initialize aggregated results
    aggregated_results = {
        metric_name: np.zeros(grid_size) for metric_name in metrics.keys()
    }

    # Add count for averaging
    aggregated_results['count'] = np.zeros(grid_size)
    # Aggregate results
    for _, row in metric_dumps.iterrows():
        image_id = row["image_id"]
        for metric_name in metrics.keys():
            # Add to aggregated results
            aggregated_results[metric_name] += row[metric_name]
            aggregated_results['count'] += np.ones(grid_size)

    # Calculate averages
    for metric_name in metrics.keys():
        mask = aggregated_results['count'] > 0
        aggregated_results[metric_name][mask] /= aggregated_results['count'][mask]

    return aggregated_results


# 5. Visualization functions

def plot_grid_heatmap(grid_data, metric_name, title=None, cmap='viridis'):
    """
    Plot a heatmap for a grid metric.

    Args:
        grid_data (numpy.ndarray): Grid data to plot
        metric_name (str): Name of the metric
        title (str, optional): Title for the plot. Default is None.
        cmap (str, optional): Colormap name. Default is 'viridis'.

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid_data, cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_name, rotation=-90, va="bottom")

    # Set title
    if title is None:
        title = f"{metric_name.replace('_', ' ').title()} Heatmap"
    ax.set_title(title)

    # Add grid
    ax.grid(False)
    ax.set_xticks(np.arange(grid_data.shape[1]))
    ax.set_yticks(np.arange(grid_data.shape[0]))

    # Add labels
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")

    # Return the figure
    plt.tight_layout()
    return fig


def plot_radial_profile(grid_data, metric_name, title=None):
    """
    Plot a radial profile (center to edge) for a grid metric.

    Args:
        grid_data (numpy.ndarray): Grid data to plot
        metric_name (str): Name of the metric
        title (str, optional): Title for the plot. Default is None.

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Get grid dimensions
    h, w = grid_data.shape
    center_y, center_x = h // 2, w // 2

    # Calculate distance from center for each grid cell
    y_indices, x_indices = np.indices(grid_data.shape)
    distances = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'distance': distances.flatten(),
        'value': grid_data.flatten()
    })

    # Group by distance (rounded to nearest integer)
    df['distance_bin'] = np.round(df['distance'])
    grouped = df.groupby('distance_bin')['value'].agg(['mean', 'std']).reset_index()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean values
    ax.plot(grouped['distance_bin'], grouped['mean'], marker='o', linestyle='-', color='b')

    # Add error bars
    ax.fill_between(
        grouped['distance_bin'],
        grouped['mean'] - grouped['std'],
        grouped['mean'] + grouped['std'],
        alpha=0.3,
        color='b'
    )

    # Add labels and title
    ax.set_xlabel("Distance from Center")
    ax.set_ylabel(metric_name.replace('_', ' ').title())

    if title is None:
        title = f"{metric_name.replace('_', ' ').title()} Radial Profile"
    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Return the figure
    plt.tight_layout()
    return fig


def visualize_all_metrics(aggregated_results, output_dir=None, exclude_keys=None):
    """
    Visualize all metrics from aggregated results.

    Args:
        aggregated_results (dict): Dictionary with aggregated metrics
        output_dir (str, optional): Directory to save the visualizations. Default is None (don't save).
        exclude_keys (list, optional): Keys to exclude from visualization. Default is None.

    Returns:
        dict: Dictionary with visualization figures
    """
    if exclude_keys is None:
        exclude_keys = ['count']

    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # Visualize each metric
    for metric_name, grid_data in aggregated_results.items():
        if metric_name in exclude_keys:
            continue

        # Create heatmap
        heatmap_fig = plot_grid_heatmap(grid_data, metric_name)
        figures[f"{metric_name}_heatmap"] = heatmap_fig

        # Create radial profile
        radial_fig = plot_radial_profile(grid_data, metric_name)
        figures[f"{metric_name}_radial"] = radial_fig

        # Save figures if output_dir is specified
        if output_dir is not None:
            heatmap_fig.savefig(os.path.join(output_dir, f"{metric_name}_heatmap.png"), dpi=300, bbox_inches='tight')
            radial_fig.savefig(os.path.join(output_dir, f"{metric_name}_radial.png"), dpi=300, bbox_inches='tight')

    return figures


def plot_all_heatmaps_together(aggregated_results, output_path=None, exclude_keys=None, figsize=(18, 12)):
    """
    Plot all heatmaps together in a single figure.

    Args:
        aggregated_results (dict): Dictionary with aggregated metrics
        output_path (str, optional): Path to save the visualization. Default is None (don't save).
        exclude_keys (list, optional): Keys to exclude from visualization. Default is None.
        figsize (tuple, optional): Figure size. Default is (18, 12).

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if exclude_keys is None:
        exclude_keys = ['count']

    # Get metrics to plot
    metrics = [key for key in aggregated_results.keys() if key not in exclude_keys]
    n_metrics = len(metrics)

    # Calculate grid layout
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single row/column case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1)

    # Plot each metric
    for i, metric_name in enumerate(metrics):
        ax = axes.flat[i] if i < len(axes.flat) else None

        if ax is None:
            break

        grid_data = aggregated_results[metric_name]
        im = ax.imshow(grid_data, cmap='viridis')
        ax.set_title(metric_name.replace('_', ' ').title())

        # Add colorbar
        fig.colorbar(im, ax=ax)

    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        if i < len(axes.flat):
            axes.flat[i].axis('off')

    # Add overall title
    plt.suptitle("Image Quality Metrics Across Grid", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure if output_path is specified
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_normalized_comparison(aggregated_results, output_path=None, exclude_keys=None):
    """
    Create a normalized comparison of all metrics to highlight relative quality differences.

    Args:
        aggregated_results (dict): Dictionary with aggregated metrics
        output_path (str, optional): Path to save the visualization. Default is None (don't save).
        exclude_keys (list, optional): Keys to exclude from visualization. Default is None.

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if exclude_keys is None:
        exclude_keys = ['count']

    # Get metrics to normalize
    metrics = [key for key in aggregated_results.keys() if key not in exclude_keys]

    # Create a normalized grid
    normalized_grid = np.zeros_like(aggregated_results[metrics[0]])

    # Normalize each metric and add to the normalized grid
    for metric_name in metrics:
        grid_data = aggregated_results[metric_name]

        # Normalize to [0, 1]
        min_val = np.min(grid_data)
        max_val = np.max(grid_data)
        if max_val > min_val:  # Avoid division by zero
            norm_data = (grid_data - min_val) / (max_val - min_val)
        else:
            norm_data = np.zeros_like(grid_data)

        # Add to normalized grid
        normalized_grid += norm_data

    # Average the normalized values
    normalized_grid /= len(metrics)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: red (low quality) to green (high quality)
    cmap = LinearSegmentedColormap.from_list('quality_map', ['red', 'yellow', 'green'])

    im = ax.imshow(normalized_grid, cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Quality", rotation=-90, va="bottom")

    # Set title
    ax.set_title("Normalized Quality Map (All Metrics Combined)")

    # Add grid
    ax.grid(False)

    # Add labels
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")

    # Save figure if output_path is specified
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    # Return the figure
    plt.tight_layout()
    return fig


# 6. Main function to run the analysis
def visualise_image_quality(grid_path, metrics, grid_size=(10, 10), output_dir=None):
    """
    Analyze image quality across multiple images.

    Args:
        folder_path (str): Path to the folder containing images
        grid_size (tuple, optional): Number of grid cells (rows, cols). Default is (10, 10).
        max_images (int, optional): Maximum number of images to process. Default is None (all images).
        output_dir (str, optional): Directory to save the results. Default is None (don't save).

    Returns:
        dict: Dictionary with aggregated metrics and visualizations
    """

    df_metrics = process_grid(grid_path, grid_size=grid_size, metrics=metrics)

    aggregated_results = aggregate_results(df_metrics, grid_size=grid_size, metrics=metrics)

    # 3. Visualize results
    figures = visualize_all_metrics(aggregated_results, output_dir)

    # 4. Create combined visualizations
    all_heatmaps_fig = plot_all_heatmaps_together(
        aggregated_results,
        output_path=os.path.join(output_dir, "all_heatmaps.png") if output_dir else None
    )

    normalized_fig = create_normalized_comparison(
        aggregated_results,
        output_path=os.path.join(output_dir, "normalized_quality.png") if output_dir else None
    )

    # 5. Return results
    return {
        'aggregated_results': aggregated_results,
        'figures': figures,
        'all_heatmaps_fig': all_heatmaps_fig,
        'normalized_fig': normalized_fig
    }


# Example usage
if __name__ == "__main__":
    # Replace with your image folder path

    IMAGE_FOLDER = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Fernandina")
    # IMAGE_FOLDER = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana")

    # Optional: Specify output directory
    OUTPUT_DIR = Path(f"image_quality_results_{IMAGE_FOLDER.name}")

    # Run the analysis
    results = visualise_image_quality(
        grid_path=OUTPUT_DIR / "image_quality.parquet",
        grid_size=(10, 10),  # 10x10 grid
        max_images=None,  # Process up to 100 images
        output_dir=OUTPUT_DIR,
    )

    # Show the normalized quality map
    # plt.figure(results['normalized_fig'].number)
    # plt.show()