import numpy as np
import typing

from loguru import logger
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import seaborn as sns
import shapely
from matplotlib import axes
from shapely import Point
from typing import List, Optional

from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage


def plot_frequency_distribution(df, columns=None, figsize=(15, 10), bins=30, kde=True):
    """
    Create frequency distribution plots for specified columns in a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    columns : list or None
        List of column names to visualize. If None, will try to use numeric columns
    figsize : tuple
        Figure size as (width, height) in inches
    bins : int
        Number of bins for the histogram
    kde : bool
        Whether to overlay a kernel density estimate

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    # If no columns specified, use numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    # Calculate number of rows and columns for subplot grid
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes array for easier iteration
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Create plots
    for i, column in enumerate(columns):
        if i < len(axes):  # Ensure we don't exceed the number of axes
            sns.histplot(data=df, x=column, kde=kde, bins=bins, ax=axes[i])
            axes[i].set_title(f'Distribution of {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_visibility_scatter(df, figsize=(18, 6)):
    """
    Create scatter plots comparing visibility with width, height, and area.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing 'width', 'height', and 'visibility' columns
    figsize : tuple
        Figure size as (width, height) in inches

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    # Calculate area if it doesn't exist
    if 'area' not in df.columns:
        df = df.copy()
        df['area'] = df['width'] * df['height']

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Width vs Visibility
    sns.scatterplot(data=df, x='width', y='visibility', ax=axes[0])
    axes[0].set_title('Width vs Visibility')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Visibility')

    # Height vs Visibility
    sns.scatterplot(data=df, x='height', y='visibility', ax=axes[1])
    axes[1].set_title('Height vs Visibility')
    axes[1].set_xlabel('Height')
    axes[1].set_ylabel('Visibility')

    # Height vs Visibility
    sns.scatterplot(data=df, x='area', y='visibility', ax=axes[2])
    axes[2].set_title('Area vs Visibility')
    axes[2].set_xlabel('Area')
    axes[2].set_ylabel('Visibility')

    plt.tight_layout()
    return fig


def plot_image_grid_by_visibility(df, image_dir, image_name_col="crop_image_name",
                                  max_images_per_visibility=3, figsize=None,
                                  width_scale=0.9):
    """
    Plot a grid of images grouped by visibility values, showing at most 3 images per visibility value.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing image name, 'visibility', 'width', and 'height' columns
    image_dir : str
        Directory path where the images are stored
    image_name_col : str
        Name of the column containing image filenames (default "crop_image_name")
    max_images_per_visibility : int
        Maximum number of images to display per visibility value (default 3)
    figsize : tuple or None
        Figure size as (width, height) in inches. If None, will be calculated automatically.
    width_scale : float
        Scale factor for figure width (0-1), smaller values place images closer together

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the image grid
    """
    # Group by visibility
    visibility_groups = df.groupby('visibility')
    unique_visibility_values = sorted(df['visibility'].unique())

    # Calculate much tighter figure size
    if figsize is None:
        # Much smaller multipliers for tight spacing
        width = max_images_per_visibility * 2 + 1.5  # Just enough for images + title
        height = len(unique_visibility_values) * 1.5  # Minimal height per row
        figsize = (width, height)

    # Create figure with no margins
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    # Create a grid with minimal spacing
    gs = gridspec.GridSpec(len(unique_visibility_values), 1, figure=fig,
                          hspace=0.01, left=0, right=1, top=1, bottom=0)

    # For each visibility value
    for i, visibility in enumerate(unique_visibility_values):
        group_df = visibility_groups.get_group(visibility)

        # Sample up to max_images_per_visibility images
        if len(group_df) > max_images_per_visibility:
            sample_df = group_df.sample(max_images_per_visibility, random_state=42)
        else:
            sample_df = group_df

        # Create a nested gridspec with title on the left and images on the right
        section_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2,
            subplot_spec=gs[i],
            width_ratios=[0.12, 0.88],  # Even smaller title area
            wspace=0.01,  # Minimal spacing
        )

        # Create title area on the left
        title_ax = fig.add_subplot(section_gs[0])
        title_ax.text(0.5, 0.5, f'Visibility:\n{visibility}',
                      fontsize=10, weight='bold', ha='center', va='center')
        title_ax.axis('off')

        # Create a nested gridspec for the images on the right
        image_gs = gridspec.GridSpecFromSubplotSpec(
            1, max_images_per_visibility,
            subplot_spec=section_gs[1],
            wspace=0.002,  # Almost no spacing between images
        )

        # Plot each image in the row
        for j, (_, row) in enumerate(sample_df.iterrows()):
            if j < max_images_per_visibility:
                # Create subplot
                ax = fig.add_subplot(image_gs[0, j])

                # Get image path
                img_path = os.path.join(image_dir, row[image_name_col])

                try:
                    # Try to read and display the image
                    img = plt.imread(img_path)
                    ax.imshow(img)
                    # Smaller font for dimensions
                    ax.set_xlabel(f"{row['width']}×{row['height']}", fontsize=7)
                except FileNotFoundError:
                    # If image not found, display placeholder
                    ax.text(0.5, 0.5, f"Image not found",
                            ha='center', va='center', fontsize=7)

                # Remove all axis elements
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # No tight_layout - it adds unwanted padding
    # Instead manually adjust if needed
    return fig

def visualise_points_only(points: List[shapely.Point],
                          labels: Optional[List[str]] = None,
                          colors: Optional[List[str]] = None,
                          markersize: float = 5.0,
                          title: str = "Points Visualization",
                          filename: Optional[Path] = None,
                          show: bool = True, text_buffer=True, font_size=10,
                          ax=None) -> axes:
    """
    Simplified function to visualize just points.
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    if not points:
        logger.warning("No points to visualize")
        return ax

    # Extract coordinates
    x_coords = [p.x for p in points if isinstance(p, Point)]
    y_coords = [p.y for p in points if isinstance(p, Point)]

    if not x_coords:
        print("No valid points found")
        return ax

    # Plot points
    if colors and len(colors) == len(x_coords):
        ax.scatter(x_coords, y_coords, c=colors, s=markersize ** 2, alpha=0.8, edgecolors='black')
    else:
        ax.scatter(x_coords, y_coords, s=markersize ** 2, alpha=0.8, edgecolors='black')

    # Add labels
    if labels:
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if i < len(labels):
                if text_buffer:
                    # add a small box around the text in white is it is easier to read
                    ax.text(x + 35, y + 5, labels[i], fontsize=font_size,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        plt.close()

    return ax


def visualise_hasty_annotation_statistics(annotated_images: typing.List[AnnotatedImage]):
    """
        Create histograms for image dimensions and annotation counts

        Args:
            annotated_images: List of AnnotatedImage objects
        """

    # Extract data from AnnotatedImage objects
    widths = [img.width for img in annotated_images]
    heights = [img.height for img in annotated_images]
    dimensions = [(img.width, img.height) for img in annotated_images]
    annotation_counts = [len(img.labels) for img in annotated_images]

    # Create figure with subplots - using GridSpec for custom layout
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Image Dataset Analysis', fontsize=16, fontweight='bold')

    # 1. Width Distribution (was position 2)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.hist(widths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Image Width Distribution')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Number of Images')
    ax1.grid(True, alpha=0.3)

    # 2. Height Distribution (was position 3)
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.hist(heights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title('Image Height Distribution')
    ax2.set_xlabel('Height (pixels)')
    ax2.set_ylabel('Number of Images')
    ax2.grid(True, alpha=0.3)

    # 3. Annotations per Image Histogram (double width - spans all 4 columns)
    ax3 = fig.add_subplot(gs[1, :])
    max_annotations = max(annotation_counts) if annotation_counts else 0

    # Create fewer, more meaningful bins
    if max_annotations <= 10:
        bins = list(range(0, max_annotations + 2))
        bin_labels = None

    elif max_annotations <= 50:
        base_bins = [0, 1, 5, 10, 20, 30, 50]
        bins = [b for b in base_bins if b <= max_annotations] + [max_annotations + 1]
        bin_labels = [f'{bins[i]}-{bins[i + 1] - 1}' if bins[i] != 0 else '0' for i in range(len(bins) - 2)]
        bin_labels = ['0'] + bin_labels[1:] + [f'{bins[-2]}+']

    elif max_annotations <= 100:
        base_bins = [0, 1, 5, 10, 25, 50, 75, 100]
        bins = [b for b in base_bins if b <= max_annotations] + [max_annotations + 1]
        bin_labels = ['0'] + [f'{bins[i]}-{bins[i + 1] - 1}' for i in range(1, len(bins) - 2)] + [f'{bins[-2]}+']

    elif max_annotations <= 200:
        base_bins = [0, 1, 5, 10, 25, 50, 75, 100, 200]
        bins = [b for b in base_bins if b <= max_annotations] + [max_annotations + 1]
        bin_labels = ['0'] + [f'{bins[i]}-{bins[i + 1] - 1}' for i in range(1, len(bins) - 2)] + [f'{bins[-2]}+']

    else:
        base_bins = [0, 1, 5, 10, 25, 50, 75, 100, 200]
        bins = base_bins + [max_annotations + 1]
        bin_labels = ['0'] + [f'{bins[i]}-{bins[i + 1] - 1}' for i in range(1, len(bins) - 1)]

    try:
        n, bins_edges, patches = ax3.hist(annotation_counts, bins=bins, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('Annotations per Image Distribution')
        ax3.set_xlabel('Number of Annotations')
        ax3.set_ylabel('Number of Images')
        ax3.grid(True, alpha=0.3)
    except ValueError as e:
        logger.error(f"Error creating histogram: {e}")
        return
    # Add custom bin labels if we have them
    if bin_labels:
        ax3.set_xticks([(bins_edges[i] + bins_edges[i + 1]) / 2 for i in range(len(bins_edges) - 1)])
        ax3.set_xticklabels(bin_labels, rotation=45, ha='right')

    plt.show()

    # Print summary statistics
    print("=== Dataset Summary ===")
    print(f"Total Images: {len(annotated_images)}")
    dimension_strings = [f"{w}x{h}" for w, h in dimensions]
    dimension_counts = typing.Counter(dimension_strings)
    sorted_dimensions = sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"Unique Dimensions: {len(dimension_counts)}")
    print(
        f"Most Common Dimension: {sorted_dimensions[0][0]} ({sorted_dimensions[0][1]} images)" if sorted_dimensions else "No images")
    print(f"Width Range: {min(widths)} - {max(widths)} pixels" if widths else "No widths")
    print(f"Height Range: {min(heights)} - {max(heights)} pixels" if heights else "No heights")
    print(
        f"Annotation Count Range: {min(annotation_counts)} - {max(annotation_counts)}" if annotation_counts else "No annotations")
    print(f"Average Annotations per Image: {np.mean(annotation_counts):.2f}" if annotation_counts else "No annotations")
    print(f"Images with 0 annotations: {annotation_counts.count(0)}")
    print(f"Images with annotations: {len([c for c in annotation_counts if c > 0])}")


def create_simple_histograms(annotated_images: typing.List[AnnotatedImage],
                             dataset_name: str, filename: Optional[Path] = None):
    """
    Create just the two main histograms requested with improved binning
    """
    # Extract data
    annotated_images

    dimensions = [(img.width, img.height) for img in annotated_images]
    annotation_counts = [len(img.labels) for img in annotated_images]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Bounding Box Size Analysis {dataset_name})',
                 fontsize=16, fontweight='bold')
    # 1. Image Dimensions Histogram
    dimension_strings = [f"{w}x{h}" for w, h in dimensions]
    dimension_counts = typing.Counter(dimension_strings)
    sorted_dimensions = sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True)

    if sorted_dimensions:
        dim_labels, dim_counts = zip(*sorted_dimensions)
        ax1.bar(range(len(dim_labels)), dim_counts, color='steelblue', alpha=0.7)
        ax1.set_title('Image Dimensions Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Image Dimensions (Width x Height)')
        ax1.set_ylabel('Number of Images')
        ax1.set_xticks(range(len(dim_labels)))
        ax1.set_xticklabels(dim_labels, rotation=45, ha='right')

        # Add value labels
        for i, count in enumerate(dim_counts):
            ax1.text(i, count + 0.1, str(count), ha='center', va='bottom')

    # 2. Annotations per Image Histogram with Custom Bins
    # Define bin categories and labels
    bin_labels = ['0', '1', '2', '3', '4', '5', '6-10', '11-20', '20+']
    bin_counts = [0] * len(bin_labels)

    # Count annotations in each bin
    for count in annotation_counts:
        if count == 0:
            bin_counts[0] += 1
        elif count == 1:
            bin_counts[1] += 1
        elif count == 2:
            bin_counts[2] += 1
        elif count == 3:
            bin_counts[3] += 1
        elif count == 4:
            bin_counts[4] += 1
        elif count == 5:
            bin_counts[5] += 1
        elif 6 <= count <= 10:
            bin_counts[6] += 1
        elif 11 <= count <= 20:
            bin_counts[7] += 1
        else:  # count > 20
            bin_counts[8] += 1

    # Create bar chart for annotations
    bars = ax2.bar(range(len(bin_labels)), bin_counts, color='orange', alpha=0.7, edgecolor='black')
    ax2.set_title("# Images per Annotations", fontsize=14, fontweight='bold') # TODO
    ax2.set_xlabel('Number of Annotations')
    ax2.set_ylabel('Number of Images')
    ax2.set_xticks(range(len(bin_labels)))
    ax2.set_xticklabels(bin_labels)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, count in enumerate(bin_counts):
        if count > 0:  # Only show labels for non-zero bars
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Print summary statistics for annotations
    print("=== Annotation Distribution Summary ===")
    for i, (label, count) in enumerate(zip(bin_labels, bin_counts)):
        percentage = (count / len(annotation_counts)) * 100 if annotation_counts else 0
        print(f"{label} annotations: {count} images ({percentage:.1f}%)")

    total_annotations = sum(annotation_counts)
    avg_annotations = total_annotations / len(annotation_counts) if annotation_counts else 0
    print(f"\nTotal images: {len(annotation_counts)}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {avg_annotations:.2f}")

    return fig

def plot_bbox_sizes(annotated_images: typing.List[AnnotatedImage],
                    suffix,
                    bins: int = 50,
                    plot_name: str = "box_sizes.png"):
    """
    Create continuous histograms for bounding box sizes

    Args:
        annotated_images: List of AnnotatedImage objects
        bins: Number of bins for the histograms (default: 50)
    """
    # Extract all bounding box data
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    bbox_aspect_ratios = []

    for img in annotated_images:
        for label in img.labels:
            # Assuming bbox format is [x, y, width, height] (COCO format)
            if hasattr(label, 'bbox') and label.bbox is not None:
                x1, y1, x2, y2 = label.bbox
                width = x2 - x1
                height = y2 - y1
                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(width * height)

                # Calculate aspect ratio (width/height), handle division by zero
                if height > 0:
                    bbox_aspect_ratios.append(width / height)

    if not bbox_widths:
        raise ValueError("No bounding boxes found in the dataset!")


    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Bounding Box Size Analysis ({len(bbox_widths)} boxes total, {suffix})',
                 fontsize=16, fontweight='bold')

    # 1. Box Width Distribution
    axes[0, 0].hist(bbox_widths, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Bounding Box Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Number of Boxes')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(bbox_widths), color='red', linestyle='--',
                       label=f'Mean: {np.mean(bbox_widths):.1f}')
    axes[0, 0].legend()

    # 2. Box Height Distribution
    axes[0, 1].hist(bbox_heights, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Bounding Box Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Number of Boxes')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(bbox_heights), color='red', linestyle='--',
                       label=f'Mean: {np.mean(bbox_heights):.1f}')
    axes[0, 1].legend()

    # 3. Box Area Distribution
    axes[1, 0].hist(bbox_areas, bins=bins, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Bounding Box Area Distribution')
    axes[1, 0].set_xlabel('Area (pixels²)')
    axes[1, 0].set_ylabel('Number of Boxes')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.mean(bbox_areas), color='red', linestyle='--',
                       label=f'Mean: {np.mean(bbox_areas):.0f}')
    axes[1, 0].legend()

    # 4. Aspect Ratio Distribution
    if bbox_aspect_ratios:
        axes[1, 1].hist(bbox_aspect_ratios, bins=bins, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Bounding Box Aspect Ratio Distribution')
        axes[1, 1].set_xlabel('Aspect Ratio (Width/Height)')
        axes[1, 1].set_ylabel('Number of Boxes')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(np.mean(bbox_aspect_ratios), color='red', linestyle='--',
                           label=f'Mean: {np.mean(bbox_aspect_ratios):.2f}')
        axes[1, 1].axvline(1.0, color='green', linestyle=':',
                           label='Square (1:1)')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("=== Bounding Box Statistics ===")
    print(f"Total bounding boxes: {len(bbox_widths)}")
    print(
        f"Width  - Min: {min(bbox_widths):.1f}, Max: {max(bbox_widths):.1f}, Mean: {np.mean(bbox_widths):.1f}, Std: {np.std(bbox_widths):.1f}")
    print(
        f"Height - Min: {min(bbox_heights):.1f}, Max: {max(bbox_heights):.1f}, Mean: {np.mean(bbox_heights):.1f}, Std: {np.std(bbox_heights):.1f}")
    print(
        f"Area   - Min: {min(bbox_areas):.0f}, Max: {max(bbox_areas):.0f}, Mean: {np.mean(bbox_areas):.0f}, Std: {np.std(bbox_areas):.0f}")
    if bbox_aspect_ratios:
        print(
            f"Aspect Ratio - Min: {min(bbox_aspect_ratios):.2f}, Max: {max(bbox_aspect_ratios):.2f}, Mean: {np.mean(bbox_aspect_ratios):.2f}, Std: {np.std(bbox_aspect_ratios):.2f}")

    return {
        "min_box_width": min(bbox_widths) if bbox_widths else None,
        "max_box_width": max(bbox_widths) if bbox_widths else None,
        "mean_box_width": np.mean(bbox_widths) if bbox_widths else None,
        "std_box_width": np.std(bbox_widths) if bbox_widths else None,
        "min_bbox_aspect_ratio": min(bbox_aspect_ratios) if bbox_aspect_ratios else None,
        "max_bbox_aspect_ratio": max(bbox_aspect_ratios) if bbox_aspect_ratios else None,
        "mean_bbox_aspect_ratio": np.mean(bbox_aspect_ratios) if bbox_aspect_ratios else None,
    }

