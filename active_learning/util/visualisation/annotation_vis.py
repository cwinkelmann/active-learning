import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import os


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

# Example usage:
# fig = plot_frequency_distribution(df, columns=['height', 'width', 'visibility'])
# plt.show()

# Or just for one column:
# fig = plot_frequency_distribution(df, columns=['height'])
# plt.show()

# Or automatic selection of numeric columns:
# fig = plot_frequency_distribution(df)
# plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


    plt.tight_layout()
    return fig


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


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

    # Automatically calculate figure size if not provided
    if figsize is None:
        # Width based on number of columns and spacing
        width = max_images_per_visibility * 4 * width_scale  # Adjust this multiplier as needed
        # Height based on number of rows
        height = len(unique_visibility_values) * 3.5  # Adjust this multiplier as needed
        figsize = (width, height)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create a grid with title + images for each visibility value
    gs = gridspec.GridSpec(len(unique_visibility_values), 1, figure=fig, hspace=0.1)

    # For each visibility value
    for i, visibility in enumerate(unique_visibility_values):
        group_df = visibility_groups.get_group(visibility)

        # Sample up to max_images_per_visibility images
        if len(group_df) > max_images_per_visibility:
            sample_df = group_df.sample(max_images_per_visibility, random_state=42)
        else:
            sample_df = group_df

        # Create a nested gridspec for title and images
        section_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs[i],
            height_ratios=[0.15, 0.85],  # Small title, large image area
            hspace=0.01  # Minimal spacing between title and images
        )

        # Create a separate area just for the visibility title
        title_ax = fig.add_subplot(section_gs[0])
        title_ax.text(0.5, 0.5, f'Visibility: {visibility}',
                      fontsize=12, weight='bold', ha='center', va='center')
        title_ax.axis('off')

        # Create a nested gridspec for the images
        image_gs = gridspec.GridSpecFromSubplotSpec(
            1, max_images_per_visibility,
            subplot_spec=section_gs[1],
            wspace=0.01  # Minimal spacing between images
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
                    # Use a smaller, more compact font for the size info
                    ax.set_xlabel(f"{row['width']}×{row['height']}", fontsize=8)
                except FileNotFoundError:
                    # If image not found, display placeholder
                    ax.text(0.5, 0.5, f"Image not found",
                            ha='center', va='center', fontsize=8)

                # Remove axis ticks and spines to reduce space
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # Add tight layout with minimal padding
    plt.tight_layout(pad=0.5)
    return fig

# Example usage:
# fig = plot_image_grid_by_visibility(df, image_dir='path/to/images', width_scale=0.8)
# plt.show()