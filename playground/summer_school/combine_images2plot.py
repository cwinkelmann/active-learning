"""
Create an image grid from individual images
"""

from matplotlib import pyplot as plt, gridspec
from pathlib import Path
from PIL import Image
import os
from loguru import logger


def combine_heatmaps_to_grid(images_path, image_template_name,
                             i_start=1, i_end=14,
                             rows=7, cols=2,
                             figsize=None, dpi=300,
                             hspace=0.01, wspace=0.01):

    # Calculate number of images
    num_images = i_end - i_start + 1  # Fixed: was missing +1
    assert num_images <= rows * cols, f"Not enough grid cells for {num_images} images ({rows}x{cols})"

    # Better figure size calculation
    if figsize is None:
        # Larger figure size helps reduce relative spacing
        width = cols * 2.0  # Increased from 1
        height = rows * 2.0  # Increased from 1
        figsize = (width, height)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Remove ALL margins and padding
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    # Create grid with specified spacing
    gs = gridspec.GridSpec(rows, cols, figure=fig,
                           hspace=hspace, wspace=wspace,
                           left=0, right=1, top=1, bottom=0)

    # Loop through images with correct indexing
    image_idx = 0  # Track position in grid
    for i in range(i_start, i_end + 1):  # Fixed: added +1
        # Calculate grid position correctly
        row = image_idx // cols  # Fixed: use image_idx instead of i
        col = image_idx % cols   # Fixed: use image_idx instead of i

        ax = fig.add_subplot(gs[row, col])

        img_path = images_path / image_template_name.format(i=i)

        try:
            img = plt.imread(img_path)
            ax.imshow(img, aspect='auto')
            logger.info(f"Loaded: {img_path}")
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"Missing\nImage {i}",
                    ha='center', va='center', fontsize=6)
            logger.warning(f"Missing: {img_path}")

        # Remove all axis decorations
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Hide spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        image_idx += 1

    # Save without bbox_inches='tight' which adds padding
    output_filename = f"heatmaps_grid_h{hspace}_w{wspace}.png"
    plt.savefig(output_filename, dpi=dpi,
                pad_inches=0,  # Fixed: was 0.01
                facecolor='white')

    logger.info(f"Saved: {output_filename}")
    return fig



if __name__ == "__main__":
    image_template_name = "FMO03___DJI_0493_x1536_y3072.jpg_{i}_heatmap.png"
    images_path = Path("/Users/christian/data/Iguanas_From_Above/visualisations/visualizations/gif_folder")

    # Regular version with small spacing
    fig1 = combine_heatmaps_to_grid(
        images_path=images_path,
        image_template_name=image_template_name,
        i_start=1, i_end=12, rows=4, cols=3,
        hspace=0.005, wspace=0.005
    )


    plt.show()