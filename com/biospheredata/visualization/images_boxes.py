"""
Visuliaztion of images in boxes
"""
from matplotlib import pyplot as plt
from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import Image


def plot_images_grid(cropped_images: list[Path],
                                    output_path='./grid.jpg',
                                    show=False):
    """
    Plot a grid of images
    TODO this is probably redundant to visualise_polygons
    :param filepaths:
    :param output_path:
    :param show:
    :return:
    """
    result_figsize_resolution = 70  # 1 = 100px

    # TODO add a check for the number of images, maybe add more rows
    fig, axes = plt.subplots(1, len(cropped_images),
                             figsize=(result_figsize_resolution, result_figsize_resolution))

    result_grid_filename = output_path

    current_file_number = 0
    for image_filename in cropped_images:
        with Image.open(image_filename) as im:
            axes[current_file_number].imshow(cropped_images[current_file_number])
            axes[current_file_number].set_title(image_filename.name, fontdict={'fontsize': 17})

        axes[current_file_number].set_xticks([])
        axes[current_file_number].set_yticks([])
        current_file_number += 1

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    if output_path:
        plt.savefig(result_grid_filename)

    if show:
        plt.show()

    return fig

