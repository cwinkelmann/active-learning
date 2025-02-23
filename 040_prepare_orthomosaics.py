"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles
This has some reasons: 1. empty tiles can be excluded easier 2. the model inferencing has difficulties with gigantic images

"""
import typing

import csv
from loguru import logger

from pathlib import Path

from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid
from active_learning.util.geospatial_slice import GeoSlicer
from com.biospheredata.converter.HastyConverter import ImageFormat
from PIL import Image


# Herdnet model


def save_tiles_to_csv(tiles, output_csv: Path, species="iguana", label=1):
    """
    Save raster tile filenames to a CSV file with additional metadata.

    Parameters:
        tiles (list): List of tile filenames (Paths or strings).
        output_csv (Path): Path to the output CSV file.
        species (str): Default species name (default: "iguana").
        label (int): Default label (default: 1).

    Returns:
        Path: Path to the saved CSV file.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Open CSV file for writing
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["images", "x", "y", "species", "labels"])

        # Iterate through tiles and write rows
        for tile in tiles:
            filename = Path(tile).name  # Extract just the filename
            x, y = 0, 0  # Default placeholders (modify if coordinates exist)
            writer.writerow([filename, x, y, species, label])

    print(f"CSV saved to: {output_csv}")
    return output_csv


def convert_tiles_to(tiles: typing.List[Path], format: ImageFormat, output_dir: Path):
    """
    Convert a list of image tiles to a specified format. Either from geospatial
    to pixel coordinates or vice versa (geospatial logic not fully implemented here).

    :param tiles: A list of Path objects pointing to the input tiles.
    :param format: The desired output image format (e.g., ImageFormat.PNG).
    :param output_dir: The directory where converted images will be saved.
    """

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile in tiles:
        if not tile.exists():
            logger.warning("Tile does not exist: %s", tile)
            continue

        # Open the image tile
        with Image.open(tile) as img:
            # If a world_file is provided, you can apply geospatial transformations here

            # Build the output filename (e.g., tile_name.png)
            output_filename = f"{tile.stem}.{format.value.lower()}"
            out_path = output_dir / output_filename
            if format == ImageFormat.JPG:
                img = img.convert("RGB")
            # Save the image in the desired format
            img.save(out_path, quality=95)

            logger.info(f"Converted {tile} -> {out_path}")



if __name__ == "__main__":
    orthomosaic_path = Path("/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif")
    output_dir = Path("/Users/christian/data/training_data/2025_02_22_HIT/FMO02_full_orthophoto_tiles/")

    # Run the function
    grid_gdf = create_regular_geospatial_raster_grid(full_image_path=Path(orthomosaic_path),
                                                     x_size=5120,
                                                     y_size=5120,
                                                     overlap_ratio=0.0)

    slicer = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf,
                       output_dir=output_dir)
    tiles = slicer.slice_very_big_raster()

    format = ImageFormat.JPG
    convert_tiles_to(tiles=tiles, format=format, output_dir=output_dir)

    print(tiles)

    # This is a bit of a hack, but we can use the function to save the tiles to a CSV file
    output_csv = output_dir / "herdnet_fake.csv"

    save_tiles_to_csv(tiles, output_csv, species = "iguana", label = 1)

