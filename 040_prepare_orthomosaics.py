"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles

"""
import csv

from pathlib import Path

from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid
from active_learning.util.geospatial_slice import GeoSlicer
from com.biospheredata.helper.image.image_coordinates import local_coordinates_to_wgs84


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


if __name__ == "__main__":
    orthomosaic_path = Path("/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif")
    output_dir = Path("/Users/christian/data/orthomosaics/tiles/")

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
    print(tiles)
    output_csv = output_dir / "herdnet_fake.csv"

    save_tiles_to_csv(tiles, output_csv, species = "iguana", label = 1)

    ## TODO inference using a model

    ## Convert predictions to COCO??

