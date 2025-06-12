"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles
This has some reasons: 1. empty tiles can be excluded easier 2. the model inferencing has difficulties with gigantic images

"""
import gc

import copy
import csv
import geopandas as gpd
import pandas as pd
import shapely
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.pipelines.geospatial_data_gen import geospatial_data_to_detection_training_data
from active_learning.types.Exceptions import ProjectionError, NoLabelsError, AnnotationFileNotSetError, \
    OrthomosaicNotSetError, LabelsOverlapError
from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.image_manipulation import convert_tiles_to, remove_empty_tiles
from active_learning.util.projection import convert_gdf_to_jpeg_coords, project_gdfcrs
from active_learning.util.super_resolution import super_resolve, SuperResolution
from com.biospheredata.converter.HastyConverter import ImageFormat
from geospatial_transformations import get_gsd, get_geotiff_compression
from util.util import visualise_image, visualise_polygons

from osgeo import gdal
from collections import defaultdict
gdal.UseExceptions()
import geopandas as gpd


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






# TODO maybe use the group_nearby_polygons_simple function to get the groups


# Optional: Basic visualization check
def plot_combined_data(gdf):
    """
    Quick plot to verify the combination worked
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot different species in different colors
    for species in gdf['species'].unique():
        subset = gdf[gdf['species'] == species]
        subset.plot(ax=ax, label=species, alpha=0.7, markersize=20)

    ax.legend()
    ax.set_title("Combined Training Data")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    # orthophoto path
    train_orthophoto = Path("/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff")

    train_shapefile = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Pix4Dmatic_Orthomosaic/counts/FLPC01-07_22012021-orthomosaic counts.shp")
    # path to hard negative examples
    train_hard_negatives = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Pix4Dmatic_Orthomosaic/counts/FLPC01-07_22012021-orthomosaic_detections_herdnet.geojson')

    gdf_train_iguanas = gpd.read_file(train_shapefile)
    gdf_train_hard_negatives = gpd.read_file(train_hard_negatives)

    gdf_train_hard_negatives["species"] = "hard_negative"
    gdf_train_iguanas["species"] = "iguana"

    label_id_mapping = {
        "iguana": 1,
        "hard_negative": 2
    }

    # METHOD 1: Convert both to same CRS (your current approach - RECOMMENDED)
    # Convert both to UTM Zone 15S
    gdf_train_iguanas_utm = gdf_train_iguanas.to_crs(epsg=32715)
    gdf_train_hard_negatives_utm = gdf_train_hard_negatives.to_crs(epsg=32715)

    # Combine using pd.concat (recommended - append is deprecated)
    df_train = pd.concat([gdf_train_iguanas_utm, gdf_train_hard_negatives_utm],
                          ignore_index=True)

    gdf_train = gpd.GeoDataFrame(df_train, geometry='geometry', crs='EPSG:32715')

    print(f"Combined GDF CRS: {gdf_train.crs}")
    print(f"Combined shape: {gdf_train.shape}")

    # map species to labels
    gdf_train['labels'] = gdf_train['species'].map(label_id_mapping)


    split = "train"

    data_splits = {
        "train": train_orthophoto,
        # "val": val_orthophotos,
        # "test": test_orthophotos
    }

    resolution = 512
    scale_factor = 1
    visualise_crops = False
    OBJECT_CENTERED = False  # If True, the crops are centered around the object, otherwise they are centered around the tile
    herdnet_annotations = defaultdict(list)
    format = ImageFormat.JPG

    tile_size = resolution // scale_factor
    base_path = Path(f"/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics_training_data/{train_orthophoto.stem}_{tile_size}_obcj_{OBJECT_CENTERED}")


    output_object_dir = base_path / "herdnet" / f"{split}/iguana"
    output_empty_dir = base_path / "herdnet" / f"{split}/empty"


    vis_output_dir = base_path / "visualisation" / f"{split}"
    tile_output_dir = base_path / "tiff_tiles" / f"{split}"

    output_object_dir.mkdir(parents=True, exist_ok=True)

    output_empty_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    tile_output_dir.mkdir(parents=True, exist_ok=True)


    logger.info(f"Processing {train_orthophoto.name}")

    tile_folder_name = train_orthophoto.stem

    herdnet_annotation = geospatial_data_to_detection_training_data(gdf_points=gdf_train,
                                                                    orthomosaic_path=train_orthophoto,
                                                                    output_obj_dir=output_object_dir,
                                                                    output_empty_dir=output_empty_dir,
                                                                    vis_output_dir=vis_output_dir,
                                                                    tile_output_dir=tile_output_dir,
                                                                    tile_size=tile_size,
                                                                    visualise_crops=visualise_crops,
                                                                    format=format,
                                                                    OBJECT_CENTERED=OBJECT_CENTERED,
                                                                    sample_fraction=0.5
                                                                    )

    logger.info(f"Done with {train_orthophoto.name}")


    # TODO This is a hack to get the annotations into the herdnet_annotations dict
    # remove "hard_negatives" from the herdnet_annotation
    herdnet_annotation = herdnet_annotation[herdnet_annotation["species"] != "hard_negative"]

    # replace the full path with just the filename
    herdnet_annotation["tile_name"] = herdnet_annotation["tile_name"].apply(lambda x: Path(x).name)

    herdnet_annotations[split].append(herdnet_annotation)

    gc.collect()




    for split, annotations in herdnet_annotations.items():
        # Save the herdnet annotations to a CSV filep
        combined_df = pd.concat(herdnet_annotations[split], ignore_index=True)

        output_annotation_dir = base_path / "herdnet" / f"{split}"
        combined_df.to_csv(
            output_annotation_dir / f"herdnet_annotations_{split}.csv", index=False)


