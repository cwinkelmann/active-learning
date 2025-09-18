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

from active_learning.pipelines.geospatial_data_gen import geospatial_data_to_detection_training_data, \
    get_training_data_stats, check_merged_split_overlaps
from active_learning.types.Exceptions import ProjectionError, NoLabelsError, AnnotationFileNotSetError, \
    OrthomosaicNotSetError, LabelsOverlapError
from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.image_manipulation import convert_tiles_to, remove_empty_tiles
from active_learning.util.projection import convert_gdf_to_jpeg_coords, project_gdfcrs
from active_learning.util.super_resolution import super_resolve, SuperResolution
from com.biospheredata.types.status import ImageFormat
from active_learning.util.geospatial_transformations import get_geotiff_compression, get_gsd
from com.biospheredata.visualization.visualize_result import (visualise_image, visualise_polygons)
from osgeo import gdal
from collections import defaultdict
gdal.UseExceptions()

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


    # See 043_reorganise_shapefiles for the creation of this file
    usable_training_data_raster_mask = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/usable_training_data_raster_mask_with_group.geojson")

    gdf_mapping = gpd.read_file(usable_training_data_raster_mask)

    train_orthophotos = ["Fer_FNE01_19122021", "Fer_FNE02_19122021"]
    # val_orthophotos = ["Mar_MBBE05_09122021"]
    # test_orthophotos = ["Fer_FNE01_19122021"]
    #
    #
    val_orthophotos = ["Fer_FNE03_19122021",]
    # val_orthophotos = ["Flo_FLPC07_22012021", "Flo_FLPC06_22012021",
    #                      "Flo_FLPC05_22012021", "Flo_FLPC04_22012021",
    #                      "Flo_FLPC01_22012021", "Flo_FLPC02_22012021", "Flo_FLPC03_22012021",
    #
    #                    # "Flo_FLPO01_04022021", "Flo_FLPO02_04022021", "Flo_FLPO03_04022021"
    #                    ]

    # train_orthophotos = gdf_mapping["Orthophoto/Panorama name"].unique().tolist()

    data_splits = {
        "train": train_orthophotos,
        "val": val_orthophotos,
        # "test": test_orthophotos
    }
    orthomosaic_to_split = {
        orthomosaic: split_name
        for split_name, orthomosaics in data_splits.items()
        for orthomosaic in orthomosaics
    }

    resolution = 512
    scale_factor = 1
    visualise_crops = True
    OBJECT_CENTERED = True  # If True, the crops are centered around the object, otherwise they are centered around the tile
    problematic_data_pairs = []
    herdnet_annotations = defaultdict(list)
    format = ImageFormat.JPG


    get_training_data_stats(gdf_mapping)

    # add column split to dataframe
    gdf_mapping["split"] = gdf_mapping["Orthophoto/Panorama name"].map(orthomosaic_to_split)

    # sanity check if any split overlaps another split
    gdf_train = gdf_mapping[gdf_mapping["split"] == "train"]
    gdf_val = gdf_mapping[gdf_mapping["split"] == "val"]
    gdf_test = gdf_mapping[gdf_mapping["split"] == "test"]
    # Run the check
    # no_overlaps = check_merged_split_overlaps(gdf_train, gdf_val, gdf_test) # FIXME do not commit


    tile_size = resolution // scale_factor

    # Expert export
    # expert = "Andrea"
    # base_path = Path(f"/Volumes/2TB/DD_MS_COG_Prepared_Training_2025_06_14_robin_{tile_size}_obcj_{OBJECT_CENTERED}")
    base_path = Path(f"/Volumes/2TB/DD_MS_COG_Prepared_Training_2025_07_06_{tile_size}_objcenter_{OBJECT_CENTERED}")

    for index, row in gdf_mapping.iterrows():
        try:
            print(f"Processing idx {index} out of {len(gdf_mapping)}")

            annotations_file = Path(row["shp_file_path"])
            orthomosaic_path = Path(row["images_path"])

            class_dict = {
                'iguana': 1
            }


            gdf_points = gpd.read_file(annotations_file)
            # if species is not in gdf_points assign it
            if "species" not in gdf_points.columns:
                logger.warning(f"Species column not found in {annotations_file}. Assigning default species.")
                gdf_points['species'] = 'iguana'
            gdf_points['labels'] = gdf_points['species'].map(class_dict)

            if not row["Orthophoto/Panorama name"] in orthomosaic_to_split:
                logger.info(f"Skipping {orthomosaic_path.name} as it is not in the orthomosaic_to_split mapping")
                continue
            # if row["Expert"] != expert :
            #     logger.warning(f"Skipping {orthomosaic_path.name} because the Expoert is not {expert}, but is {row['Expert']}")
            #     continue

            # get the split
            split = row["split"]

            output_object_dir = base_path / "herdnet" / f"{split}/iguana"
            output_empty_dir = base_path / "herdnet" / f"{split}/empty"


            vis_output_dir = base_path / "visualisation" / f"{split}"
            tile_output_dir = base_path / "tiff_tiles" / f"{split}"

            output_object_dir.mkdir(parents=True, exist_ok=True)

            output_empty_dir.mkdir(parents=True, exist_ok=True)
            vis_output_dir.mkdir(parents=True, exist_ok=True)
            tile_output_dir.mkdir(parents=True, exist_ok=True)

            if row["Orthophoto/Panorama quality"] == "Bad":
                logger.warning(f"This orthomosaic is of bad quality: {row['Orthophoto/Panorama name']}")


            island_code = row["island_code"]
            logger.info(f"Processing {orthomosaic_path.name}")

            tile_folder_name = orthomosaic_path.stem

            herdnet_annotation = geospatial_data_to_detection_training_data(gdf_points=gdf_points,
                                                                            orthomosaic_path=orthomosaic_path,
                                                                            output_obj_dir=output_object_dir,
                                                                            output_empty_dir=output_empty_dir,
                                                                            vis_output_dir=vis_output_dir,
                                                                            tile_output_dir=tile_output_dir,
                                                                            tile_size=tile_size,
                                                                            visualise_crops=visualise_crops,
                                                                            format=format,
                                                                            OBJECT_CENTERED=OBJECT_CENTERED
                                                                            )

            logger.info(f"Done with {orthomosaic_path.name}")


            herdnet_annotations[split].append(herdnet_annotation)

            gc.collect()
        except ProjectionError as e:
            logger.error(f"Projection error for {orthomosaic_path.name}: {e}")
            problematic_data_pairs.append((orthomosaic_path, annotations_file))
        except NoLabelsError as e:
            logger.error(f"No labels found for {orthomosaic_path.name}: {e}")
            problematic_data_pairs.append((orthomosaic_path, annotations_file))



    for split, annotations in herdnet_annotations.items():
        # Save the herdnet annotations to a CSV filep
        combined_df = pd.concat(herdnet_annotations[split], ignore_index=True)
        raise ValueError(
            "replace {tile_name,local_pixel_x,local_pixel_y,species,labels} with {images,x,y,species,labels} in the next line")
        raise ValueError("convert labels to int")
        output_annotation_dir = base_path / "herdnet" / f"{split}"
        combined_df.to_csv(
            output_annotation_dir / f"herdnet_annotations_{split}.csv", index=False)


