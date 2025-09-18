"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles
This has some reasons: 1. empty tiles can be excluded easier 2. the model inferencing has difficulties with gigantic images

"""
from collections import defaultdict

import csv
import gc
import pandas as pd
from loguru import logger
from osgeo import gdal
from pathlib import Path

from active_learning.pipelines.geospatial_data_gen import geospatial_data_to_detection_training_data_with_hard_neg
from active_learning.types.Exceptions import ProjectionError
from com.biospheredata.types.status import ImageFormat

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


def identify_hard_negatives_by_proximity(gdf_predictions: gpd.GeoDataFrame,
                                         gdf_true_positives: gpd.GeoDataFrame, buffer_distance = 1.0):
    """
    Identify hard negatives by finding predictions that don't match any true positives
    because there are no good keys in true positives, we will use the unique_id column. remove every prediction which is within 5m of a true positive."


    """

    # Create a buffer of 5 meters around each true positive
      # meters
    gdf_true_positives_buffered = gdf_true_positives.copy()
    gdf_true_positives_buffered['geometry'] = gdf_true_positives.geometry.buffer(buffer_distance)

    # Create a union of all buffered true positives to create exclusion zones
    exclusion_zones = gdf_true_positives_buffered.unary_union

    # Find predictions that are NOT within any exclusion zone (5m of true positives)
    is_hard_negative = ~gdf_predictions.geometry.within(exclusion_zones)

    # Filter to get hard negatives
    gdf_hard_negatives = gdf_predictions[is_hard_negative].copy()

    return gdf_hard_negatives, gdf_true_positives_buffered



def get_hard_negatives(gdf_prediction: gpd.GeoDataFrame, gdf_true_positves: gpd.GeoDataFrame):
    """
    Get the hard negatives from the prediction which are not in the true positives
    :param gdf_prediction: GeoDataFrame with the predictions
    :param gdf_true_positves: GeoDataFrame with the true positives
    :return: GeoDataFrame with the hard negatives
    """
    # Ensure both GeoDataFrames have the same CRS
    if gdf_prediction.crs != gdf_true_positves.crs:
        raise ProjectionError("CRS of prediction and true positives do not match")


    # look for points which are not in the true positives
    # TODO I need a hacky way to find the hard negatives
    # Identify hard negatives
    hard_negatives, true_positives_buffered = identify_hard_negatives_by_proximity(
        gdf_prediction,
        gdf_true_positves
    )
    hard_negatives['species'] = 'hard_negative'
    hard_negatives['labels'] = None

    return hard_negatives, true_positives_buffered




if __name__ == "__main__":
    EPSG_Code = 32715  # UTM Zone 15S
    EPSG_String = f"EPSG:{EPSG_Code}"

    # # orthophoto path
    # train_orthophoto_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff")
    #
    # train_shapefile = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Pix4Dmatic_Orthomosaic/counts/FLPC01-07_22012021-orthomosaic counts.shp")
    # # path to hard negative examples
    # train_hard_negatives = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Pix4Dmatic_Orthomosaic/counts/FLPC01-07_22012021-orthomosaic_detections_herdnet.geojson')

    # orthophoto path
    train_orthophoto_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Flo/FLPL01-02_28012023_Tiepoints-orthomosaic.tiff")

    train_shapefile_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Flo/FLPL01-02_28012023_Tiepoints-orthomosaic_tiles/corrected_detections_FLPL01-02_28012023_Tiepoints-orthomosaic.geojson.shp")
    gdf_true_positves = gpd.read_file(train_shapefile_path)

    train_all_predictions_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Flo/FLPL01-02_28012023_Tiepoints-orthomosaic_tiles/detections_FLPL01-02_28012023_Tiepoints-orthomosaic.geojson")
    gdf_prediction = gpd.read_file(train_all_predictions_path)

    # add a unique_id column to the predictions
    if "unique_id" not in gdf_prediction.columns:
        logger.warning(f"Adding unique_id to the predictions GeoDataFrame {train_all_predictions_path.name}. That id should be there in the first place.")
        gdf_prediction["unique_id"] = gdf_prediction.index.astype(str) + "_" + gdf_prediction["images"]

    # remove points which are not present in the train shapefile or have been corrected
    gdf_prediction = gdf_prediction[gdf_prediction["geometry"].isna() == False]  # Remove null geometries


    # add a unique_id column to the predictions
    if "unique_id" not in gdf_true_positves.columns:
        logger.warning(f"Adding unique_id to the predictions GeoDataFrame {train_shapefile_path.name}. That id should be there in the first place.")
        gdf_true_positves["unique_id"] = gdf_prediction.index.astype(str) + "_" + gdf_prediction["images"]
    # remove points which are not present in the train shapefile or have been corrected

    gdf_true_positves = gdf_true_positves[gdf_true_positves["geometry"].isna() == False]  # Remove null geometries

    gdf_train_hard_negatives, gdf_true_positives_buffered = get_hard_negatives(gdf_prediction, gdf_true_positves)


    gdf_true_positives_buffered.to_file(train_all_predictions_path.parent / "true_positives_buffered_exclusion_zones.geojson", driver="GeoJSON")

    # path to hard negative examples
    train_hard_negatives_path = train_all_predictions_path.parent / "hard_negatives.geojson"
    gdf_train_hard_negatives.to_file(train_hard_negatives_path, driver="GeoJSON")

    gdf_true_positves["species"] = "iguana"

    label_id_mapping = {
        "iguana": 1,
        "hard_negative": 2
    }


    # Convert both to UTM Zone 15S
    gdf_train_iguanas_utm = gdf_true_positves.to_crs(epsg=EPSG_Code)
    gdf_train_hard_negatives_utm = gdf_train_hard_negatives.to_crs(epsg=EPSG_Code)

    # map species to labels
    gdf_train_iguanas_utm['labels'] = gdf_train_iguanas_utm['species'].map(label_id_mapping)
    gdf_train_hard_negatives_utm['labels'] = gdf_train_hard_negatives_utm['species'].map(label_id_mapping)


    split = "train"

    data_splits = {
        "train": train_orthophoto_path,
        # "val": val_orthophotos,
        # "test": test_orthophotos
    }

    resolution = 512
    scale_factor = 1
    visualise_crops = True
    OBJECT_CENTERED = False  # If True, the crops are centered around the object, otherwise they are centered around the tile
    herdnet_annotations = defaultdict(list)
    format = ImageFormat.JPG

    tile_size = resolution // scale_factor
    base_path = Path(f"/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics_training_data/{train_orthophoto_path.stem}_{tile_size}_obcj_{OBJECT_CENTERED}")


    output_object_dir = base_path / "herdnet" / f"{split}/iguana"
    output_empty_dir = base_path / "herdnet" / f"{split}/empty"


    vis_output_dir = base_path / "visualisation" / f"{split}"
    tile_output_dir = base_path / "tiff_tiles" / f"{split}"

    output_object_dir.mkdir(parents=True, exist_ok=True)

    output_empty_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    tile_output_dir.mkdir(parents=True, exist_ok=True)


    logger.info(f"Processing {train_orthophoto_path.name}")

    tile_folder_name = train_orthophoto_path.stem

    herdnet_annotation = geospatial_data_to_detection_training_data_with_hard_neg(
        gdf_points_objects=gdf_train_iguanas_utm,
        gdf_points_hard_neg=gdf_train_hard_negatives_utm,
        orthomosaic_path=train_orthophoto_path,
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
    # There can be more objects in herdnet_annotation than there are points because of overlapping tiles
    logger.info(f"Done with {train_orthophoto_path.name}")

    herdnet_annotations[split].append(herdnet_annotation)

    gc.collect()




    for split, annotations in herdnet_annotations.items():
        # Save the herdnet annotations to a CSV filep
        combined_df = pd.concat(herdnet_annotations[split], ignore_index=True)
        raise ValueError(
            "replace {tile_name,local_pixel_x,local_pixel_y,species,labels} with {images,x,y,species,labels} in the next line")
        raise ValueError("convert labels to int")
        output_annotation_dir = base_path / "herdnet" / f"{split}"
        combined_df.to_csv(
            output_annotation_dir / f"herdnet_annotations_{split}.csv", index=False)


