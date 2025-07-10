"""
Prepare an orthomosaic for use with the Herdnet model. For now this includes slicing the orthomosaic into tiles
This has some reasons: 1. empty tiles can be excluded easier 2. the model inferencing has difficulties with gigantic images

"""
from collections import defaultdict

import csv
import gc
import geopandas as gpd
import pandas as pd
from loguru import logger
from osgeo import gdal
from pathlib import Path

from active_learning.pipelines.geospatial_data_gen import geospatial_data_to_detection_training_data
from active_learning.types.Exceptions import ProjectionError, NoLabelsError
from com.biospheredata.converter.HastyConverter import ImageFormat

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

    FLPC01_07_22012021_geopackage = Path(
        "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/FLPC01_07_22012021.gpkg")
    Fer_FNE01_02_03_19122021_geopackage = Path(
        "Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg")

    # load geopackage layers
    pix4d_orthomosaic_body_counts_layername = "FLPC01-07_22012021-Pix4D orthomosaic body counts"
    gdf_pix4d_orthomosaic_body_counts = gpd.read_file(FLPC01_07_22012021_geopackage,
                                                      layer=pix4d_orthomosaic_body_counts_layername)

    pix4d_orthomosaic = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/FLPC01-07_22012021/exports/FLPC01-07_22012021-orthomosaic.tiff")

    metashape_FLPC01_orthomosaic = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/cog/Flo/Flo_FLPC01_22012021.tif")
    metashape_FLPC03_orthomosaic = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/cog/Flo/Flo_FLPC03_22012021.tif")
    metashape_FLPC04_orthomosaic = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/cog/Flo/Flo_FLPC04_22012021.tif")
    metashape_FLPC05_orthomosaic = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/cog/Flo/Flo_FLPC05_22012021.tif")
    metashape_FLPC06_orthomosaic = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/cog/Flo/Flo_FLPC06_22012021.tif")
    metashape_FLPC07_orthomosaic = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/cog/Flo/Flo_FLPC07_22012021.tif")

    # metashape body counts layer
    metashape_Flo_FLPC01_orthomosaic_head_counts_layername = "Flo_FLPC01_22012021 counts"
    # metashape_Flo_FLPC02_orthomosaic_head_counts_layername = "Flo_FLPC02_22012021 counts"
    metashape_Flo_FLPC03_orthomosaic_head_counts_layername = "Flo_FLPC03_22012021 counts"
    metashape_Flo_FLPC04_orthomosaic_head_counts_layername = "Flo_FLPC04_22012021 counts"
    metashape_Flo_FLPC05_orthomosaic_head_counts_layername = "Flo_FLPC05_22012021 counts"
    metashape_Flo_FLPC06_orthomosaic_head_counts_layername = "Flo_FLPC06_22012021 counts"
    metashape_Flo_FLPC07_orthomosaic_head_counts_layername = "Flo_FLPC07_22012021 counts"

    metashape_Flo_FLPC01_orthomosaic_body_counts_layername = "Flo_FLPC01_22012021 body counts"
    # metashape_Flo_FLPC02_orthomosaic_body_counts_layername = "Flo_FLPC02_22012021 body counts"
    metashape_Flo_FLPC03_orthomosaic_body_counts_layername = "Flo_FLPC03_22012021 body counts"
    metashape_Flo_FLPC04_orthomosaic_body_counts_layername = "Flo_FLPC04_22012021 body counts"
    metashape_Flo_FLPC05_orthomosaic_body_counts_layername = "Flo_FLPC05_22012021 body counts"
    metashape_Flo_FLPC06_orthomosaic_body_counts_layername = "Flo_FLPC06_22012021 body counts"
    metashape_Flo_FLPC07_orthomosaic_body_counts_layername = "Flo_FLPC07_22012021 body counts"

    # for testing
    geopackage_file_FNJ_02_03_04_19122021 = "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/FNJ_02-03-04_19122021.gpkg"
    layer_name_FNJ02_03_04 = "FNJ02-03-04_19122021-Pix4D-orthomosaic body counts"
    orthomosaics_FNJ_02_03_04_19122021 = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Pix4D orthomosaics/Fer/FNJ02-03-04_19122021-Pix4D-orthomosaic.tiff")

    data_validation = {
            "gpg_path": geopackage_file_FNJ_02_03_04_19122021,
            "layer_name": layer_name_FNJ02_03_04,
            "orthomosaic_path": orthomosaics_FNJ_02_03_04_19122021,
            "split": "val"
    }

    data_mapping_sample = [

        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE02_19122021 body counts Pix4D",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Pix4D Orthomosaics/FNE02_19122021/exports/FNE02_19122021-orthomosaic.tiff",
            "split": "train"
        },

    ]

    # Data set to test if how the image quality affects the model performance
    data_mapping_HQ_body = [
        # FLPC
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": pix4d_orthomosaic_body_counts_layername,
            "orthomosaic_path": pix4d_orthomosaic,
            "split": "train"
        },

        # FNE
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE01_19122021 body counts Pix4D",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Pix4D Orthomosaics/FNE01_19122021/exports/FNE01_19122021-orthomosaic.tiff",
            "split": "train"
        },
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE02_19122021 body counts Pix4D",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Pix4D Orthomosaics/FNE02_19122021/exports/FNE02_19122021-orthomosaic.tiff",
            "split": "train"
        },
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE03_19122021 body counts Pix4D",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Pix4D Orthomosaics/FNE03_19122021/exports/FNE03_19122021-orthomosaic.tiff",
            "split": "train"
        # raise ValueError("FNE03 is not available yet Pix4D orthomosaic format, only in Drone Deploy format. Please check the data mapping.")
        },

        # EPC
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/EPC03_10022023.gpkg",
            "layer_name": "Esp_EPC03_10022023 body counts pix4d",
            "orthomosaic_path": "/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/EPC03_10022023/exports/EPC03_10022023-orthomosaic.tiff",
            "split": "train"
        },

    ]

    data_mapping_LQ_body = [
        # FLPC01_07_22012021 body
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC01_orthomosaic_body_counts_layername,
            "orthomosaic_path": metashape_FLPC01_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC03_orthomosaic_body_counts_layername,
            "orthomosaic_path": metashape_FLPC03_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC04_orthomosaic_body_counts_layername,
            "orthomosaic_path": metashape_FLPC04_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC05_orthomosaic_body_counts_layername,
            "orthomosaic_path": metashape_FLPC05_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC06_orthomosaic_body_counts_layername,
            "orthomosaic_path": metashape_FLPC06_orthomosaic, "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC07_orthomosaic_body_counts_layername,
            "orthomosaic_path": metashape_FLPC07_orthomosaic,
            "split": "train"
        },

        # FNE body
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE01_19122021 body counts",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNE01_19122021.tif",
            "split": "train"
        },
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE02_19122021 body counts",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNE02_19122021.tif",
            "split": "train"
        },
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE03_19122021 body counts",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNE03_19122021.tif",
            "split": "train"
        },

        # EPC body
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/EPC03_10022023.gpkg",
            "layer_name": "Esp_EPC03_10022023 body counts",
            "orthomosaic_path": "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Esp/Esp_EPC03_10022023.tif",
            "split": "train"
        },
    ]

    data_mapping_LQ_head = [
        ## FLPC01_07_22012021
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC01_orthomosaic_head_counts_layername,
            "orthomosaic_path": metashape_FLPC01_orthomosaic,
            "split": "val"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC03_orthomosaic_head_counts_layername,
            "orthomosaic_path": metashape_FLPC03_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC04_orthomosaic_head_counts_layername,
            "orthomosaic_path": metashape_FLPC04_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC05_orthomosaic_head_counts_layername,
            "orthomosaic_path": metashape_FLPC05_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC06_orthomosaic_head_counts_layername,
            "orthomosaic_path": metashape_FLPC06_orthomosaic,
            "split": "train"
        },
        {
            "gpg_path": FLPC01_07_22012021_geopackage,
            "layer_name": metashape_Flo_FLPC07_orthomosaic_head_counts_layername,
            "orthomosaic_path": metashape_FLPC07_orthomosaic,
            "split": "val"
        },

        ## FNE01_02_03_19122021 HEAD COUNTS
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE01_19122021 counts",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNE01_19122021.tif",
            "split": "val"
        },
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE02_19122021 counts",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNE02_19122021.tif",
            "split": "train"
        },
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Fer_FNE_01-02-03_19022021.gpkg",
            "layer_name": "Fer_FNE03_19122021 counts",
            "orthomosaic_path": "/Volumes/2TB/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNE03_19122021.tif",
            "split": "train"
        },

        # EPC
        {
            "gpg_path": "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/EPC03_10022023.gpkg",
            "layer_name": "Esp_EPC03_10022023 body counts",
            "orthomosaic_path": "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Esp/Esp_EPC03_10022023.tif",
            "split": "train"
        },
    ]


    # 1. we will do data_mapping_LQ_head vs data_mapping_orthomosaics_LQ_body
    # 2. we will do data_mapping_orthomosaics_PIX4D_body vs. data_mapping_orthomosaics_LQ_body

    scenario = "LQ_head_vs_LQ_body"
    # scenario = "HQ_body_vs_LQ_body"
    datasets = {}

    df_lq_head = pd.DataFrame(data_mapping_LQ_head)
    df_hq_body = pd.DataFrame(data_mapping_HQ_body)
    df_lq_body = pd.DataFrame(data_mapping_LQ_body)

    # which scenario yield the best results?
    datasets["sample"] = pd.DataFrame(data_mapping_sample)
    datasets["train_hq_body"] = df_hq_body
    datasets["train_lq_head"] = df_lq_head
    datasets["train_lq_body"] = df_lq_body
    #datasets["val_FNJ"] = pd.DataFrame([data_validation])

    resolution = 512
    scale_factor = 1
    visualise_crops = True
    OBJECT_CENTERED = False  # If True, the crops are centered around the object, otherwise they are centered around the tile
    problematic_data_pairs = []

    format = ImageFormat.JPG

    tile_size = resolution // scale_factor

    # Expert export
    # expert = "Andrea"
    # base_path = Path(f"/Volumes/2TB/DD_MS_COG_Prepared_Training_2025_06_14_robin_{tile_size}_obcj_{OBJECT_CENTERED}")


    for subset, df_datasets in datasets.items():

        base_path = Path(f"/Users/christian/data/training_data/2025_07_06_DD_MS_COG_Exp_Training/{subset}/2025_07_06_{tile_size}_objcenter_{OBJECT_CENTERED}")
        base_path.mkdir(parents=True, exist_ok=True)
        herdnet_annotations = defaultdict(list)

        for index, row in df_datasets.iterrows():
            try:

                annotations_file = row["gpg_path"]
                layer_name = row["layer_name"]
                gdf_points = gpd.read_file(annotations_file,
                                           layer=layer_name)

                gdf_points = gdf_points[gdf_points.geometry.notnull()]

                split = row["split"]
                orthomosaic_path = Path(row["orthomosaic_path"])

                class_dict = {
                    'iguana': 1
                }

                # if species is not in gdf_points assign it
                if "species" not in gdf_points.columns:
                    logger.warning(f"Species column not found in {annotations_file}. Assigning default species.")
                    gdf_points['species'] = 'iguana'
                gdf_points['labels'] = gdf_points['species'].map(class_dict)

                # get the split

                output_object_dir = base_path / "herdnet" / f"{split}/iguana"
                output_empty_dir = base_path / "herdnet" / f"{split}/empty"

                vis_output_dir = base_path / "visualisation" / f"{split}"
                tile_output_dir = base_path / "tiff_tiles" / f"{split}"

                output_object_dir.mkdir(parents=True, exist_ok=True)

                output_empty_dir.mkdir(parents=True, exist_ok=True)
                vis_output_dir.mkdir(parents=True, exist_ok=True)
                tile_output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Processing {orthomosaic_path.name}")

                tile_folder_name = orthomosaic_path.stem

                herdnet_annotation = geospatial_data_to_detection_training_data(
                    gdf_points=gdf_points,
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
                problematic_data_pairs.append((orthomosaic_path, annotations_file, layer_name))
            except NoLabelsError as e:
                logger.error(f"No labels found for {orthomosaic_path.name}: {e}")
                problematic_data_pairs.append((orthomosaic_path, annotations_file, layer_name))

        for split, annotations in herdnet_annotations.items():
            # Save the herdnet annotations to a CSV filep
            combined_df = pd.concat(herdnet_annotations[split], ignore_index=True)

            # replace the column names to match the Herdnet format {tile_name,local_pixel_x,local_pixel_y,species,labels} with {images,x,y,species,labels}
            combined_df.rename(columns={
                "tile_name": "images",
                "local_pixel_x": "x",
                "local_pixel_y": "y"
            }, inplace=True)

            combined_df["labels"] = 1  # convert labels to int
            combined_df["species"] = "iguana"  # convert labels to int
            # remove full path from images
            combined_df["images"] = combined_df["images"].apply(lambda x: Path(x).name)

            output_annotation_dir = base_path / "herdnet" / f"{split}"
            output_annotation_dir.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(
                output_annotation_dir / f"herdnet_annotations_{split}.csv", index=False)


    if len(problematic_data_pairs) > 0:
        logger.error(f"Problematic data pairs: {problematic_data_pairs}")
        with open(base_path / "problematic_data_pairs.txt", "w") as f:
            for pair in problematic_data_pairs:
                f.write(f"{pair}\n")