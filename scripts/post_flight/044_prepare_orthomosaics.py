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

from active_learning.types.Exceptions import ProjectionError, NoLabelsError
from active_learning.util.geospatial_image_manipulation import create_regular_geospatial_raster_grid
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.image_manipulation import convert_tiles_to, remove_empty_tiles
from active_learning.util.projection import convert_gdf_to_jpeg_coords
from active_learning.util.super_resolution import super_resolve, SuperResolution
from com.biospheredata.converter.HastyConverter import ImageFormat
from geospatial_transformations import get_gsd, get_geotiff_compression
from util.util import visualise_image, visualise_polygons

from osgeo import gdal
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


def main(annotations_file: Path,
         orthomosaic_path: Path,
         island_code: str,
         tile_folder_name: str,
         output_dir: Path,
         output_empty_dir: Path,
         tile_size: int,
         vis_output_dir: Path,
         visualise_crops: bool,
         format: ImageFormat

         ):
    gdf_points = gpd.read_file(annotations_file)
    gdf_points["image_name"] = orthomosaic_path.name

    if len(gdf_points) == 0:
        raise NoLabelsError(f"No labels found in {annotations_file}")
    # incase the orthomosaic has a different CRS than the annotations
    # gdf_points = project_gdfcrs(gdf_points, orthomosaic_path)
    # project the global coordinates to the local coordinates of the orthomosaic



    # TODO clean this up
    gdf_local = convert_gdf_to_jpeg_coords(gdf_points, orthomosaic_path)
    # create an ImageCollection of the annotations

    # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file
    cog_compression = get_geotiff_compression(orthomosaic_path)
    logger.info(f"COG compression: {cog_compression}")
    gsd_x, gsd_y = get_gsd(orthomosaic_path)
    if round(gsd_x, 4) == 0.0093:
        logger.warning(
            "You are either a precise pilot or you wasted quality by using drone deploy, which caps images at about 0.93cm/px, compresses images a lot throws away details")

    # TODO make sure the CRS is the for both

    logger.info(f"Ground Sampling Distance (GSD): {100 * gsd_x:.3f} x {100 * gsd_y:.3f} cm/px")
    # Run the function

    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

    grid_manager.gdf_raster_mask.to_file(filename= vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson", driver='GeoJSON')

    logger.info(f"Partitioning annotations into a balanced dataset")
    grid_gdf, gdf_empty_cells = grid_manager.create_balanced_dataset_grids(points_gdf=gdf_points,
                                               box_size_x=tile_size,
                                               box_size_y=tile_size, num_empty_samples=len(gdf_points))



    # grid = grid_manager.create_regular_grid(x_size=tile_size, y_size=tile_size, overlap_ratio=0.0)
    # grid_gdf, gdf_empty_cells = grid_manager.filter_by_points(gdf_points)

    # grid_gdf = create_regular_geospatial_raster_grid(full_image_path=Path(orthomosaic_path),
    #                                                  x_size=tile_size,
    #                                                  y_size=tile_size,
    #                                                  overlap_ratio=0.0)
    logger.info(f"Done creating polygons for cutouts")
    gdf_empty_cells.to_file(vis_output_dir / f"grid_empty_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)
    # remove grid cells which don't contain points which are saved in gdf
    grid_gdf_filtered = grid_gdf[grid_gdf.geometry.apply(lambda poly: gdf_points.geometry.within(poly).any())]
    try:
        grid_gdf_filtered.to_file(vis_output_dir / f"grid_occupied_{orthomosaic_path.stem}.geojson", driver='GeoJSON')
    except Exception:
        pass

    # grid_gdf_filtered = grid_gdf_filtered[:1] ### TODO remove this

    logger.info(f"start slicing: {orthomosaic_path.stem}")
    slicer_occupied = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf_filtered,
                       output_dir=vis_output_dir)

    slicer_empty = GeoSlicer(base_path=orthomosaic_path.parent,
                             image_name=orthomosaic_path.name,
                             grid=gdf_empty_cells,
                             output_dir=vis_output_dir)

    logger.info(f"cut empty tiles from the orthomosaic into {len(slicer_occupied.grid)} tiles")
    slicer_empty.slice_very_big_raster(num_chunks=len(gdf_points) // 20 + 1, num_workers=3)
    converted_empty_tiles = convert_tiles_to(tiles=slicer_empty.slices, format=format,
                                       output_dir=output_empty_dir)

    # TODO increase image quality
    sr = SuperResolution(scale_factor=2, model_type=None)
    for ct in converted_empty_tiles:
        # sr_path = ct.parent / f"{ct.stem}_sr{ct.suffix}"
        sr_path = ct
        sr.super_resolution(input_path=ct, output_path=sr_path)

    gdf_sliced_points = slicer_occupied.slice_annotations(gdf_points, grid_gdf)


    # for tile_name, gdf_group in gdf_sliced_points.groupby(by="tile_name"):
    #     gdf_group.to_file(vis_output_dir / f"{tile_name}.geojson", driver='GeoJSON')

    logger.info(f"cut occupied tiles from the orthomosaic into {len(slicer_occupied.grid)} tiles")
    occupied_tiles = slicer_occupied.slice_very_big_raster(num_chunks=len(gdf_points) // 20 + 1, num_workers=3)
    # for tile in tiles:
    #    save_world_file_json(tile, output_dir)
    # remove tiles which contain no pixels
    occupied_tiles = remove_empty_tiles(occupied_tiles)

    converted_tiles = convert_tiles_to(tiles=occupied_tiles, format=format,
                                       output_dir=output_dir)

    # TODO increase image quality
    sr = SuperResolution(scale_factor=2, model_type=None)
    for ct in converted_tiles:

        #sr_path = ct.parent / f"{ct.stem}_sr{ct.suffix}"
        sr_path = ct
        sr.super_resolution(input_path=ct, output_path=sr_path)


    ### TODO create herdnet annotations for each tile
    if visualise_crops:
        vis_path = vis_output_dir / f"visualisations"
        vis_path.mkdir(exist_ok=True, parents=True)
        for tile_name, gdf_group in gdf_sliced_points.groupby(by="tile_name"):
            # just to check if that actually works we take the file not the gdf
            annotations_file = output_dir / f"{tile_name}.csv"
            df_herdnet = copy.copy(gdf_group[["tile_name", "local_pixel_x", "local_pixel_y"]])
            df_herdnet.loc[:, "species"] = "iguana"
            df_herdnet.loc[:, "labels"] = 1

            logger.info(f"Visualising {tile_name}")
            tile_image_name = f"{tile_name}.jpg"
            assert output_dir.joinpath(
                tile_image_name).exists(), f"{tile_image_name} does not exist"

            ax_s = visualise_image(image_path=output_dir / tile_image_name, show=False,
                                   title=f"Visualisation of {len(df_herdnet)} labels in {tile_image_name}")

            filename = vis_path / f"{tile_name}.jpg"
            visualise_polygons(
                points=[shapely.Point(x, y) for x, y in zip(df_herdnet.local_pixel_x, df_herdnet.local_pixel_y)],
                labels=df_herdnet["species"], ax=ax_s, show=False, linewidth=6, markersize=10,
                filename=filename)

            plt.close()

    df_herdnet = copy.copy(gdf_sliced_points[["tile_name", "local_pixel_x", "local_pixel_y"]])
    df_herdnet.loc[:, "species"] = "iguana"
    df_herdnet.loc[:, "labels"] = 1
    df_herdnet.loc[:, "island_code"] = island_code

    return df_herdnet


if __name__ == "__main__":
    annotations_file = None
    # annotations_file = Path('/Users/christian/data/Manual Counting/Fer_FNF02_19122021/Fer_FNF02_19122021 counts.shp')
    # annotations_file = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/Fer/Fer_FNE02_19122021 counts.geojson')
    # orthomosaic_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/DD_MS_COG_ALL/Fer/Fer_FNE02_19122021.tif")

    orthomosaic_shapefile_mapping_path = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/shapefile_orthomosaic_mapping.csv")
    df_mapping = pd.read_csv(orthomosaic_shapefile_mapping_path)

    tile_size = 224 // 2
    # /Volumes/G-DRIVE/DD_MS_COG_ALL_TILES
    output_dir = Path(f"/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_{tile_size}/iguana")
    output_empty_dir = Path(f"/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_{tile_size}/empty")

    analysis_output_dir = Path("/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_analysis/")
    vis_output_dir = Path("/Volumes/2TB/DD_MS_COG_ALL_TILES/visualisation")


    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    output_empty_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    herdnet_annotations = []
    problematic_data_pairs = []

    for index, row in df_mapping.iterrows():

        try:
            orthomosaic_path = Path(row["images_path"])
            annotations_file = Path(row["geojson_path"])

            island_code = row["island_code"]

            # if not orthomosaic_path.name == "Pnz_PZE10_08012023.tif":
            #     continue

            # island_code = orthomosaic_path.parts[-2]
            tile_folder_name = orthomosaic_path.stem

            logger.info(f"Processing {orthomosaic_path.name}")

            visualise_crops = False
            format = ImageFormat.JPG

            herdnet_annotation = main(annotations_file=annotations_file,
                                      orthomosaic_path=orthomosaic_path,
                                      island_code=island_code,
                                      tile_folder_name=tile_folder_name,
                                      output_dir=output_dir,
                                      output_empty_dir=output_empty_dir,
                                      vis_output_dir=vis_output_dir,
                                      tile_size=tile_size, visualise_crops=visualise_crops, format=format)

            logger.info(f"Done with {orthomosaic_path.name}")

            herdnet_annotations.append(herdnet_annotation)

            gc.collect()

        except ProjectionError:
            row["reason"] = "ProjectionError"
            problematic_data_pairs.append(row)
            logger.error(f"ProjectionError: {row}")
        except KeyError:
            row["reason"] = "KeyError"
            logger.error(f"KeyError: {row}")
            problematic_data_pairs.append(row)
        except NoLabelsError:
            row["reason"] = "NoLabelsError"
            logger.error(f"KeyError: {row}")
            problematic_data_pairs.append(row)



        # Save the herdnet annotations to a CSV filep
    combined_df = pd.concat(herdnet_annotations, ignore_index=True)
    combined_df.to_csv(
        analysis_output_dir / "herdnet_annotations.csv", index=False)

    pd.DataFrame(problematic_data_pairs).to_csv(analysis_output_dir / "problematic_data.csv", index=False)
