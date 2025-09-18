"""
HUMAN IN THE LOOP

Take prediction we don't have a ground truth for and double check if the prediction is right.
There are two options: 1. mark it as iguanas or 2. mark it as a partial iguana

Then prepare the output for another training round

"""
import hashlib

import geopandas as gpd
from loguru import logger
from pathlib import Path

from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from active_learning.util.geospatial_transformations import get_geotiff_compression, get_gsd
from active_learning.util.image_manipulation import convert_tiles_to
from active_learning.util.projection import project_gdfcrs, convert_gdf_to_jpeg_coords
from com.biospheredata.types.status import ImageFormat
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
import fiftyone as fo
from loguru import logger


def verfify_points(gdf_points, config: GeospatialDatasetCorrectionConfig):
    gdf_points.drop(columns=["images", "x", "y" ], inplace=True, errors='ignore')
    gdf_points = project_gdfcrs(gdf_points, config.image_path)
    # project the global coordinates to the local coordinates of the orthomosaic

    if "scores" not in gdf_points.columns:
        gdf_points["scores"] = None
    if "species" not in gdf_points.columns:
        gdf_points["species"] = "iguana_point"
    if "labels" not in gdf_points.columns:
        gdf_points["labels"] = 1

    # TODO clean this up or make very ordinary hasty annotations out of it. This way the rest of the code would be kind of unnecessary
    gdf_local = convert_gdf_to_jpeg_coords(gdf_points, config.image_path)
    # create an ImageCollection of the annotations

    # Then I could use the standard way of slicing the orthomosaic into tiles and save the tiles to a CSV file
    cog_compression = get_geotiff_compression(config.image_path)
    logger.info(f"COG compression: {cog_compression}")
    gsd_x, gsd_y = get_gsd(config.image_path)
    if round(gsd_x, 4) == 0.0093:
        logger.warning(
            "You are either a precise pilot or you wasted quality by using 'DroneDeploy', which caps Orthophoto GSD at about 0.93cm/px, compresses images a lot and throws away details")

    logger.info(f"Ground Sampling Distance (GSD): {100 * gsd_x:.3f} x {100 * gsd_y:.3f} cm/px")
    # Run the function

    return gdf_points

def main(config: GeospatialDatasetCorrectionConfig,
         output_dir: Path,
         vis_output_dir: Path,
         geospatial_flag=True):
    tile_output_dir = output_dir / "tiles"
    tile_output_dir.mkdir(parents=True, exist_ok=True)
    format = ImageFormat.JPG
    hA_reference = HastyAnnotationV2.from_file(config.hasty_reference_annotation_path)
    config.image_tiles_path = tile_output_dir

    try:
        # fo.dataset_exists(config.dataset_name)
        fo.delete_dataset(config.dataset_name)
        pass
    except:
        logger.warning(f"Dataset {config.dataset_name} does not exist")
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset = fo.Dataset(name=config.dataset_name)
        dataset.persistent = True


    if geospatial_flag:
        gdf_points = gpd.read_file(config.geojson_prediction_path)

        gdf_points = verfify_points(gdf_points, config)

        grid_manager = GeoSpatialRasterGrid(Path(config.image_path))

        grid_manager.gdf_raster_mask.to_file(filename=vis_output_dir / f"raster_mask_{config.image_path.stem}.geojson",
                                             driver='GeoJSON')
        logger.info(f"Raster mask saved to {vis_output_dir / f'raster_mask_{config.image_path.stem}.geojson'}")

        grid_gdf = grid_manager.create_filtered_grid(points_gdf=gdf_points,
                                                                      box_size_x=config.box_size_x,
                                                                      box_size_y=config.box_size_y,
                                                                      num_empty_samples=len(gdf_points) * 0.0,
                                                                      object_centered=False,
                                                                      min_distance_pixels=0,
                                                                      overlap_ratio=0.0)

        grid_gdf_filtered = grid_gdf[grid_gdf.geometry.apply(lambda poly: gdf_points.geometry.within(poly).any())]

        logger.info(f"start slicing: {config.image_path.stem}")
        slicer_occupied = GeoSlicer(base_path=config.image_path.parent,
                                    image_name=config.image_path.name,
                                    grid=grid_gdf_filtered,
                                    output_dir=tile_output_dir)

        gdf_sliced_points = slicer_occupied.slice_annotations_regular_grid(gdf_points, grid_gdf_filtered)
        gdf_sliced_points
        occupied_tiles = slicer_occupied.slice_very_big_raster(num_chunks=len(gdf_points) // 20 + 1, num_workers=5)

        converted_tile_output_dir = tile_output_dir / "converted_tiles"
        converted_tiles = convert_tiles_to(tiles=list(slicer_occupied.gdf_slices.slice_path),
                                           format=format,
                                           output_dir=converted_tile_output_dir, )
        converted_tiles = [a for a in converted_tiles]

        samples = [fo.Sample(filepath=path) for path in converted_tiles]
        dataset.add_samples(samples)

        gdf_sliced_points["images"] = gdf_sliced_points["tile_name"].apply(
            lambda x: f"{x}.{str(format.value)}"
        )
        gdf_sliced_points.rename(columns={"local_pixel_x": "x", "local_pixel_y": "y"}, inplace=True)
        gdf_sliced_points["image_id"] = gdf_sliced_points["tile_name"].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()
        )
        predicted_images = herdnet_prediction_to_hasty(gdf_sliced_points,
                                                    hA_reference=None,
                                                    images_path=converted_tile_output_dir,
                                                   )
        logger.info(f"Converted {len(converted_tiles)} tiles to {format.name} format")
        # TODO Keep only tiles which are of interest
        hA_intermediate_path = config.output_path / f"{config.dataset_name}_intermediate_hasty.json"

        hA_reference.images = predicted_images
        hA_reference.save(hA_intermediate_path)
        config.hasty_intermediate_annotation_path = hA_intermediate_path


        dataset = submit_for_cvat_evaluation(dataset=dataset,
                                             detections=predicted_images)

        # CVAT correction, see https://docs.voxel51.com/integrations/cvat.html for documentation
        dataset.annotate(
            anno_key=config.dataset_name,
            label_field=f"detection",
            attributes=[],
            launch_editor=True,
            organization="IguanasFromAbove",
            project_name="geospatial_model_prediction_correction"
        )

        return config

    else:

        return "TODO: implement non-geospatial correction"


if __name__ == "__main__":

    # correct a model annotation
    # base_path = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021")
    # config = GeospatialDatasetCorrectionConfig(
    #     dataset_name=f"FLPC03_correction",
    #     type="points",
    #     geojson_prediction_path="/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021/detections_Flo_FLPC03_22012021.geojson",
    #     output_path=base_path,
    #     image_path=Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/cog/Flo_FLPC03_22012021.tif"),
    #     hasty_reference_annotation_path=Path("/raid/cwinkelmann/Manual_Counting/2025_08_13_iguana_reference.json")
    # )

    # correct a human annotation
    base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/Fer/Fer_FNA01-02_20122021")
    config = GeospatialDatasetCorrectionConfig(
        dataset_name=f"Fer_FNA01_02_20122021_dd_corr_evalu",
        type="points",
        geojson_prediction_path="/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/Fer/Fer_FNA01-02_20122021 counts.geojson",
        output_path=base_path,
        image_path=Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNA01-02_20122021.tif"),
        hasty_reference_annotation_path=Path("/Users/christian/data/training_data/2025_08_10_label_correction/fernandina_s_correction_hasty_corrected_1.json"),
        box_size_x=800,
        box_size_y = 800,
    )

    config.output_path.mkdir(exist_ok=True)
    config_path = base_path / f"{config.dataset_name}_config.json"
    config.save(config_path)
    vis_output_dir = base_path / "visualisation"
    vis_output_dir.mkdir(exist_ok=True)
    report_config = main(config, output_dir=base_path, vis_output_dir=vis_output_dir)

    report_config.save(config_path)

    logger.info(f"Report saved to {config_path}")
