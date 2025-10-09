"""
Create a database of images
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.types.image_metadata import list_images, get_image_metadata, get_image_gdf, \
    convert_to_serialisable_dataframe, CompositeMetaData
from active_learning.util.drone_flight_check import get_analysis_ready_image_metadata, get_flight_metrics
from pyproj import CRS
"""
There is the old version: GeoreferencedImage.calculate_image_metadata
testLoadImage

And there is image_metadata.image_metadata_yaw_tilt()
Then there is flight_image_capturing_sim
"""




def create_image_db(images_path: Path = None,
                    local_epsg=32715,
                    gdf_preexisting_database: gpd.GeoDataFrame = None,
                    image_extension="JPG") -> gpd.GeoDataFrame:
    """
    Main function to create the image database from a folder of images.
    :param image_extension:
    :param gdf_preexisting_database:
    :param local_epsg:
    :param base_folder:
    :return:
    """
    if images_path is not None:
        assert images_path.is_dir(), f"{images_path} is not a directory"
        images_list = list_images(images_path, extension=image_extension, recursive=True)
    else:
        images_list = []

    if gdf_preexisting_database is not None:
        existing_files = set(gdf_preexisting_database['filepath'])
        images_list = [img for img in images_list if str(img) not in existing_files]
        logger.info(f"Found {len(existing_files)} existing files in the preexisting database, {len(images_list)} new files to process")

    # Generate the image metadata database
    image_metadata = images_data_extraction(images_list=images_list )
    if len(image_metadata) > 0:
        lst_image_metadata = [x.to_series() for x in image_metadata]
        df_image_metadata = pd.DataFrame(lst_image_metadata)

        # geospatial dataframe
        gdf_image_metadata = get_image_gdf(df_image_metadata)

        assert gdf_image_metadata is not None
        assert gdf_image_metadata.shape[
                   1] == 84, f"GeoDataFrame should have 84 columns, {gdf_image_metadata.shape}"
        assert CRS(
            gdf_image_metadata.crs).to_epsg() == 4326, f"GeoDataFrame CRS {gdf_image_metadata.crs} is not equivalent to EPSG:4326"

        gdf_image_metadata.to_crs(epsg=local_epsg, inplace=True)

    if gdf_preexisting_database is not None:
        if len(image_metadata)>0:
            gdf_image_metadata = pd.concat([gdf_preexisting_database, gdf_image_metadata], ignore_index=True)
        else:
            gdf_image_metadata = gdf_preexisting_database
        gdf_image_metadata.reset_index(drop=True, inplace=True)
        logger.info(f"Combined preexisting database with new data, total {gdf_image_metadata.shape[0]} records")

    gdf_analysis_ready_image_metadata = derive_image_metadata(gdf_image_metadata)

    return gdf_analysis_ready_image_metadata


def images_data_extraction(
                           images_list: list[Path]) -> list[CompositeMetaData]:
    """
    read all images from a folder and its subfolders, calculate metadata and store it in a pandas dataframe
    :param images_path:
    :return:
    """

    # extract the image metadata
    image_metadata = get_image_metadata(images_list)

    return image_metadata

def derive_image_metadata(gdf_image_metadata_2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate different metrics from the image metadata like distance between images, time between images, etc.
    """

    gdf_all = gdf_image_metadata_2[gdf_image_metadata_2["model"] != "MAVIC2-ENTERPRISE-ADVANCED"]  # remove the thermal drone images
    gdf_all = get_analysis_ready_image_metadata(gdf_all)

    groups = []
    grouped = gdf_all.groupby(['YYYYMMDD', 'flight_code', 'site_code'])

    for (date, site, site_code), group_data in grouped:
        gdf_group_data_metrics = get_flight_metrics(group_data)


        # Calculate center and distances

        center = gdf_group_data_metrics.geometry.buffer(15).unary_union.centroid
        distances = gdf_group_data_metrics.geometry.distance(center)

        # Filter out points more than 10,000m from center
        valid_points_mask = distances <= 5000
        outlier_count = (~valid_points_mask).sum()

        if outlier_count > 0:
            logger.warning(
                f"Flight on {date} at {site} ({site_code}): Removing {outlier_count} outlier points (max distance was {distances.max():.2f}m from center)")
            gdf_group_data_metrics = gdf_group_data_metrics[valid_points_mask].reset_index(drop=True)

        groups.append(gdf_group_data_metrics)

    gdf_all = gpd.GeoDataFrame(pd.concat(groups, ignore_index=True))

    # TODO detect which of the flighs where manual flights and which were automatic
    # remove inf values and find out where they come from
    # First replace inf/-inf with NaN
    for col in gdf_all.select_dtypes(include=np.number).columns:
        # Skip the geometry column if it exists
        if col != gdf_all._geometry_column_name:
            gdf_all[col] = gdf_all[col].replace([np.inf, -np.inf], np.nan)


    gdf_all['time_diff_seconds'] = pd.to_numeric(gdf_all['time_diff_seconds'], errors='coerce')
    gdf_all['distance_to_prev'] = pd.to_numeric(gdf_all['distance_to_prev'], errors='coerce')
    gdf_all['risk_score'] = pd.to_numeric(gdf_all['risk_score'], errors='coerce')
    gdf_all['shift_mm'] = pd.to_numeric(gdf_all['shift_mm'], errors='coerce')
    gdf_all['shift_pixels'] = pd.to_numeric(gdf_all['shift_pixels'], errors='coerce')

    return gdf_all

