"""
Create a database of images
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

from active_learning.types.image_metadata import list_images, get_image_metadata, get_image_gdf, \
    convert_to_serialisable_dataframe
from active_learning.util.drone_flight_check import get_analysis_ready_image_metadata, get_flight_metrics

"""
There is the old version: GeoreferencedImage.calculate_image_metadata
testLoadImage

And there is image_metadata.image_metadata_yaw_tilt()
Then there is flight_image_capturing_sim
"""





def images_data_extraction(images_path: Path):
    """
    read all images from a folder and its subfolders, calculate metadata and store it in a pandas dataframe
    :param images_path:
    :return:
    """
    assert images_path.is_dir(), f"{images_path} is not a directory"
    images = list_images(images_path, extension="JPG", recursive=True)
    db_name = f"{images_path.name}_database.csv" # basic metadata like exif and xmp
    image_metadata = get_image_metadata(images)

    lst_image_metadata = [x.to_series() for x in image_metadata]
    df_image_metadata = pd.DataFrame(lst_image_metadata)

    # geospatial dataframe
    gdf_image_metadata = get_image_gdf(df_image_metadata)

    gdf_image_metadata_ser = convert_to_serialisable_dataframe(gdf_image_metadata)

    gdf_image_metadata_ser.to_file(images_path / Path(db_name).with_suffix(".geojson"),
                                 driver="GeoJSON")

    gdf_image_metadata_ser.to_parquet(images_path / Path(db_name).with_suffix(".parquet"),
                       compression="snappy")

    return gdf_image_metadata

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


if __name__ == '__main__':
    island_folder = Path("/Volumes/G-DRIVE/Iguanas_From_Above/fake_data/Ruegen")
    # res = images_data_extraction(island_folder)

    gdf_image_metadata_2 = gpd.read_parquet(island_folder / "Ruegen_database.parquet")
    gdf_image_metadata_2.to_crs(epsg="32715", inplace=True)
    gdf_all = derive_image_metadata(gdf_image_metadata_2)

    gdf_all