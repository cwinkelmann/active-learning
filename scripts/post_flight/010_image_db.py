"""
Create a image database from a folder of images

expects a folder structure like
Iguanas_From_Above/
└── 2020_2021_2022_2023_2024/
    ├── Island_A/
    │   ├── BT01_11012023/
    │   │   ├── Island_A_BT01_DJI_0001_11012023.JPG
    │   │   ├── Island_A_BT01_DJI_0002_11012023.JPG
    │   │   └── Island_A_BT01_DJI_0003_11012023.JPG
    │   ├── BT02_15012023/
    │   │   ├── Island_A_BT02_DJI_0001_15012023.JPG
    │   │   └── Island_A_BT02_DJI_0002_15012023.JPG
    │   └── BT03_18012023/
    │       └── Island_A_BT03_DJI_0001_18012023.JPG
    ├── Island_B/
    │   ├── CR01_05022023/
    │   │   ├── Island_B_CR01_DJI_0001_05022023.JPG
    │   │   └── Island_B_CR01_DJI_0002_05022023.JPG
    │   └── CR02_08022023/
    │       └── Island_B_CR02_DJI_0001_08022023.JPG
    └── Island_C/
        ├── FL01_28012023/
        │   ├── Island_C_FL01_DJI_0001_28012023.JPG
        │   ├── Island_C_FL01_DJI_0002_28012023.JPG
        │   └── Island_C_FL01_DJI_0003_28012023.JPG
        └── FL02_02022023/
            └── Island_C_FL02_DJI_0001_02022023.JPG

"""
from loguru import logger
from pathlib import Path
from pyproj import CRS

from active_learning.database import images_data_extraction
from active_learning.types.image_metadata import ExposureMode, ExposureProgram, CompositeMetaData, \
    convert_to_serialisable_dataframe

import geopandas as gpd

from active_learning.database import derive_image_metadata


def main(base_folder: Path = None, local_epsg = "32715") -> gpd.GeoDataFrame:
    """
    Main function to create the image database from a folder of images.
    :param base_folder:
    :return:
    """


    # Generate the image metadata database
    gdf_image_metadata_geojson = images_data_extraction(base_folder)

    assert gdf_image_metadata_geojson is not None
    assert gdf_image_metadata_geojson.shape[1] == 84, f"GeoDataFrame should have 84 columns, {gdf_image_metadata_geojson.shape}"
    assert CRS(gdf_image_metadata_geojson.crs).to_epsg() == 4326, f"GeoDataFrame CRS {gdf_image_metadata_geojson.crs} is not equivalent to EPSG:4326"

    # gdf_image_metadata_geojson.to_crs(epsg="32715", inplace=True) # Galapagos UTM zone
    gdf_image_metadata_geojson.to_crs(epsg=local_epsg, inplace=True) # ETRS89/UTM Zone 33N (EPSG:25833)

    get_analysis_ready_image_metadata = derive_image_metadata(gdf_image_metadata_geojson)


    return get_analysis_ready_image_metadata

if __name__ == "__main__":
    island_folder_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024")
    # island_folder_path = Path("/Users/christian/data/missions/norway_db_test")
    # island_folder_path = Path("/Users/christian/data/missions/norway_db_test_2/")

    # get_analysis_ready_image_metadata = main(base_folder=island_folder_path, local_epsg="25833")
    get_analysis_ready_image_metadata = main(base_folder=island_folder_path, local_epsg="32715")

    get_analysis_ready_image_metadata = convert_to_serialisable_dataframe(get_analysis_ready_image_metadata)
    get_analysis_ready_image_metadata.to_parquet(island_folder_path / "database_analysis_ready.parquet")

    logger.info(f"Image metadata saved to {island_folder_path / 'database_analysis_ready.parquet'}")
