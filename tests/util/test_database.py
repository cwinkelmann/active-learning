from pathlib import Path

from active_learning.database import images_data_extraction, derive_image_metadata
import geopandas as gpd


def test_images_data_extraction():
    island_folder = Path("/Volumes/G-DRIVE/Iguanas_From_Above/fake_data/Ruegen")
    res = images_data_extraction(island_folder)

    gdf_image_metadata_2 = gpd.read_parquet(island_folder / "Ruegen_database.parquet")
    gdf_image_metadata_2.to_crs(epsg="32715", inplace=True)
    gdf_all = derive_image_metadata(gdf_image_metadata_2)

    assert gdf_all.shape == (205, 96)

    row_one = gdf_all.loc[gdf_all.image_name == "Rue_RGOE01_DJI_0540_09032025.JPG"].to_dict()

    gdf_all