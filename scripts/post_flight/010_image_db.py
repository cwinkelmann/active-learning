"""
Create a image database from a folder of images
"""
from active_learning.database import derive_image_metadata


def main():
    from pathlib import Path
    from pyproj import CRS

    from active_learning.database import images_data_extraction
    from active_learning.types.image_metadata import ExposureMode, ExposureProgram, CompositeMetaData

    import geopandas as gpd

    island_folder = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024")

    res = images_data_extraction(island_folder)

    assert res is not None
    assert res.shape == (205, 71)
    assert CRS(res.crs).to_epsg() == 4326, f"GeoDataFrame CRS {res.crs} is not equivalent to EPSG:4326"

    assert res.iloc[1].to_dict()[
               "exposure_mode"] == ExposureMode.AUTO_EXPOSURE, "exposure_mode should be should be the ENUM Class"
    assert res.iloc[1].to_dict()[
               "exposure_program"] == ExposureProgram.SHUTTER_PRIORITY, "exposure_program should be the ENUM Class"

    gdf_image_metadata_geojson = gpd.read_file(island_folder / "Ruegen_database.geojson")
    assert CRS(
        gdf_image_metadata_geojson.crs).to_epsg() == 4326, f"GeoDataFrame CRS {res.crs} is not equivalent to EPSG:4326"

    gdf_image_metadata_2 = gpd.read_parquet(island_folder / "Ruegen_database.parquet")
    assert CRS(gdf_image_metadata_2.crs).to_epsg() == 4326, f"GeoDataFrame CRS {res.crs} is not equivalent to EPSG:4326"

    assert gdf_image_metadata_2.iloc[1].to_dict()[
               "exposure_mode"] == "AUTO_EXPOSURE", "exposure_mode should be should be string because of the serialisation"
    assert gdf_image_metadata_2.iloc[1].to_dict()[
               "exposure_program"] == "SHUTTER_PRIORITY", "exposure_program should be a string"

    read_res = [CompositeMetaData.from_serialisable_series(s) for s in gdf_image_metadata_2.iterrows()]

    derive_image_metadata

if __name__ == "__main__":
    main()
