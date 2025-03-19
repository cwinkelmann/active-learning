import math

from enum import Enum

import pandas as pd
from pathlib import Path
from shapely.geometry.point import Point

from active_learning.database import images_data_extraction, derive_image_metadata
import geopandas as gpd
from pyproj import CRS


from active_learning.types.image_metadata import CompositeMetaData, ExifData, ExposureMode, ExposureProgram, \
    convert_to_serialisable_dataframe, get_image_gdf, DerivedImageMetaData

import unittest


class MyTestCase(unittest.TestCase):

    def assert_equal_with_enums(self, dict1, dict2):
        """Custom assertion to compare dicts with enum values."""
        normalized_dict1 = {k: v.name if isinstance(v, Enum) else v for k, v in dict1.items() if not pd.isna(v)}
        normalized_dict2 = {k: v.name if isinstance(v, Enum) else v for k, v in dict2.items() if not pd.isna(v)}

        self.assertEqual(normalized_dict1, normalized_dict2)

    def test_images_data_extraction(self):
        island_folder = Path("/Volumes/G-DRIVE/Iguanas_From_Above/fake_data/Ruegen")
        res = images_data_extraction(island_folder)


        assert res is not None
        assert res.shape == (205, 71)
        assert CRS(res.crs).to_epsg() == 4326, f"GeoDataFrame CRS {res.crs} is not equivalent to EPSG:4326"

        assert res.iloc[1].to_dict()["exposure_mode"] == ExposureMode.AUTO_EXPOSURE, "exposure_mode should be should be the ENUM Class"
        assert res.iloc[1].to_dict()["exposure_program"] == ExposureProgram.SHUTTER_PRIORITY, "exposure_program should be the ENUM Class"

        gdf_image_metadata_geojson = gpd.read_file(island_folder / "Ruegen_database.geojson")
        assert CRS(gdf_image_metadata_geojson.crs).to_epsg() == 4326, f"GeoDataFrame CRS {res.crs} is not equivalent to EPSG:4326"

        gdf_image_metadata_2 = gpd.read_parquet(island_folder / "Ruegen_database.parquet")
        assert CRS(gdf_image_metadata_2.crs).to_epsg() == 4326, f"GeoDataFrame CRS {res.crs} is not equivalent to EPSG:4326"

        assert gdf_image_metadata_2.iloc[1].to_dict()["exposure_mode"] == "AUTO_EXPOSURE", "exposure_mode should be should be string because of the serialisation"
        assert gdf_image_metadata_2.iloc[1].to_dict()["exposure_program"] == "SHUTTER_PRIORITY", "exposure_program should be a string"

        read_res = [CompositeMetaData.from_serialisable_series(s) for s in gdf_image_metadata_2.iterrows()]

        gdf_image_metadata = get_image_gdf(pd.DataFrame([x.to_series() for x in read_res]))

        assert res.iloc[1].to_dict() == gdf_image_metadata.iloc[1].to_dict()

        assert CRS(gdf_image_metadata_2.crs).to_epsg() == 4326, f"GeoDataFrame CRS {res.crs} is not equivalent to EPSG:4326"
        gdf_image_metadata_2.to_crs(epsg="32715", inplace=True)
        gdf_all = derive_image_metadata(gdf_image_metadata)


        assert gdf_all.shape == (205, 93)

        row_one = gdf_all.iloc[0].to_dict()
        del row_one["geometry"]
        obj_dIMD = [DerivedImageMetaData.from_dataframe_row(s) for s in gdf_all.iterrows()]


        self.assert_equal_with_enums( row_one, obj_dIMD[0].model_dump())

        df = pd.DataFrame([d.model_dump() for d in obj_dIMD])

        # Convert to GeoDataFrame using latitude and longitude
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        assert gdf.shape == (205, 93)
