"""
Create a image database from a folder of from the hasty zip file
These include annoations too

TODO: not only extract the images but also the derived metadata, assumed height, ,

"""
from active_learning.database import derive_image_metadata
from pathlib import Path
from pathlib import Path
from pyproj import CRS
from loguru import logger
from active_learning.database import images_data_extraction
from active_learning.types.image_metadata import ExposureMode, ExposureProgram, CompositeMetaData

import geopandas as gpd

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


def main(image_base_path: Path, geospatial_output_name: str):


    df_result = images_data_extraction(image_base_path)

    assert df_result is not None
    assert CRS(df_result.crs).to_epsg() == 4326, f"GeoDataFrame CRS {df_result.crs} is not equivalent to EPSG:4326"

    try:
        # TODO fix this part
        read_res = [CompositeMetaData.from_serialisable_series(s) for s in df_result.iterrows()]
    except Exception as e:
        logger.error(e)
    gdf_all = derive_image_metadata(df_result)

    file_name = str(image_base_path / Path(geospatial_output_name).with_suffix(".geojson"))
    gdf_all.to_file(file_name, driver="GeoJSON")

    logger.info(f"Image metadata saved to {file_name}")

if __name__ == "__main__":
    image_base_path = Path("/Users/christian/data/training_data/2025_04_18_all/unzipped_images")
    df_annoated_images = main(image_base_path=image_base_path, geospatial_output_name="hasty_annotated_images")

    annotation_file = Path("/Users/christian/data/training_data/2025_04_18_all/unzipped_hasty_annotation/labels.json")
    annotations = HastyAnnotationV2.from_file(annotation_file)

    df_annotations_flat = annotations.get_flat_df()