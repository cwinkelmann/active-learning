"""
get predictions from various models

"""
from pathlib import Path

from com.biospheredata.helper.image.manipulation.slice import GeoSlicer


# Herdnet model





if __name__ == "__main__":
    geotiff_path = Path("/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif")
    sliced_predict_geotiff(geotiff_path)