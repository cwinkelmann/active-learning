import os

from pathlib import Path

import gc
import json
import pandas as pd
import shutil
import yaml
from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd
from shapely import unary_union

from active_learning.config.dataset_filter import DataPrepReport
from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from active_learning.util.visualisation.annotation_vis import visualise_points_only
from com.biospheredata.types.status import AnnotationType
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.visualization.visualize_result import visualise_polygons, visualise_image
import geopandas as gpd

kmz_path2 = '/raid/cwinkelmann/work/active_learning/mapping/database/mapping/dji_missions_kml/FCD_merged_buffer.kmz'
gdf_2 = gpd.read_file(kmz_path2)

kmz_path = '/raid/cwinkelmann/work/active_learning/mapping/database/mapping/dji_manual/rpark-deck-2ms-gsd04.kmz'
gdf = gpd.read_file(kmz_path)



print(f"Number of features: {len(gdf)}")
print(f"CRS: {gdf.crs}")
print(f"Columns: {gdf.columns.tolist()}")
print(f"\nFirst few rows:")
print(gdf.head())

# Access the geometries
print(f"\nGeometry types: {gdf.geometry.geom_type.unique()}")

# Plot if you want
gdf.plot()