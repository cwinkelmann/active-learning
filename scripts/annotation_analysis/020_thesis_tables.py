"""
just get some metadata from the datasets
"""


import typing
from loguru import logger
from matplotlib_map_utils import inset_map, indicate_extent

from active_learning.config.mapping import mission_names_filter
from active_learning.types.image_metadata import list_images
from active_learning.util.drone_flight_check import get_analysis_ready_image_metadata
from active_learning.util.mapping.helper import get_islands, find_closest_island
from active_learning.util.visualisation.drone_flights import visualise_flights







import pandas as pd
import geopandas as gpd
from pathlib import Path

from active_learning.database import images_data_extraction, derive_image_metadata, create_image_db
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path
import typing
from shapely.geometry import Point, LineString
import rasterio
from rasterio.plot import plotting_extent
import contextily as ctx
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import LineString


missions = gdf = gpd.read_file(
    '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Iguanas_From_Above_all_data.gpkg',
    layer='iguana_missions')

site_code_filter = ["FCD", "FEA", "FPE","FPM", "FWK", "FNA", "FEF", "FNJ", "FNI", "FND",
                    "FLMO", "FLBB", "FLPC", "FLSCA", "FLPA",
                    "GES", "GWA",
                    ]

missions = missions[missions['site_code'].isin(site_code_filter)].drop("geometry", axis=1)