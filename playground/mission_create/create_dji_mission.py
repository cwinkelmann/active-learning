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

base_path = Path("/raid/cwinkelmann/work/active_learning/mapping/database/")
base_path_mapping = base_path / "mapping"
base_path_mapping.mkdir(parents=True, exist_ok=True)
output_dir = base_path_mapping / "dji_missions_kml"
output_dir.mkdir(parents=True, exist_ok=True)
flight_database_path = Path(base_path / "2020_2021_2022_2023_2024_database_analysis_ready.parquet")
CRS_utm_zone_15 = "EPSG:32715"
CRS_geographic = "EPSG:4326"
import geopandas as gpd

flight_database = gpd.read_parquet(flight_database_path).to_crs(CRS_utm_zone_15)
mission_database  = gpd.read_file(
    '/raid/cwinkelmann/work/active_learning/mapping/database/mapping/Iguanas_From_Above_all_data.gpkg',
    layer='iguana_missions')

mission_database.drop(columns=['unique_exposure_programs'], inplace=True, errors='ignore')
mission_database = mission_database.to_crs(CRS_utm_zone_15)
reproducable_missions = mission_database.copy()

# Buffer the linestrings by 10 meters to create polygons
reproducable_missions['geometry'] = reproducable_missions['geometry'].buffer(10)

# Now the geometry column contains POLYGON objects instead of LINESTRING
print(f"Original geometry type: LINESTRING")
print(f"New geometry type: {reproducable_missions.geometry.geom_type.unique()}")
print(f"\nFirst buffered polygon:")
print(reproducable_missions.iloc[0]['geometry'])

# create buffered missions from the data
# Save each polygon as an individual KML file
for idx, row in reproducable_missions.iterrows():
    # Create a single-row GeoDataFrame
    single_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=CRS_utm_zone_15).to_crs(CRS_geographic)

    # Create filename using flight_code and mission_folder
    filename = f"{row['mission_folder']}.kml"
    filename_geojson = f"{row['mission_folder']}.geojson"
    filepath = output_dir / filename
    filepath_geojson = output_dir / filename_geojson

    # Save as KML
    single_gdf.to_file(filepath, driver='KML')
    single_gdf.to_file(filepath_geojson, driver='GeoJSON')

    print(f"Saved: {filename}")

print(f"\nAll {len(mission_database)} KML files saved to '{output_dir}' directory")

selected_site_code = "FCD"
flpc_gdf = reproducable_missions[reproducable_missions['site_code'].str.upper().str.startswith(selected_site_code)].copy()

print(f"Found {len(flpc_gdf)} missions with site_code starting with FLPC")

if len(flpc_gdf) > 0:
    # Display the missions found
    print("\nMissions to merge:")
    for idx, row in flpc_gdf.iterrows():
        print(f"  - {row['flight_code']}: {row['mission_folder']}")

    # Merge all the buffered polygons into one
    merged_geometry = unary_union(flpc_gdf.geometry)

    # Create a new GeoDataFrame with the merged geometry
    merged_gdf = gpd.GeoDataFrame(
        {
            'site_code': [selected_site_code],
            'num_missions': [len(flpc_gdf)],
            'mission_list': [', '.join(flpc_gdf['flight_code'].tolist())],
            'total_images': [flpc_gdf['image_count'].sum()],
            'geometry': [merged_geometry]
        },
        geometry='geometry',
        crs=mission_database.crs
    )


    merged_gdf.to_file(os.path.join(output_dir, f'{selected_site_code}_merged_buffer.shp'))
    print(f"\nSaved merged buffer as shapefile: {output_dir}/{selected_site_code}_merged_buffer.shp")

    # Save as GeoJSON
    merged_gdf.to_file(os.path.join(output_dir, f'{selected_site_code}_merged_buffer.geojson'), driver='GeoJSON')
    print(f"Saved merged buffer as GeoJSON: {output_dir}/FLPC_merged_buffer.geojson")

    # Convert to WGS84 and save as KML
    merged_gdf_wgs84 = merged_gdf.to_crs('EPSG:4326')
    merged_gdf_wgs84.to_file(os.path.join(output_dir, f'{selected_site_code}_merged_buffer.kml'), driver='KML')
    print(f"Saved merged buffer as KML: {output_dir}/{selected_site_code}_merged_buffer.kml")

    # Optional: Save as KMZ
    import zipfile

    kml_path = output_dir / f'{selected_site_code}_merged_buffer.kml'

    print(f"Saved merged buffer as KMZ: {output_dir}/f'{selected_site_code}_merged_buffer.kmz")

    print(f"\nMerged buffer statistics:")
    print(f"  Number of missions merged: {len(flpc_gdf)}")
    print(f"  Total images: {flpc_gdf['image_count'].sum()}")
    print(f"  Area: {merged_geometry.area:.2f} square meters")

else:
    print(f"\nNo missions found with site_code starting with {selected_site_code}")

import os
import zipfile
from pathlib import Path

kml_dir = output_dir
kml_files = list(Path(kml_dir).glob('*.kml'))

print(f"Found {len(kml_files)} KML files to convert")

for kml_path in kml_files:
    kmz_path = kml_path.with_suffix('.kmz')

    with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
        kmz.write(kml_path, arcname='doc.kml')

    print(f"✓ Created: {kmz_path.name}")

print(f"\nConversion complete! {len(kml_files)} KMZ files created (KML files preserved).")