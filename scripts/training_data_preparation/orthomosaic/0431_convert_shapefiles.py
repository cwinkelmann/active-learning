"""
The shapefiles are not very well organised, named or encoded
This script will reorganise the shapefiles into a more structured form, correct the CRS and convert them to geojson
"""

import geopandas as gpd
import pandas as pd
import typing
from loguru import logger
from pathlib import Path

from active_learning.util.projection import project_gdfcrs
from active_learning.util.rename import get_site_code

shapefile_base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Counts shp')
geojson_base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/')

# copy images to new folder
output_dir = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations")
orthomosaic_drone_deploy_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/")
orthomosaic_metashape_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics/")
progress_report_file = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/iguanacounts_progress.xlsx - raw counts.csv')

# Get all relevant files
shapefiles = [f for f in shapefile_base_path.glob('**/*counts.shp') if not f.name.startswith('.')]


# Read progress report
df_progress = pd.read_csv(progress_report_file)
# Filter for entries where Count Method is "Expert (GIS)" or "Expert(GIS)"
df_progress = df_progress[
    (df_progress["Count Method"] == "Expert (GIS)") |
    (df_progress["Count Method"] == "Expert(GIS)")
]

progress_report_file = Path(
    '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/iguanacounts_progress.xlsx - raw counts.csv')

df_progress_report = pd.read_csv(progress_report_file)

# Add the new boolean columns
df_enriched = df_progress.copy()

# Iterate through the enriched dataframe
for index, row in df_progress.iterrows():
    orthophoto_name = str(row["Orthophoto/Panorama name"]) if pd.notna(row["Orthophoto/Panorama name"]) else ""
    panorama_type = str(row["Orthophoto/Panorama"]) if pd.notna(row["Orthophoto/Panorama"]) else ""
    site_code = str(row["Site code"]) if pd.notna(row["Site code"]) else ""
    flight_code = str(row["flight code"]) if pd.notna(row["flight code"]) else ""

    if not orthophoto_name:
        logger.warning(f"Missing orthophoto name for row {index}")
        continue

    logger.info(f"Processing {orthophoto_name}")

    # Find matching geojson
    matching_shapefile = None
    for shp in geojson_files:
        if orthophoto_name in str(shp):
            matching_shapefile = shp
            break


    # Read the shapefile
    try:
        gdf = gpd.read_file(matching_shapefile)

        # Extract info from shapefile name
        shp_name = matching_shapefile.name
        island_code = shp_name.split('_')[0] if '_' in shp_name else ""

        # Add metadata to GeoDataFrame
        gdf["expert"] = row['Expert'] if pd.notna(row['Expert']) else None
        gdf["species"] = "iguana"
        gdf["island_code"] = island_code
        gdf["site_code"] = get_site_code(site_code)
        gdf["program"] = panorama_type

        # Create output directory and save GeoJSON
        output_subdir = output_dir / island_code
        output_subdir.mkdir(exist_ok=True, parents=True)
        output_file = output_subdir / f"{orthophoto_name}.geojson"

        logger.info(f"Creating {output_file.name} in {island_code}, full path: {output_file}")
        gdf.to_file(output_file, driver='GeoJSON')

        # Add to mapping
        shapefile_orthomosaic_mapping.append({
            "images_path": orthomosaic_path,
            "shp_file_path": matching_shapefile,
            "shp_name": shp_name,
            "geojson_path": output_file,
            "expert": row['Expert'] if pd.notna(row['Expert']) else None,
            "island_code": island_code,
            "island": row['Island'] if pd.notna(row['Island']) else None,
            "panorama_type": panorama_type,
            "number_of_iguanas": row['Number of iguanas']
        })

    except Exception as e:
        logger.error(f"Error processing {matching_shapefile}: {e}")
        continue

# Save mappings and missing entries
logger.info(f"Missing report entries: {missing_report_entries}")
logger.info(f"Missing orthomosaics: {missing_orthomosaics_entries}")
logger.info(f"Missing shapefiles: {missing_shapefile_entries}")

pd.DataFrame(shapefile_orthomosaic_mapping).to_csv(output_dir / 'shapefile_orthomosaic_mapping.csv')

with open(output_dir / 'missing_shapefile_report_entries.txt', 'w') as f:
    for item in missing_report_entries:
        f.write(f"{item}\n")

with open(output_dir / 'missing_orthomosaics_entries.txt', 'w') as f:
    for item in missing_orthomosaics_entries:
        f.write(f"{item}\n")

with open(output_dir / 'missing_shapefile_entries.txt', 'w') as f:
    for item in missing_shapefile_entries:
        f.write(f"{item}\n")

