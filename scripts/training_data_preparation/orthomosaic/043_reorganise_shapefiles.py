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
# progress_report_file = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/iguanacounts_progress.xlsx - raw counts.csv')
progress_report_file = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/iguanacounts_progress.xlsx - raw counts_2025_08_30.csv')

# Get all relevant files
shapefiles = [f for f in shapefile_base_path.glob('**/*counts.shp') if not f.name.startswith('.')]
geojson_files = [f for f in geojson_base_path.glob('**/*counts.geojson') if not f.name.startswith('.')]

# Convert the shapefiles to geojson
if len(shapefiles) != len(geojson_files):
    logger.error(f"Number of shapefiles ({len(shapefiles)}) does not match number of geojson files ({len(geojson_files)})")
    for shp in shapefiles:
        if not any(shp.stem in str(geojson) for geojson in geojson_files):
            logger.info(f"Shapefile {shp.stem} has no matching geojson file")
            gdf = gpd.read_file(shp)

            island_code = shp.stem.split('_')[0]
            geojson_file_dir = output_dir / island_code
            geojson_file_dir.mkdir(exist_ok=True, parents=True)
            geojson_file = geojson_file_dir / f"{shp.stem}.geojson"
            if not geojson_file.exists():
                logger.info(f"Converting {shp} to {geojson_file}")
                gdf.to_file(geojson_file, driver='GeoJSON')
            else:
                logger.warning(f"Geojson file {geojson_file} already exists")

            # shapefile_orthomosaic_mapping.append({
            #     "images_path": orthomosaic_path,
            #     "shp_file_path": matching_shapefile,
            #     "shp_name": shp_name,
            #     "geojson_path": output_file,
            #     "expert": row['Expert'] if pd.notna(row['Expert']) else None,
            #     "island_code": island_code,
            #     "island": row['Island'] if pd.notna(row['Island']) else None,
            #     "panorama_type": panorama_type,
            #     "number_of_iguanas": row['Number of iguanas']
            # })
        else:
            logger.info(f"Shapefile {shp.stem} has matching geojson file")


agisoft_orthomosaics = [f for f in orthomosaic_metashape_path.glob('**/*.tif') if not f.name.startswith('.')]
drone_deploy_orthomosaics = [f for f in orthomosaic_drone_deploy_path.glob('**/*.tif') if not f.name.startswith('.')]

# Read progress report
df_progress = pd.read_csv(progress_report_file)
# Filter for entries where Count Method is "Expert (GIS)" or "Expert(GIS)"
df_progress = df_progress[
    (df_progress["Count Method"] == "Expert (GIS)") |
    (df_progress["Count Method"] == "Expert(GIS)")
]

# progress_report_file = Path(
#     '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/iguanacounts_progress.xlsx - raw counts.csv')
#
# df_progress_report = pd.read_csv(progress_report_file)

# TODO map the shapefile to progress report
# Function to check if files exist for a given row
def check_files_exist(row):
    # Get identifiers from row
    orthophoto_name = str(row["Orthophoto/Panorama/3Dmodel name"]) if pd.notna(row["Orthophoto/Panorama/3Dmodel name"]) else ""
    site_code = str(row["Site code"]) if pd.notna(row["Site code"]) else ""
    flight_code = str(row["flight code"]) if pd.notna(row["flight code"]) else ""
    date_str = str(row["Date"]).replace(".", "") if pd.notna(row["Date"]) else ""

    # Create patterns to match against filenames
    patterns = [
        orthophoto_name,
        # f"{site_code}_{flight_code}",
        #f"{site_code}{flight_code}",
        #flight_code
    ]
    patterns = [p for p in patterns if p]  # Remove empty patterns

    # Check for matching shapefile
    has_shapefile = any(
        any(pattern in str(shapefile) for pattern in patterns)
        for shapefile in shapefiles
    )

    # Check for matching Agisoft orthomosaic
    has_agisoft = any(
        any(pattern in str(ortho) for pattern in patterns)
        for ortho in agisoft_orthomosaics
    )

    # Check for matching Drone Deploy orthomosaic
    has_drone_deploy = any(
        any(pattern in str(ortho) for pattern in patterns)
        for ortho in drone_deploy_orthomosaics
    )

    return pd.Series({
        'HasShapefile': has_shapefile,
        'HasAgisoftOrthomosaic': has_agisoft,
        'HasDroneDeployOrthomosaic': has_drone_deploy
    })


# Add the new boolean columns
df_enriched = df_progress.copy()
boolean_columns = df_enriched.apply(check_files_exist, axis=1)
df_enriched = pd.concat([df_enriched, boolean_columns], axis=1)



df_enriched

# Save the enriched progress report
output_file = output_dir / 'enriched_GIS_progress_report_2025_08_30.csv'
stats_output_file = output_dir / 'enriched_GIS_progress_report_with_stats_2025_08_30.csv'
df_enriched.to_csv(output_file, index=False)

print(f"Progress report enriched and saved to {output_file}")

# Print a summary of the results
total_rows = len(df_enriched)
with_shapefile = df_enriched['HasShapefile'].sum()
with_agisoft = df_enriched['HasAgisoftOrthomosaic'].sum()
with_drone_deploy = df_enriched['HasDroneDeployOrthomosaic'].sum()

print(f"\nSummary of {total_rows} entries:")
print(f"  - Entries with shapefiles: {with_shapefile} ({with_shapefile/total_rows:.1%})")
print(f"  - Entries with Agisoft orthomosaics: {with_agisoft} ({with_agisoft/total_rows:.1%})")
print(f"  - Entries with Drone Deploy orthomosaics: {with_drone_deploy} ({with_drone_deploy/total_rows:.1%})")
print(f"  - Entries with all three file types: {(df_enriched['HasShapefile'] & df_enriched['HasAgisoftOrthomosaic'] & df_enriched['HasDroneDeployOrthomosaic']).sum()}")
print(f"  - Entries missing all file types: {((~df_enriched['HasShapefile']) & (~df_enriched['HasAgisoftOrthomosaic']) & (~df_enriched['HasDroneDeployOrthomosaic'])).sum()}")

# Create the logical condition
condition = (
    ((df_enriched['HasShapefile'] == True) & (df_enriched['HasAgisoftOrthomosaic'] == True)) |
    ((df_enriched['HasShapefile'] == True) & (df_enriched['HasDroneDeployOrthomosaic'] == True))
)
df_enriched['Number of iguanas'] = df_enriched['Number of iguanas'].astype(str)

# Replace the thousand separator (.) with empty string, then convert to float then int
# This handles values like "1.234" (meaning 1,234 in some locales)
df_enriched['Number of iguanas'] = df_enriched['Number of iguanas'].str.replace('.', '').fillna(0).astype(float).astype(int)


# Calculate the sum of iguanas where the condition is met
sum_of_iguanas = df_enriched.loc[condition, 'Number of iguanas'].sum()

print(f"Total number of iguanas: {sum_of_iguanas}")


# missing_report_entries: typing.List[str] = []
missing_orthomosaics_entries: typing.List[str] = []
shapefile_orthomosaic_mapping = []
missing_shapefile_entries: typing.List[str] = []

# Iterate through the enriched dataframe and analyse the annotations
def check_analyse_content(row):
    orthophoto_name = str(row["Orthophoto/Panorama/3Dmodel name"]) if pd.notna(row["Orthophoto/Panorama/3Dmodel name"]) else ""
    panorama_type = str(row["Orthophoto/Panorama"]) if pd.notna(row["Orthophoto/Panorama"]) else ""
    site_code = str(row["Site code"]) if pd.notna(row["Site code"]) else ""
    flight_code = str(row["flight code"]) if pd.notna(row["flight code"]) else ""



    if not orthophoto_name:
        logger.warning(f"Missing orthophoto name for row {orthophoto_name}")
    if orthophoto_name == "Fer_FNB02_19122021":
        pass

    # Find matching geojson
    matching_shapefile = None
    gdf_annotations = None
    shp_name = None
    island_code = orthophoto_name.split('_')[0] if '_' in orthophoto_name else ""

    for shp in geojson_files:
        if orthophoto_name in str(shp):
            matching_shapefile = shp
            shp_name = shp.stem
            gdf_annotations = gpd.read_file(matching_shapefile)
            break

    if matching_shapefile is None and row['HasShapefile'] == "TRUE":
        """ THIS MUST not Happen """
        logger.error(f"Shapefile not found for {orthophoto_name}")
        raise(ValueError(f"Shapefile not found for {orthophoto_name}"))
    elif matching_shapefile is None:
        # logger.error(f"Shapefile not found for {orthophoto_name}")
        missing_shapefile_entries.append(orthophoto_name)
    else:
        # logger.info(f"Shapefile found: {matching_shapefile}")
        pass

    # Determine which orthomosaic to use based on the "Orthophoto/Panorama" column
    orthomosaic_path = None
    if panorama_type == "Agisoft":
        # Look for matching Agisoft orthomosaic
        for ortho in agisoft_orthomosaics:
            if orthophoto_name in str(ortho):
                orthomosaic_path = ortho
                break
    elif panorama_type == "Drone Deploy":
        # Look for matching Drone Deploy orthomosaic
        for ortho in drone_deploy_orthomosaics:
            if orthophoto_name in str(ortho):
                orthomosaic_path = ortho
                break

    number_of_iguanas_shp = len(gdf_annotations) if gdf_annotations is not None else None
    number_of_iguanas_progress_report_file = row["Number of iguanas"]

    if number_of_iguanas_shp is not None and number_of_iguanas_shp != number_of_iguanas_progress_report_file:
        logger.error(f"Number of iguanas in shapefile ({number_of_iguanas_shp}) does not match progress report ({number_of_iguanas_progress_report_file}) for {orthophoto_name}")
        # raise ValueError(f"Number of iguanas in shapefile ({number_of_iguanas_shp}) does not match progress report ({number_of_iguanas_progress_report_file}) for {orthophoto_name}")

    # Add to mapping
    return pd.Series({
        "images_path": orthomosaic_path,
        "shp_file_path": matching_shapefile,
        "shp_name": matching_shapefile.stem if matching_shapefile else None,
        "island_code": island_code,
        "island": row['Island'] if pd.notna(row['Island']) else None,
        "panorama_type": panorama_type,
        "number_of_iguanas_shp": int(len(gdf_annotations)) if gdf_annotations is not None else None,
    })



new_columns = df_enriched.apply(check_analyse_content, axis=1)
df_enriched = pd.concat([df_enriched, new_columns], axis=1)



df_enriched.to_csv(stats_output_file, index=False)

df_enriched['number_of_iguanas_shp'].sum()
df_enriched['Number of iguanas'].sum()

print(f"Progress report enriched and saved to {stats_output_file}")



# Add entries for missing orthomosaics (either Agisoft or DroneDeploy)
df_missing_orthomosaics_entries = df_enriched[(df_enriched['HasAgisoftOrthomosaic'] == False) & (df_enriched['HasDroneDeployOrthomosaic'] == False)]

# Save mappings and missing entries
# logger.info(f"Missing report entries: {missing_report_entries}")
logger.info(f"Missing orthomosaics: {missing_orthomosaics_entries}")
logger.info(f"Missing shapefiles: {missing_shapefile_entries}")

pd.DataFrame(shapefile_orthomosaic_mapping).to_csv(output_dir / 'shapefile_orthomosaic_mapping.csv')

# with open(output_dir / 'missing_shapefile_report_entries.txt', 'w') as f:
#     for item in missing_report_entries:
#         f.write(f"{item}\n")

df_missing_orthomosaics_entries.to_csv("missing_orthomosaics_entries.csv", index=False)

with open(output_dir / 'missing_shapefile_entries.txt', 'w') as f:
    for item in missing_shapefile_entries:
        f.write(f"{item}\n")

