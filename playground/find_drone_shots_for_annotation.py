"""
Given a marked iguana find the drone shots that nearest to the image.


"""
import pandas as pd
# get the database

from pathlib import Path
import geopandas as gpd

from active_learning.util.geospatial_slice import GeoSpatialRasterGrid
from examples.register_two_orthomosaics import georeference_image
from image_patch_finder import ImagePatchFinderLG
from util.util import visualise_image, visualise_polygons

CRS_utm_zone_15 = 32715
EPSG_WGS84 = 4326
web_mercator_projection_epsg = 3857

flight_database_path = Path(
    "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/database/2020_2021_2022_2023_2024_database_analysis_ready.parquet")
gdf_flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=EPSG_WGS84)

orthomosaic_shapefile_mapping_path = Path(
    "/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Geospatial_Annotations/enriched_GIS_progress_report_with_stats.csv")



# for each annotation find the nearest drone shot
def find_nearest_drone_shots(gdf_annotations, gdf_flight_database, n=5):
    """
    Find the nearest drone shots for each annotation in the annotations GeoDataFrame.

    :param gdf_annotations: GeoDataFrame containing annotations with geometry.
    :param gdf_flight_database: GeoDataFrame containing flight data with geometry.
    :param n: Number of nearest drone shots to return for each annotation.
    :return: GeoDataFrame with nearest drone shots for each annotation.
    """
    nearest_shots = []

    for _, annotation in gdf_annotations.iterrows():
        # Find the n nearest drone shots
        nearest = gdf_flight_database.distance(annotation.geometry).nsmallest(n)
        nearest_shots.append(nearest.index.tolist())

    gdf_annotations['nearest_drone_shots'] = nearest_shots
    return gdf_annotations


def get_annotations_database(path) -> gpd.GeoDataFrame:
    geojson_list = [
        f for f in Path(path).glob("**/*.geojson") if not f.name.startswith('.')
    ]

    annotation_gdf_list = []
    
    for geojson_file in geojson_list:
        print(f"Reading {geojson_file}")
        gdf = gpd.read_file(geojson_file)
        if 'geometry' not in gdf.columns:
            raise ValueError(f"GeoJSON {geojson_file} does not contain a 'geometry' column.")
        gdf = gdf.to_crs(epsg=CRS_utm_zone_15)
        
        gdf['island_code'] = geojson_file.parent.name  # Extract island code from filename

        gdf['annotation_path'] = str(geojson_file)  # Use the filename without extension as image path
        gdf['annotation_name'] = str(geojson_file.stem)  # Use the filename without extension as image path

        annotation_gdf_list.append(gdf[['geometry', 'island_code', 'annotation_path', "annotation_name"]])

    gdf_annotations = gpd.GeoDataFrame(
        pd.concat(annotation_gdf_list, ignore_index=True),
        crs=f"EPSG:{CRS_utm_zone_15}"
    )

    return gdf_annotations

if __name__ == "__main__":
    annotations_db_path=Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/annotations_database.parquet')

    df_mapping = pd.read_csv(orthomosaic_shapefile_mapping_path)

    tile_size = 1024

    if not annotations_db_path.exists() :
#    if True:
        gdf_annotations = get_annotations_database(path="/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations")

        gdf_annotations.to_parquet(path=annotations_db_path, index=False)
    else:
        gdf_annotations = gpd.read_parquet(annotations_db_path)


    gdf_annotations = gdf_annotations.merge(df_mapping, left_on='annotation_name', right_on="shp_name", how='left')


    gdf_annotations = gdf_annotations[gdf_annotations.Expert.notna()]
    gdf_annotations = gdf_annotations[gdf_annotations.island_code_x == "Flo"]
    gdf_annotations = gdf_annotations[gdf_annotations.annotation_name == "Flo_FLPC07_22012021 counts"]
    # gdf_annotations = gdf_annotations[gdf_annotations.annotation_name == "Flo_FLPC06_22012021 counts"]
    # gdf_annotations = gdf_annotations[gdf_annotations.Expert.notna()][:5]  # Limit to first 5 annotations for testing


    nearestshots = find_nearest_drone_shots(gdf_annotations, gdf_flight_database, n=2)

    nearestshots

    drone_images_folder = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana/FLPC07_22012021")
    output_path = drone_images_folder / "georeferenced_drone_shots"
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO crop the annotation from the orthomosaic, then find this in the nearest drone shots with the template search code
    for images_path, gdf_points in gdf_annotations.groupby('images_path'):  # Fixed syntax
        for drone_image_path in drone_images_folder.glob("*.JPG"): # TODO this is quick and dirty, should be replaced with a proper glob
            print(f"Processing image path: {images_path}")


            # TODO crop the orthomosaic using the bounds
            ipf = ImagePatchFinderLG(template_path=drone_image_path,
                                     large_image_path=images_path)

            ipf.find_patch()
            ax_i = visualise_image(image_path=ipf.large_image_path, show=False, dpi=150, title="Project Orthomosaic")
            visualise_polygons(polygons=[ipf.proj_template_polygon], ax=ax_i, show=True, color="red", linewidth=4)


            output_path_image = output_path / drone_image_path.name.replace(".JPG", ".tif")
            georeference_image(image_path=drone_image_path, orthomosaic_path=images_path, M=ipf.M, output_path=output_path_image)

