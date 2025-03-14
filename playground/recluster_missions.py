"""
flights are often named by their island, site, then flight number
FLights like SCBB01_13012023, SCBB02_13012023, SCBB02a_13012023, SCBB03_13012023, SCBB04_13012023
like belong togeher, while SCBTC01_14012023 is not part of it, yet it might be just 300 meters away

"""
import geopandas
import geopandas as gpd
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN
import numpy as np

# gdf_all = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024_database.shp')
# TODO shapeliles suck, because of the 10 character limit
gdf_all = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024_database.geojson')
gdf_all.to('/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024_database.gpkg')
gdf_all.to_crs(epsg="32715", inplace=True)


# Ensure 'datetime' column exists
if "datetime" in gdf_all.columns:
    # Convert datetime using the correct format: YYYY:MM:DD HH:MM:SS
    gdf_all["datetime"] = pd.to_datetime(
        gdf_all["datetime"],
        format="%Y:%m:%d %H:%M:%S",  # Custom format matching your data
        errors="coerce"  # Invalid dates become NaT (null)
    )
else:
    raise KeyError("Column 'datetime' not found in dataset")


# TODO mark and remove oblique images

# gdf_all["datetime_digitized"] # not in there when shapefiles are used
# Drop rows with invalid datetime (NaT values)
gdf_all = gdf_all.dropna(subset=["datetime"])

# Extract date (without time) for clustering
gdf_all["date"] = gdf_all["datetime"].dt.date

# TODO this should be done in the database creation
# Extract X, Y coordinates
gdf_all["x"] = gdf_all.geometry.x
gdf_all["y"] = gdf_all.geometry.y

# Remove invalid coordinates (inf, NaN, or extreme values)
gdf_all = gdf_all[
    (~np.isinf(gdf_all["x"])) &
    (~np.isinf(gdf_all["y"]))
]

# Initialize cluster column
gdf_all["cluster"] = -1  # Default to -1 (unclustered)

gdf_NADIR = gdf_all[gdf_all["GimbalPitc"] < -80.0]
gdf_Oblique = gdf_all[gdf_all["GimbalPitc"] >= -80.0]

# Apply DBSCAN clustering
eps_distance = 300  # Distance threshold (in meters) for clustering
min_samples = 30  # Minimum points to form a cluster

def cluster_missions(gdf: geopandas.GeoDataFrame , eps_distance=300, min_samples=30):

    # Extract coordinates
    # coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

    # Track unique cluster ID across dates
    current_cluster_id = 0

    # Perform clustering separately for each date
    for date, gdf_date in gdf.groupby("date"):
        coords = np.array(list(zip(gdf_date.geometry.x, gdf_date.geometry.y)))
        if len(coords) > 1:  # Avoid clustering on single points
            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                logger.warning(f"Skipping clustering for {date} due to invalid coordinates")
                continue

            db = DBSCAN(eps=eps_distance, min_samples=min_samples, metric="euclidean").fit(coords)
            labels = db.labels_

            logger.info(
                f"Clustering got {len(np.unique(np.array(labels)))} clusters for {len(coords)} points on {date}, max clusters: {gdf_date['folder_nam'].nunique()}")

            # Ensure unique cluster IDs (ignore noise points with label -1)
            new_labels = []
            for label in labels:
                if label == -1:
                    new_labels.append(-1)  # Keep noise points as -1
                else:
                    new_labels.append(label + current_cluster_id)

            # Assign new unique cluster labels to the original DataFrame
            gdf.loc[gdf_date.index, "cluster"] = new_labels

            # Update the cluster ID counter
            current_cluster_id = max(new_labels) + 1 if len(new_labels) > 0 else current_cluster_id

    return gdf

gdf_cluster_NADIR = cluster_missions(gdf=gdf_NADIR, eps_distance=eps_distance, min_samples= min_samples)

# Save clustered results
gdf_cluster_NADIR.to_file("/Volumes/G-DRIVE/Iguanas_From_Above/clustered_points_nadir.shp")

gdf_cluster_Oblique = cluster_missions(gdf=gdf_Oblique, eps_distance=eps_distance, min_samples= min_samples)

# Save clustered results
gdf_cluster_Oblique.to_file("/Volumes/G-DRIVE/Iguanas_From_Above/clustered_points_oblique.shp")
