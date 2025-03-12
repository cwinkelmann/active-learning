


import geopandas as gpd
import pandas as pd

gdf_2020_jan = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/database/2020 Jan_database.shp')
gdf_2021_jan = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/database/2021 Jan_database.shp')
gdf_2021_dec = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/database/2021 Dec_database.shp')
gdf_2023_jan = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/database/Jan 2023_database.shp')

gdfs = [gdf_2020_jan, gdf_2021_jan, gdf_2021_dec, gdf_2023_jan]
gdf_combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Ensure the geometry column is preserved
gdf_combined.set_geometry("geometry", inplace=True)


gdf_all = gpd.read_file('/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024_database.shp')


# find images which are in 2020 Jan but not in gdf_all
gdf_not_in_all = gdf_combined[~gdf_combined['image_hash'].isin(gdf_all['image_hash'])]

gdf_not_in_all