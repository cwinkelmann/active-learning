import geopandas as gpd




geojson = '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/Fer/Fer_FNA01-02_20122021 counts.geojson'

shpfiile = '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Counts shp/Fer_FNA01-02_20122021/Fer_FNA01-02_2012202 counts.shp'


gdf_geojson = gpd.read_file(geojson)
gdf_shp = gpd.read_file(shpfiile)


gdf_shp
gdf_geojson.to_crs('EPSG:32715')

gdf_geojson == gdf_shp