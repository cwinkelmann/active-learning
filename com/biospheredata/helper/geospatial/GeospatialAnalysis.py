from loguru import logger
from pathlib import Path

import copy

import geopandas as gpd
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString


def process_coastlines(projected_CRS):
    """
    take the manually processed coastlines and get their lenghts etc in a dataframe
    :return:
    """

    df_coastlines = []

    coastlines_path = Path("/Users/christian/PycharmProjects/object-detection-pytorch/playground/mapping/Overview")
    coastlines = {
        "coastline_Baltra.geojson",
        "coastline_Darwin.geojson",
        "coastline_Espanola.geojson",
        "coastline_Fernandina.geojson",
        "coastline_Floreana.geojson",
        "coastline_Genovesa.geojson",
        "coastline_Isabela.geojson",
        "coastline_Marchena.geojson",
        "coastline_Pinta.geojson",
        "coastline_Pinzon.geojson",
        "coastline_Rabida.geojson",
        "coastline_San_Cristobal.geojson",
        "coastline_Santa_Cruz.geojson",
        "coastline_Santa_Fe.geojson",
        "coastline_Santiago.geojson",
        "coastline_Wolf.geojson"
    }

    for coastline in coastlines:
        gdf_coastline = gpd.read_file(coastlines_path.joinpath(coastline))
        gdf_coastline = gdf_coastline.to_crs(projected_CRS)
        gdf_coastline["length"] = gdf_coastline.length
        df_coastlines.append(gdf_coastline)

    df_coastlines = pd.concat(df_coastlines)

    return df_coastlines.sort_values("name").to_latex(index=False, columns=["name", "length"])


def convert_points_to_buffer(gdf_photopoints: gpd.GeoDataFrame,
                            projected_CRS: str,
                            output_path: Path,
                            prefix: str = None) -> [gpd.GeoDataFrame]:
    """
    take a Geodataframe and convert it to a linestring
    :param gdf_photopoints:
    :return:
    """
    output_path.mkdir(exist_ok=True)

    #FIXME this is more or less implemented in GeoreferencedImageList
    gdf_photopoints = gdf_photopoints.sort_values("datetime_digitized", ascending=True)
    gdf_photopoints = gdf_photopoints.to_crs(projected_CRS)

    if prefix is not None:
        file_name = str(output_path.joinpath(f"{prefix}_point_buffer.geojson"))
    else:
        file_name = str(output_path.joinpath("point_buffer.geojson"))
    # create a buffer around each point
    gdf_photopoints["geometry"] = gdf_photopoints.buffer(15)

    with open(file_name, 'w') as f:
        gdf_photopoints.to_file(filename=file_name, driver="GeoJSON")
    dissolved_gdf_photopoints = gdf_photopoints.dissolve()
    dissolved_gdf_photopoints["images"] = gdf_photopoints["filepath"].count()
    dissolved_gdf_photopoints["area"] = dissolved_gdf_photopoints.area.to_dict()

    dissolved_gdf_photopoints.to_file(filename=str(output_path.joinpath(f"{prefix}_dissolved_point_buffer.geojson")),
                                      driver="GeoJSON")

    return dissolved_gdf_photopoints

def convert_points_to_lines(gdf_photopoints: gpd.GeoDataFrame,
                            projected_CRS: str,
                            output_path: Path,
                            prefix: str = None) -> [gpd.GeoDataFrame]:
    """
    take a Geodataframe and convert it to a linestring
    :param gdf_photopoints: 
    :return:
    """
    output_path.mkdir(exist_ok=True)

    #FIXME this is more or less implemented in GeoreferencedImageList
    gdf_photopoints = gdf_photopoints.sort_values("datetime_digitized", ascending=True)

    lines = gdf_photopoints.groupby(['mission_name'])['geometry']\
        .apply(lambda x: shapely.LineString(x.tolist()))
    lines.crs = gdf_photopoints.crs


    lines = gpd.GeoDataFrame(lines, geometry='geometry', crs="EPSG:4326")
    lines["images_taken"] = len(gdf_photopoints)
    # lines["geometry"] = lines_2
    lines = lines.to_crs(projected_CRS)
    lines.reset_index(inplace=True)

    multipoint = gdf_photopoints.groupby(['mission_name'])['geometry'] \
        .apply(lambda x: shapely.MultiPoint(x.tolist()))

    # TODO add the z-value like this



    try:
        format = "%Y-%m-%d %H:%M:%S"
        gdf_photopoints["datetime_digitized"] = pd.to_datetime(gdf_photopoints["datetime_digitized"], format=format)
    except ValueError:
        format = "%Y:%m:%d %H:%M:%S"
        logger.warning(f"Dateformat {format} was wrong, try again with ")
        gdf_photopoints["datetime_digitized"] = pd.to_datetime(gdf_photopoints["datetime_digitized"], format=format)

    lines["start_time"] = gdf_photopoints["datetime_digitized"].min()
    lines["end_time"] = gdf_photopoints["datetime_digitized"].max()
    # lines["duration"] = (lines["end_time"] - lines["start_time"]).total_seconds()

    lines["duration"] = lines.apply(lambda x: (x["end_time"] - x["start_time"]).total_seconds(), axis="columns")


    lines["avg_height"] = gdf_photopoints["height"].mean()
    ## TODO this looks like a bug
    image_gdf_projected = gdf_photopoints.to_crs(projected_CRS)

    # TODO why am I not writing the photopoints here?

    # project the data to get metrics in meter.
    # lines["geometry"] = flight_path = LineString(image_gdf_projected["geometry"])
    lines["distance"] = round(lines.length, 2)

    buffer = copy.deepcopy(lines)
    buffer["geometry"] = lines.buffer(10)

    if prefix is not None:
        file_name = str(output_path.joinpath(f"{prefix}_buffer.geojson"))
    else:
        file_name = str(output_path.joinpath("buffer.geojson"))

    buffer["area"] = buffer.area
    buffer.to_file(filename=file_name, driver="GeoJSON")

    ## TODO draw the buffer first



    if prefix is not None:
        file_name = str(output_path.joinpath(f"{prefix}_line.geojson"))
    else:
        file_name = str(output_path.joinpath("line.geojson"))
    lines.to_file(filename=file_name,
                   driver="GeoJSON")

    if prefix is not None:
        file_name = str(output_path.joinpath(f"{prefix}_multipoint.geojson"))
    else:
        file_name = str(output_path.joinpath("multipoint.geojson"))
    multipoint.to_file(filename=file_name,
                   driver="GeoJSON")

    return [lines, buffer, multipoint]


def merge_mission_aggregations(path: Path, type="buffer") -> GeoDataFrame:
    """

    :param path:
    :return:
    """
    found_aggregations = list(path.glob(f"*_{type}.geojson"))
    gdf_list = [gpd.read_file(f) for f in found_aggregations]

    return merge_multiple_geodataframes(gdf_list)

def merge_multiple_geodataframes(gdf_list):
    """
    take geodataframes and merge them into one
    :param gdf_list:
    :return:
    """
    return pd.concat(gdf_list)