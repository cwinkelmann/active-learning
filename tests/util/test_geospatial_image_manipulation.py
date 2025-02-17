import pandas as pd
import tempfile

import json

from pathlib import Path

import pytest
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon

from active_learning.util.geospatial_image_manipulation import save_world_file_json, get_raster_crs, \
    create_regular_geospatial_raster_grid, save_grid, \
    cut_geospatial_raster_with_grid_gdal
from active_learning.util.geospatial_slice import GeoSlicer

# Define the raster path
RASTER_PATH = "/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif"

@pytest.fixture()
def FMO02_full_orthophoto_grid():
    # TODO     # TODO get the right local path
    # read data/grid.geojson from disk
    grid_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/FMO02_full_orthophoto_grid.geojson")
    grid_gdf = gpd.read_file(grid_path)

    return grid_gdf


@pytest.fixture()
def dot_annotations():
    # TODO get the right local path
    point_annotations_path = Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/annotations/FMO02.shp")
    point_annotations = gpd.read_file(point_annotations_path)

    return point_annotations


@pytest.fixture()
def orthomosaic_path():
    # TODO get the right local path
    return Path("/Users/christian/PycharmProjects/hnee/active_learning/tests/data/images/orthomosaics/FMO02_full_orthophoto.tif")

@pytest.fixture
def load_raster():
    """Fixture to open the raster and return its metadata."""
    with rasterio.open(RASTER_PATH) as src:
        return {
            "crs": src.crs.to_epsg(),
            "bounds": src.bounds,
            "transform": src.transform
        }


def test_save_world_file_json():
    """
    Test the save_world_file_json function to ensure it correctly saves a JSON world file.
    """
    # Define test parameters
    metadata = {
        'crs': 'EPSG:32715',
        'height': 28357,
        'pixel_size_x': 0.007403963402457573,
        'pixel_size_y': -0.007404039264948083,
        'rotation_x': 0.0,
        'rotation_y': 0.0,
        'top_left_x': 776931.0587583557,
        'top_left_y': 9854613.061393445,
        'width': 28772
    }

    # Run the function
    json_file_path = save_world_file_json(Path(RASTER_PATH))

    # Assertions
    assert Path(json_file_path).is_file(), "JSON file should be created"

    # Load the JSON file
    with open(json_file_path, "r") as jf:
        loaded_metadata = json.load(jf)

    # Check the loaded metadata
    assert loaded_metadata == metadata, "Loaded metadata should match input"

    print("✅ All tests passed for save_world_file_json()!")

def test_get_raster_crs():
    """
    Test the get_raster_crs function to ensure it correctly returns the CRS of a raster.
    """
    # Run the function
    crs = get_raster_crs(RASTER_PATH)

    # Assertions
    assert crs == 32715, "CRS should be EPSG:32715"

    print("\t✅ All tests passed for get_raster_crs()!")




def test_save_grid():
    """
    Test the save_grid function to ensure it correctly saves a grid to a file.
    """
    # Define test parameters
    grid_gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    output_path = Path("data/grid.geojson")

    # Run the function
    save_grid(grid_gdf, output_path)

    # Assertions
    assert output_path.is_file(), "Output file should be created"

    # Load the grid file
    loaded_grid = gpd.read_file(output_path)

    # Check the loaded grid
    assert loaded_grid.equals(grid_gdf), "Loaded grid should match input"

    print("\t✅ All tests passed for save_grid()!")

def test_create_regular_geospatial_raster_grid(load_raster):
    """
    Test the create_vector_grid function to ensure it correctly creates grid polygons over the raster.
    """
    # Define test parameters
    x_size = 5000  # Tile width in raster CRS units
    y_size = 5000  # Tile height in raster CRS units
    overlap_ratio = 0.1  # 10% overlap


    # Run the function
    grid_gdf = create_regular_geospatial_raster_grid(full_image_path=Path(RASTER_PATH),
                                                     x_size=x_size,
                                                     y_size=y_size,
                                                     overlap_ratio=0.0)

    # Assertions
    assert isinstance(grid_gdf, gpd.GeoDataFrame), "Output should be a GeoDataFrame"
    assert not grid_gdf.empty, "Grid should contain polygons"

    # Validate CRS
    assert grid_gdf.crs.to_epsg() == load_raster["crs"], "Grid CRS should match raster CRS"

    # Ensure the number of grid cells is reasonable
    assert len(grid_gdf) == 36, "Grid should have more than 10 tiles"
    assert len(grid_gdf) < 5000, "Grid should not be excessively large"

    # Ensure all geometries are valid
    assert all(grid_gdf.is_valid), "All grid polygons should be valid"

    print("\t✅ All tests passed for create_vector_grid()!")





def test_cut_geospatial_raster_with_grid():
    """
    Test the cut_geospatial_raster_with_grid function to ensure it correctly cuts a raster using a grid.
    """
    # Define test parameters
    output_dir = Path("data/tiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the function
    grid_gdf = create_regular_geospatial_raster_grid(full_image_path=Path(RASTER_PATH),
                                                     x_size=5120,
                                                     y_size=5120,
                                                     overlap_ratio=0.0)

    save_grid(grid_gdf, Path("data/grid.geojson"))

    # Run the function
    cut_geospatial_raster_with_grid_gdal(raster_path=Path(RASTER_PATH),
                                    grid_gdf=grid_gdf,
                                    output_dir=output_dir)

    # Check the output directory
    assert len(list(output_dir.glob("*.tif"))) == len(grid_gdf), "Number of output tiles should match grid"

    print("\t✅ All tests passed for cut_geospatial_raster_with_grid()!")




def test_GeoSlicer(FMO02_full_orthophoto_grid: gpd.GeoDataFrame, orthomosaic_path: Path,
                   dot_annotations: gpd.GeoDataFrame):
    """
    With normal images the process was to create a grid and then crop out the images with that.
    If I convert to jpg, cut out there is no proper way to convert back to geotiff
    :return:
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a GeoSlicer instance
        temp_dir = Path(temp_dir)

        slicer = GeoSlicer(base_path=orthomosaic_path.parent,
                           image_name=orthomosaic_path.name,
                           grid=FMO02_full_orthophoto_grid, output_dir=temp_dir)
        tiles = slicer.slice_very_big_raster()
        print(tiles)
        assert len(tiles) == 36, "only one slice should be created"

        dot_annotations_path = slicer.slice_annotations(dot_annotations)
        df_project_annotations = pd.read_csv(dot_annotations_path)

        assert len(df_project_annotations) == 10, "only one slice should be created"

