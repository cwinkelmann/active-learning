import xarray as xr
import xarray as xr
import rioxarray as rxr
import numpy as np


def netcdf_to_geotiff(netcdf_path, output_path, variable_name=None):
    """
    Convert NetCDF to GeoTIFF
    """
    # Open the NetCDF file
    ds = xr.open_dataset(netcdf_path)

    # Print available variables if not specified
    if variable_name is None:
        print("Available variables:", list(ds.data_vars))
        variable_name = input("Enter variable name: ")

    # Select the variable
    data_array = ds[variable_name]

    # Set spatial dimensions and CRS if not already set
    if 'x' in data_array.dims and 'y' in data_array.dims:
        data_array = data_array.rename({'x': 'x', 'y': 'y'})
    elif 'longitude' in data_array.dims and 'latitude' in data_array.dims:
        data_array = data_array.rename({'longitude': 'x', 'latitude': 'y'})
    elif 'lon' in data_array.dims and 'lat' in data_array.dims:
        data_array = data_array.rename({'lon': 'x', 'lat': 'y'})


    # data_array.rio.set_crs(ds.crs)
    # Set CRS if not present (assuming WGS84 for lat/lon data)
    if not hasattr(data_array, 'spatial_ref'):
        data_array = data_array.rio.set_crs("EPSG:4326")

    # Write to GeoTIFF
    data_array.rio.to_raster(output_path)
    print(f"Converted to {output_path}")




def inspect_netcdf(netcdf_path):
    """
    Inspect NetCDF file structure before conversion
    """
    ds = xr.open_dataset(netcdf_path)

    print("Dataset info:")
    print(ds.info())
    print("\nVariables:")
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dims} - {ds[var].shape}")
    print("\nCoordinates:")
    for coord in ds.coords:
        print(f"  {coord}: {ds[coord].shape}")

    # Check for CRS information
    for var in ds.data_vars:
        if hasattr(ds[var], 'spatial_ref'):
            print(f"\nCRS info for {var}: {ds[var].spatial_ref}")

    return ds


# Usage
ds = inspect_netcdf("/Volumes/2TB/SamplingIssues/galapagos_1_isl_2016.nc")

# Usage
netcdf_to_geotiff("/Volumes/2TB/SamplingIssues/galapagos_1_isl_2016.nc",
                  "/Volumes/2TB/SamplingIssues/galapagos_1_isl_2016.tif",
                  "Band1")