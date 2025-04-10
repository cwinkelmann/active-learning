"""
create the outline of orthomosaics
"""

from pathlib import Path
from loguru import logger
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid

# Run the function

DD_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog")
DD_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics")
DD_mosaics = list(DD_path.glob("*/*.tif"))
DD_mosaics = [x for x in DD_mosaics if not x.stem.startswith("._")]
vis_output_dir = Path("/Volumes/2TB/Manual_Counting")
vis_output_dir = Path("/Volumes/2TB/SamplingIssues/RasterMask/MS")

problems = []

for orthomosaic_path in DD_mosaics:
    try:
        logger.info(f"Processing {orthomosaic_path}")
        grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))
        outputile = vis_output_dir / f"raster_mask_{orthomosaic_path.stem}.geojson"
        logger.info(f"Saving raster mask to {outputile}")
        grid_manager.gdf_raster_mask.to_file(filename=outputile,
                                             driver='GeoJSON')
    except Exception as e:
        logger.error(e)
        problems.append(orthomosaic_path)


print(problems)