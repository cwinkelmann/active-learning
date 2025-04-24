"""
TODO split an orthomosaic into rastertiles

TODO apply the prediction model to each tile

TODO create a density map of the classification and create a geospatial dataset from this


"""
from pathlib import Path
import geopandas as gpd

from active_learning.feature_extraction.timm_logit_extraction import LogitExtractor
from active_learning.types.Exceptions import AnnotationFileNotSetError
from active_learning.types.GeospatialImageClassificationGriddedDataset import \
    GeospatialImageClassificationGriddedDataset
from active_learning.util.geospatial_slice import GeoSpatialRasterGrid, GeoSlicer
from com.biospheredata.converter.HastyConverter import ImageFormat


# TODO implement this code into: GeospatialImageClassificationGriddedDataset

def main(orthomosaic_path, output_dir, model_path, tile_size=512 ):
    # GeospatialImageClassificationGriddedDataset(images_paths=[orthomosaic_path], gridcls=GeoSpatialRasterGrid)

    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

    grid_gdf = grid_manager.create_regular_grid(x_size=tile_size, y_size=tile_size, overlap_ratio=0.5)
    # grid_gdf.to_file(output_dir / "grid.geojson", driver="GeoJSON")
    grid_gdf.to_file(output_dir / "grid.shp", driver="ESRI Shapefile")


    slicer = GeoSlicer(base_path=orthomosaic_path.parent,
                       image_name=orthomosaic_path.name,
                       grid=grid_gdf, output_dir=output_dir)
    gdf_tiles = slicer.slice_very_big_raster()

    tiles = gdf_tiles["slice_path"]

    # Initialize with a custom checkpoint

    # Initialize with a custom checkpoint
    extractor2 = LogitExtractor(
        model_name='resnet34',  # Base architecture
        checkpoint_path=model_path,  # Your trained weights
        num_classes=2,  # Number of classes in your model
    )
    results_df = extractor2.extract_from_image_list(
        image_paths=tiles,
        class_mapping=["empty", "iguana"],  # Maps index 0 to "empty" and index 1 to "iguana"

        save_path=output_dir / 'logits_results.csv'
    )

    # TODO join the tile to the grid
    # TODO visualise the logits on the map

    # TODO create a density map of the logits
    results_df
    results_df = results_df.reset_index()
    gdf_tiles["slice_name"] = gdf_tiles["slice_path"].apply(lambda x: x.name)

    joined_df = results_df.merge(
        gdf_tiles,
        left_on='index',
        right_on='slice_path',
        how='left'
    )
    gdf_tiles = gpd.GeoDataFrame(joined_df, geometry='geometry')
    gdf_tiles.to_file(output_dir / "gdf_tiles.shp", driver="ESRI Shapefile")

if __name__ == '__main__':
    orthomosaic_path = Path("/Users/christian/data/orthomosaics/FMO02_full_orthophoto.tif")


    annotations_file = None
    # annotations_file = Path('/Users/christian/data/Manual Counting/Fer_FNF02_19122021/Fer_FNF02_19122021 counts.shp')
    # annotations_file = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/Fer/Fer_FNE02_19122021 counts.geojson')
    # orthomosaic_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/DD_MS_COG_ALL/Fer/Fer_FNE02_19122021.tif")
    scale_factor = 2
    tile_size = 512
    model_path = "/Users/christian/PycharmProjects/hnee/pytorch-image-models/output/iguanas_empty_resnet34.a1_in1k/model_best.pth.tar"

    # /Volumes/G-DRIVE/DD_MS_COG_ALL_TILES
    output_dir = Path(f"/Users/christian/data/orthomosaics/herdnet_{tile_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    herdnet_annotations = []
    problematic_data_pairs = []



    # if not orthomosaic_path.name == "Esp_EGB02_12012021.tif":
    #     continue

    # island_code = orthomosaic_path.parts[-2]
    tile_folder_name = orthomosaic_path.stem

    visualise_crops = False
    format = ImageFormat.JPG

    main(orthomosaic_path, tile_size=tile_size,
         output_dir=output_dir, model_path=model_path)