"""
Given the model is not perfect, we need to apply a correction factor to the model's output.

THere is user bias, i.e. some is rather overcounting or undercounting. Usuually this can't be determined
Then there is the bias of the model which would be determined.

"""
import typing

from codecs import ignore_errors

import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_geospatial, \
    analyse_multiple_user_point_detections_geospatial, get_agreement_summary, \
    analyse_multiple_user_point_detections_geospatial_v2
from active_learning.util.convenience_functions import get_tiles
from active_learning.util.evaluation.evaluation import plot_confidence_density, Evaluator, plot_error_curve
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.projection import project_gdfcrs





def main(prediction_paths: typing.Dict[str, Path], orthomosaic_path: Path,
         output_path: Path, threshold_distance: float = 0.2, num_agreeement = 2):
    predictions = {}
    user_names = "_".join(list(prediction_paths.keys()))

    for username, prediction_path in prediction_paths.items():
        _prediction = gpd.read_file(prediction_path)
        _prediction = project_gdfcrs(_prediction, orthomosaic_path)
        predictions[username] = _prediction
        _prediction.to_file(output_path / Path(f"projected_predictions_{username}_{orthomosaic_path.stem}.geojson"), driver="GeoJSON")

    # make sure all are in the same CRS and in the CRS in meters
    gdf_agreement, gdf_disagreement, gdf_agreement_locations = analyse_multiple_user_point_detections_geospatial_v2(predictions, radius_m=threshold_distance, N_agree=num_agreeement)

    # TODO keep aggreement points of humans,
    #  correct all disagreement points of humans vs AI and gdf_agreement_locations too

    # gdf_agreement.to_file(output_path / Path(f"{orthomosaic_path.stem}_{user_names}_agreement.geojson"), driver="GeoJSON")
    gdf_disagreement.to_file(output_path / Path(f"{orthomosaic_path.stem}_{user_names}_disagreement_locations.geojson"), driver="GeoJSON")
    gdf_agreement_locations.to_file(output_path / Path(f"{orthomosaic_path.stem}_{user_names}_distance{threshold_distance}_user_th{num_agreeement}_agreement_locations.geojson"), driver="GeoJSON")

    agreement_summary = get_agreement_summary(gdf_agreement, gdf_disagreement)

    print(f"The predictors aggree on : {agreement_summary['unique_agreement_locations']} locations and have a an aggreement rate of {agreement_summary['agreement_rate']*100:.2f} %")

if __name__ == '__main__':
    # a kind of hard edge case
    # prediction_path = "/raid/cwinkelmann/Manual_Counting/AI_detection/Flo_FLPC03_22012021_detections.geojson"
    # reference_path = "/raid/cwinkelmann/Manual_Counting/reference_predictions/Flo/Flo_FLPC03_22012021 counts.geojson"
    # orthomosaic_path = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/cog/Flo_FLPC03_22012021.tif")
    # outputdir = Path("/raid/cwinkelmann/Manual_Counting/temp")
    # vis_output_dir = Path("/raid/cwinkelmann/Manual_Counting/temp/plots")

    # prediction_paths = {
    #     "ali": Path('/Users/christian/Downloads/Scris_SRL12_10012021/Scris_SRL12_10012021 counts Ali.shp'),
    #     "andrea": Path('/Users/christian/Downloads/Scris_SRL12_10012021/Scris_SRL12_10012021 counts Andrea.shp'),
    #     "robin": Path('/Users/christian/Downloads/Scris_SRL12_10012021/Scris_SRL12_10012021 counts Robin.shp'),
    #     "izzy": Path('/Users/christian/Downloads/Scris_SRL12_10012021_Izzy/iguana_training_count.shp'),
    #     # "AI": '/Users/christian/Downloads/Scris_SRL12_10012021/Scris_SRL12_10012021 detections.shp',
    # }
    # orthomosaic_path = Path(
    #     "/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Scris/Scris_SRL12_10012021.tif")
    # output_path = Path("/Users/christian/Downloads/Scris_SRL12_10012021")

    # Fernandina
    prediction_paths = {
        # "ali": Path('/Users/christian/Downloads/Fer_FPE08_18122021/Fer_FPE08_18122021 counts Ali.shp'),
        "andrea": Path('/Users/christian/Downloads/Fer_FPE08_18122021/Fer_FPE08_18122021 counts Andrea.shp'),
        # "robin": Path('/Users/christian/Downloads/Fer_FPE08_18122021/Fer_FPE08_18122021 counts Robin.shp'),
        # "izzy": Path('/Users/christian/Downloads/Fer_FPE_18122021_Izzy/iguana_training_count.shp'),
        "AI": '/Users/christian/Downloads/Fer_FPE08_18122021/Fer_FPE08_18122021 detections.shp',
    }
    orthomosaic_path = Path(
        "/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FPE08_18122021.tif")
    output_path = Path("/Users/christian/Downloads/Fer_FPE08_18122021")

    body_head_distance_threshold = 0.25  # in meters, i.e. 25 cm
    large_body_head_distance_threshold = 0.35  # in meters, i.e. 35 cm
    head_head_distance_threshold = 0.1  # in meters, i.e. 10 cm

    num_agreeement = 2 # when comparing AI to human, 1 is enough
    threshold_distance = large_body_head_distance_threshold

    tile_size = 5000  # Adjust as needed

    # predictions = Path(prediction_paths).glob("*.geojson")
    main(prediction_paths,
         orthomosaic_path,
         output_path,
         threshold_distance=threshold_distance,
         num_agreeement=num_agreeement)