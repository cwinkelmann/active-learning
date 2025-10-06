"""
Given the model is not perfect, we need to apply a correction factor to the model's output.

THere is user bias, i.e. some is rather overcounting or undercounting. Usuually this can't be determined
Then there is the bias of the model which would be determined.

"""
from codecs import ignore_errors

import pandas as pd
import geopandas as gpd
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_geospatial
from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.util.convenience_functions import get_tiles
from active_learning.util.evaluation.evaluation import plot_confidence_density, Evaluator, plot_error_curve
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


def simple_scaling_calibration(predictions_gdf, reference_gdf,
                               pred_column='predictions', ref_column='reference'):
    """
    Simple proportional scaling to match totals.
    """
    pred_total = predictions_gdf[pred_column].sum()
    ref_total = reference_gdf[ref_column].sum()

    scaling_factor = ref_total / pred_total

    calibrated_gdf = predictions_gdf.copy()
    calibrated_gdf[f'{pred_column}_calibrated'] = calibrated_gdf[pred_column] * scaling_factor

    print(f"Original total: {pred_total:.2f}")
    print(f"Reference total: {ref_total:.2f}")
    print(f"Scaling factor: {scaling_factor:.4f}")
    print(f"Calibrated total: {calibrated_gdf[f'{pred_column}_calibrated'].sum():.2f}")

    return calibrated_gdf, scaling_factor

def main(prediction_path,
         reference_path,
         orthomosaic_path,
         outputdir,
         vis_output_dir,
         tile_size=5000,
         radius = 1,
         confidence_threshold = 0.3):


    gdf_detections = gpd.read_file(prediction_path)
    gdf_detections.rename(columns={"images": "tile_name"}, inplace=True)
    gdf_reference = gpd.read_file(reference_path)
    # df_detections.species = "iguana_point" # if that does not show up in CVAT you will have to create it manually

    # everything should be in the right projection already, but just to be sure
    gdf_detections = project_gdfcrs(gdf_detections, orthomosaic_path)
    gdf_reference = project_gdfcrs(gdf_reference, orthomosaic_path)

    gdf_detections = gdf_detections[gdf_detections.scores > confidence_threshold]
    gdf_detections.drop(columns=["x", "y", "count_1", "count_2", "count_3", "count_4", "count_5", "count_6", "count_7"], inplace=True)
    grid_manager = GeoSpatialRasterGrid(Path(orthomosaic_path))

    grid_gdf = grid_manager.create_regular_grid(x_size=tile_size, y_size=tile_size, overlap_ratio=0)
    slicer_occupied = GeoSlicer(base_path=orthomosaic_path.parent,
                                image_name=orthomosaic_path.name,
                                grid=grid_gdf,
                                output_dir=outputdir)

    grid_gdf.to_file(vis_output_dir / f"grid_all_{orthomosaic_path.stem}.geojson", driver='GeoJSON', index=False)

    gdf_sliced_reference = slicer_occupied.slice_annotations_regular_grid(gdf_reference, grid_gdf)
    gdf_sliced_detections = slicer_occupied.slice_annotations_regular_grid(gdf_detections, grid_gdf)

    gdf_detections["orthomosaic_name"] = orthomosaic_path.stem
    gdf_reference["orthomosaic_name"] = orthomosaic_path.stem

    # TODO this should be done per tile
    df_false_positives, df_true_positives, df_false_negatives = analyse_point_detections_geospatial(
        gdf_detections=gdf_detections,
        gdf_ground_truth=gdf_reference,
        radius_m=radius,
        tile_name="orthomosaic_name",
        tile_name_prediction="orthomosaic_name",
    )
    logger.info(f"Both predictors agreed on {len(df_true_positives)}, predictor a found {len(df_false_positives)} predictor b didn't and missed {len(df_false_negatives)}")



    gdf_detections_vs_reference = zip(gdf_sliced_detections.geometry, gdf_sliced_reference.geometry)

    fig, ax = plot_confidence_density(df_false_positives,
                                      title="Confidence Score Density Distribution of False Positves")
    plt.show()

    fig, ax = plot_confidence_density(df_true_positives, title="Confidence Score Density Distribution of True Positves")
    plt.show()

    # raise ValueError("Stop here to check the density plot of false positives")

    df_concat = pd.concat([df_false_positives, df_true_positives, df_false_negatives])


    df_concat.to_csv(outputdir / f"comparison_{orthomosaic_path.stem}.csv", index=False)
    df_detections = df_concat


    # TODO draw a curve: x: confidence, y: precision, recall, f1, MAE, MSE
    ev = Evaluator(df_detections=df_detections, df_ground_truth=gdf_reference, radius=radius)
    df_recall_curve = ev.get_precition_recall_curve(values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                            # 0.95,
                                                            # 0.96, 0.97, 0.98, 0.99,
                                                            # 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999,
                                                            0.9999, 1.0])

    # Plot comprehensive view
    plot_error_curve(df_recall_curve, title="Performance Analysis")


if __name__ == '__main__':
    # a kind of hard edge case
    # prediction_path = "/raid/cwinkelmann/Manual_Counting/AI_detection/Flo_FLPC03_22012021_detections.geojson"
    # reference_path = "/raid/cwinkelmann/Manual_Counting/reference_predictions/Flo/Flo_FLPC03_22012021 counts.geojson"
    # orthomosaic_path = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/cog/Flo_FLPC03_22012021.tif")
    # outputdir = Path("/raid/cwinkelmann/Manual_Counting/temp")
    # vis_output_dir = Path("/raid/cwinkelmann/Manual_Counting/temp/plots")

    correction_config_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy")

    configs = (f for f in correction_config_path.glob("*_config.json") if not f.name.startswith("._"))

    simple_diff = []
    for config_path in configs:
        logger.info(f"Processing config {config_path}")

        config = GeospatialDatasetCorrectionConfig.load(config_path)

        prediction_path = config.geojson_prediction_path
        reference_path = config.geojson_reference_annotation_path
        hasty_intermediate_annotation_path = config.hasty_intermediate_annotation_path

        gdf_prediction = gpd.read_file(prediction_path)
        gdf_reference = gpd.read_file(reference_path)
        if hasty_intermediate_annotation_path:
            hA_intermediate = HastyAnnotationV2.from_file(hasty_intermediate_annotation_path)
            len_ii  = len(hA_intermediate.images)
            label_count  = sum(len(i.labels) for i in hA_intermediate.images)
        else:
            len_ii = 0
            label_count = 0
        dataset_name = config.dataset_name
        # orthomosaic = config.orthomosaic_path

        simple_diff.append({
            "mission": config_path.name,
            "amount_prediction": len(gdf_prediction),
            "amount_reference": len(gdf_reference),
            "amount_intermediate_tiles": len_ii,
            "amount_labels": label_count,
            "correction_config": config_path,
        })

    df_simple_diff = pd.DataFrame(simple_diff)
    df_simple_diff["diff"] = df_simple_diff["amount_reference"] - df_simple_diff["amount_prediction"]

    total_diff = df_simple_diff["diff"].sum()
    total_annotations = df_simple_diff["amount_labels"].sum()

    df_simple_diff.to_csv(correction_config_path / "correction_summary.csv", index=False)

    prediction_path = "/Volumes/2TB/work/training_data_sync/Manual_Counting/AI_detection/Flo_FLPC03_22012021_detections.geojson"
    reference_path = "/Volumes/2TB/work/training_data_sync/Manual_Counting/reference_predictions/Flo/Flo_FLPC03_22012021 counts.geojson"
    orthomosaic_path = Path("/Volumes/2TB/work/training_data_sync/Manual_Counting/Drone Deploy orthomosaics/cog/Flo_FLPC03_22012021.tif")
    outputdir = Path("/Volumes/2TB/work/training_data_sync/Manual_Counting/temp")
    vis_output_dir = Path("/Volumes/2TB/work/training_data_sync/Manual_Counting/temp/plots")
    
    tile_size = 5000  # Adjust as needed
    # predictions = Path(prediction_paths).glob("*.geojson")

    main(prediction_path,
         reference_path,
         orthomosaic_path,
         outputdir, vis_output_dir, tile_size)