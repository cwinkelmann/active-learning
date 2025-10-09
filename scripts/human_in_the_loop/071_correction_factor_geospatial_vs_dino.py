"""
look into the predictions and see how they compare to the reference data

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
    """
    Main function to compare geospatial predictions with reference data and evaluate performance.
    :param prediction_path:
    :param reference_path:
    :param orthomosaic_path:
    :param outputdir:
    :param vis_output_dir:
    :param tile_size:
    :param radius:
    :param confidence_threshold:
    :return:
    """


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
    df_false_positives, df_true_positives, df_false_negatives, _ = analyse_point_detections_geospatial(
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
                                                            0.95,
                                                            # 0.96, 0.97, 0.98, 0.99,
                                                            # 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999,
                                                            0.9999, 1.0])

    # Plot comprehensive view
    plot_error_curve(df_recall_curve, title="Performance Analysis")

    return {
        "df_detections": df_detections,
        "gdf_detections": gdf_detections,
        "gdf_reference": gdf_reference,
    }


if __name__ == '__main__':

    orthomosaic_base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog")
    outputdir = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/training_data/temp")
    outputdir.mkdir(parents=True, exist_ok=True)
    vis_output_dir = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/training_data/temp/plots")
    tile_size = 800  # Adjust as needed

    model_predictions_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Counts AI")
    model_predictions = list(model_predictions_path.glob("*.geojson"))
    model_predictions.sort()

    human_predictions_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Geospatial_Annotations')
    human_predictions = list(human_predictions_path.glob("**/*.geojson"))
    human_predictions.sort()

    human_predictions_mapping = {p.name: p for p in human_predictions}

    mapping = []
    # build the mapping
    for model_prediction in model_predictions:
        island_prefix = model_prediction.name.split("_")[0]

        assumed_human_predictions_path = human_predictions_path / island_prefix / model_prediction.name.replace("_detections",
                                                                                                 " counts")

        assumed_orthomosaic_predictions_path = orthomosaic_base_path / model_prediction.name.replace(".geojson",
                                                                                                     ".tif")

        mapping.append({
            "model_prediction": model_prediction,
            "human_prediction": assumed_human_predictions_path,
            "assumed_orthomosaic_predictions_path": orthomosaic_base_path,
        }
    )
    simple_diff = []
    for m in mapping:
        # replace _detections with ' counts' to find the human prediction


        prediction_path = m["model_prediction"]
        reference_path = m["human_prediction"]
        assumed_orthomosaic_predictions_path = m["assumed_orthomosaic_predictions_path"]

        simple_stats = {
            "mission": prediction_path.stem,
            "model_prediction": prediction_path,
            "human_prediction": reference_path,
            "orthomosaic": assumed_orthomosaic_predictions_path / prediction_path.name.replace(".geojson", ".tif"),
        }


        if prediction_path.exists():
            logger.info(f"Processing {prediction_path}")
            gdf_predictions = gpd.read_file(prediction_path)
            simple_stats["amount_prediction"] = len(gdf_predictions)
        else:
            logger.warning(f"Prediction path {prediction_path} does not exist, skipping")

        if reference_path and reference_path.exists():
            logger.info(f"Using reference {reference_path}")
            gdf_reference = gpd.read_file(reference_path)
            simple_stats["amount_reference"] = len(gdf_reference)
        else:
            logger.warning(f"Reference path {reference_path} does not exist, skipping")
            reference_path = None

        if assumed_orthomosaic_predictions_path and assumed_orthomosaic_predictions_path.exists():
            logger.info(f"Using orthomosaic {assumed_orthomosaic_predictions_path}")
        else:
            logger.warning(f"Orthomosaic path {assumed_orthomosaic_predictions_path} does not exist, skipping")
            assumed_orthomosaic_predictions_path = None

        simple_diff.append(simple_stats)


        # predictions = Path(prediction_paths).glob("*.geojson")

        # main(prediction_path,
        #      reference_path,
        #      assumed_orthomosaic_predictions_path,
        #      outputdir, vis_output_dir, tile_size)

    df_simple_diff = pd.DataFrame(simple_diff)
    df_simple_diff["diff"] = df_simple_diff["amount_reference"] - df_simple_diff["amount_prediction"]


    # in percent
    df_simple_diff["diff_percent"] = df_simple_diff["diff"] / df_simple_diff["amount_reference"] * 100

    df_simple_diff.to_csv(outputdir / "correction_summary.csv", index=False)