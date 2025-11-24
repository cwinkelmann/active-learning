"""
look into the predictions and see how they compare to the reference data

The mmode is as follows load a set of geospatial predictions, the tile that and then

There are multiple factors:
1. just apply a global correction per orthomosaic
2. estimate the error based on tiles
3. look into the error estimation


"""
from codecs import ignore_errors

import pandas as pd
import geopandas as gpd
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.analyse_detections import analyse_point_detections_geospatial_hungarian
from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig
from active_learning.util.convenience_functions import get_tiles
from active_learning.util.evaluation.correction_factor import estimate_geospatial_correction_factor, \
    simple_scaling_calibration, global_estimate_geospatial_correction_factor, plot_density_scaling_factors
from active_learning.util.evaluation.evaluation import plot_confidence_density, Evaluator, plot_error_metrics_curve
from active_learning.util.geospatial_slice import GeoSlicer, GeoSpatialRasterGrid
from active_learning.util.projection import project_gdfcrs
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2





if __name__ == '__main__':

    orthomosaic_base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog")
    # orthomosaic_base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics")
    # orthomosaic_base_path = Path("/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog")

    outputdir = Path("./temp")
    vis_output_dir = outputdir / "plots"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = 800  # Adjust as needed

    # island_of_interest = "Isa"
    # island_of_interest = "Gen"
    # island_of_interest = "Fer"
    # island_of_interest = "Esp"
    island_of_interest = "Mar"

    # model_predictions_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Counts AI")
    model_predictions_path = Path("/Volumes/2TB/work/training_data_sync/Manual_Counting/AI_detection_dla_20251118")
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
        if island_of_interest is not None and not island_prefix == island_of_interest:
            continue
        assumed_human_predictions_path = human_predictions_path / island_prefix / model_prediction.name.replace("_detections",
                                                                                                 " counts")

        assumed_orthomosaic_predictions_path = orthomosaic_base_path / island_prefix / model_prediction.name.replace(".geojson",
                                                                                                     ".tif")
        # remove "_detections" from filename
        assumed_orthomosaic_predictions_path = assumed_orthomosaic_predictions_path.parent / assumed_orthomosaic_predictions_path.name.replace("_detections", "")

        mapping.append({
            "model_prediction": model_prediction,
            "human_prediction": assumed_human_predictions_path,
            "assumed_orthomosaic_predictions_path": assumed_orthomosaic_predictions_path,
            "orthomosaic_base_path": orthomosaic_base_path,
        }
    )


    simple_diff = []
    grid_enriched = []
    for i, m in enumerate(mapping):
        # replace _detections with ' counts' to find the human prediction


        prediction_path = m["model_prediction"]
        reference_path = m["human_prediction"]
        assumed_orthomosaic_predictions_path = m["assumed_orthomosaic_predictions_path"]

        simple_stats = {
            # island
            "mission": prediction_path.stem,
            "model_prediction": prediction_path,
            "human_prediction": reference_path,
            # "orthomosaic": assumed_orthomosaic_predictions_path / prediction_path.name.replace(".geojson", ".tif"),
        }


        if prediction_path.exists():
            logger.info(f"Processing {prediction_path}")
            gdf_predictions = gpd.read_file(prediction_path)
            simple_stats["amount_prediction"] = len(gdf_predictions)
            simple_stats["avg_score"] = gdf_predictions.scores.mean()

        else:
            logger.warning(f"Prediction path {prediction_path} does not exist, skipping")
            continue

        if reference_path and reference_path.exists():
            logger.info(f"Using reference {reference_path}")
            gdf_reference = gpd.read_file(reference_path)
            simple_stats["amount_reference"] = len(gdf_reference)
        else:
            logger.warning(f"Reference path {reference_path} does not exist, skipping")
            reference_path = None
            continue

        if assumed_orthomosaic_predictions_path and assumed_orthomosaic_predictions_path.exists() and assumed_orthomosaic_predictions_path.is_file():
            logger.info(f"Using orthomosaic {assumed_orthomosaic_predictions_path}")
        else:
            logger.warning(f"{assumed_orthomosaic_predictions_path} Orthomosaic path does not exist, skipping")
            assumed_orthomosaic_predictions_path = None
            continue

        simple_diff.append(simple_stats)

        try:
            # predictions = Path(prediction_paths).glob("*.geojson")

            data = estimate_geospatial_correction_factor(prediction_path,
                 reference_path,
                 assumed_orthomosaic_predictions_path,
                 outputdir, vis_output_dir, tile_size)

            gdf_grid_enriched = data["gdf_grid_enriched"]
            gdf_grid_enriched.to_file(vis_output_dir / f"grid_enriched_{assumed_orthomosaic_predictions_path.stem}.geojson",
                                      driver='GeoJSON', index=False)
            gdf_concat = data["gdf_concat"]
            gdf_concat.to_file(outputdir / f"pred_vs_gt_{assumed_orthomosaic_predictions_path.stem}.geojson",
                               driver='GeoJSON', index=False)

            # calibrated_gdf, scaling_factor = simple_scaling_calibration(gdf_grid_enriched)

            # density_scaling_factors = global_estimate_geospatial_correction_factor(gdf_grid_enriched)

            # plot_density_scaling_factors(density_scaling_factors.T, overall_scaling_factor=scaling_factor,
            #                              title=f'Density-Based Prediction Bias Analysis for {prediction_path.stem}',
            #                              show=True, save_path=vis_output_dir / f"density_based_prediction_bias_analysis_{prediction_path.stem}")

            grid_enriched.append(gdf_grid_enriched)
        except Exception as e:
            logger.error(f"Error processing {prediction_path}")
            logger.error(e)
        # if i > 3:
        #     break

    for ge in grid_enriched:
        ge.to_crs(epsg=32715, inplace=True)
    gdf_grid_enriched_all = gpd.GeoDataFrame(pd.concat(grid_enriched))

    gdf_grid_enriched_all.to_file(outputdir / f"grid_enriched_{island_of_interest}.geojson", driver='GeoJSON', index=False)

    density_scaling_factors = global_estimate_geospatial_correction_factor(gdf_grid_enriched_all)
    calibrated_gdf, scaling_factor = simple_scaling_calibration(gdf_grid_enriched_all)
    plot_density_scaling_factors(density_scaling_factors.T, overall_scaling_factor=scaling_factor, )

    df_simple_diff = pd.DataFrame(simple_diff)
    df_simple_diff["diff"] = df_simple_diff["amount_reference"] - df_simple_diff["amount_prediction"]


    # in percent
    df_simple_diff["diff_percent"] = df_simple_diff["diff"] / df_simple_diff["amount_reference"] * 100

    df_simple_diff.to_csv(outputdir / "correction_summary.csv", index=False)