from pathlib import Path

import pytest

from active_learning.util.converter import dota2coco
from active_learning.util.evaluation.correction_factor import estimate_geospatial_correction_factor, \
    global_estimate_geospatial_correction_factor, simple_scaling_calibration, plot_density_scaling_factors


@pytest.fixture()
def dota_annoatations():
    input_data = """imagesource:GoogleEarth
    gsd:0.504225526997
    8184.0 364.0 8189.0 364.0 8188.0 375.0 8183.0 376.0 small-vehicle 1
    4825.0 1314.0 4821.0 1314.0 4821.0 1304.0 4824.0 1304.0 small-vehicle 1
    4875.0 1166.0 4873.0 1161.0 4884.0 1158.0 4885.0 1163.0 small-vehicle 1
    4867.0 1178.0 4866.0 1173.0 4876.0 1168.0 4877.0 1173.0 small-vehicle 1
    4853.0 1186.0 4851.0 1181.0 4858.0 1176.0 4861.0 1181.0 small-vehicle 1
    4841.0 1198.0 4837.0 1195.0 4844.0 1186.0 4847.0 1189.0 small-vehicle 1
    4831.0 1210.0 4827.0 1207.0 4831.0 1199.0 4834.0 1202.0 small-vehicle 1
    4851.0 1274.0 4847.0 1272.0 4847.0 1259.0 4851.0 1260.0 small-vehicle 1
    4845.0 1271.0 4839.0 1270.0 4839.0 1259.0 4843.0 1259.0 small-vehicle 1
    4825.0 1297.0 4819.0 1296.0 4820.0 1285.0 4824.0 1285.0 small-vehicle 1
    4891.0 1177.0 4891.0 1173.0 4900.0 1171.0 4901.0 1175.0 small-vehicle 1"""

    return input_data


def test_estimate_geospatial_correction_factor():
    prediction_path = Path(__file__).parent.parent / "../data/annotations/Isa_ISVP01_27012023_detections.geojson"
    reference_path = Path(__file__).parent.parent / "../data/annotations/Isa_ISVP01_27012023 counts.geojson"
    corrected_reference_path = Path(
        __file__).parent.parent / "../data/annotations/Isa_ISVP01_27012023 counts corrected.geojson"
    "/Users/christian/PycharmProjects/hnee/active_learning/tests/data/annotations/Isa_ISVP01_27012023_detections.geojson"
    assert prediction_path.exists()
    assert reference_path.exists()

    # assumed_orthomosaic_predictions_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Isa/Isa_ISVP01_27012023.tif")
    assumed_orthomosaic_predictions_path = Path(
        "/Users/christian/PycharmProjects/hnee/active_learning/tests/data/images/Isa_ISVP01_27012023.tif")
    assert assumed_orthomosaic_predictions_path.exists()

    outputdir = Path("./temp")
    vis_output_dir = outputdir / "plots"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = 800

    data = estimate_geospatial_correction_factor(prediction_path,
                                                 reference_path,
                                                 assumed_orthomosaic_predictions_path,
                                                 outputdir, vis_output_dir, tile_size,
                                                 radius=0.5)

    gdf_grid_enriched = data["gdf_grid_enriched"]
    gdf_grid_enriched.to_file(vis_output_dir / f"grid_enriched_{assumed_orthomosaic_predictions_path.stem}.geojson",
                              driver='GeoJSON', index=False)
    gdf_concat = data["gdf_concat"]
    gdf_concat.to_file(outputdir / f"pred_vs_gt_{assumed_orthomosaic_predictions_path.stem}.geojson", driver='GeoJSON', index=False)

    calibrated_gdf, scaling_factor = simple_scaling_calibration(gdf_grid_enriched)

    density_scaling_factors = global_estimate_geospatial_correction_factor(gdf_grid_enriched)

    plot_density_scaling_factors(density_scaling_factors.T, overall_scaling_factor=scaling_factor,)



    assert round(scaling_factor, 2) == 0.78


    # TODO look into the tiled correction factor

    # THis contains tiles with the human reference and the prediction

    ## do the same with the corrected data
    data_corrected = estimate_geospatial_correction_factor(prediction_path,
                                                           corrected_reference_path,
                                                           assumed_orthomosaic_predictions_path,
                                                           outputdir, vis_output_dir, tile_size,
                                                           radius=0.5)




    gdf_grid_enriched_corr = data_corrected["gdf_grid_enriched"]
    calibrated_gdf, scaling_factor_corr = simple_scaling_calibration(gdf_grid_enriched_corr)
    density_scaling_factors_corr = global_estimate_geospatial_correction_factor(gdf_grid_enriched_corr)

    assert round(scaling_factor_corr, 2) == 1.04

    gdf_grid_enriched_corr.to_file(vis_output_dir / f"grid_enriched_corr_{assumed_orthomosaic_predictions_path.stem}.geojson",
                              driver='GeoJSON', index=False)
    gdf_concat_corr = data_corrected["gdf_concat"]
    gdf_concat_corr.to_file(outputdir / f"pred_vs_gt_corr_{assumed_orthomosaic_predictions_path.stem}.geojson", driver='GeoJSON', index=False)

    # TODO look into the tiled correction factor
    plot_density_scaling_factors(density_scaling_factors_corr.T, overall_scaling_factor=scaling_factor_corr,)
