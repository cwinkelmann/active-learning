"""
HUMAN IN THE LOOP for multiple orthomosaics

Take prediction we don't have a ground truth for and double check if the prediction is right.
There are two options: 1. mark it as iguanas or 2. mark it as a partial iguana

Then prepare the output for another training round

"""
import pandas as pd
from loguru import logger
from pathlib import Path

from active_learning.config.dataset_filter import GeospatialDatasetCorrectionConfig, \
    GeospatialDatasetCorrectionConfigCollection
from active_learning.util.hit.geospatial import batched_main


def config_generator_human_ai_correction(df_mapping: pd.DataFrame,
                                         # orthomosaics_base_path: Path,

                                         predictions_base_path: Path,
                                         human_prediction_base_path: Path,
                                         hasty_reference_annotation_path: Path,
                                         output_path: Path,
                                         box_size=800
                                         ):
    for i, row in df_mapping.iterrows():
        logger.info(f"Processing {i+1}/{len(df_mapping)}: {row['Orthophoto/Panorama/3Dmodel name']}")

        if "Esp_EM03_13012021" == row['Orthophoto/Panorama/3Dmodel name']:
            pass
        if (row['HasAgisoftOrthomosaic'] == False) & (row['HasDroneDeployOrthomosaic'] == False):
            logger.warning(f"Skipping {row['shp_file_path']} as no orthomosaic is available")
            continue

        if not row['HasShapefile'] or pd.isna(row['shp_file_path']):
            logger.warning(f"Skipping {row['shp_file_path']} as no shapefile is available")
            continue

        human_prediction_path = human_prediction_base_path / row['island_code'] / Path(
            row['shp_file_path']).with_suffix(".geojson").name
        orthomosaic_path = Path(row["images_path"])

        # usually the prediction of ml model, could be of a person too
        prediction_a_path = predictions_base_path / f"{Path(row['images_path']).stem}_detections.geojson"

        if not prediction_a_path.exists():
            logger.warning(f"prediction_a_path {prediction_a_path} does not exist")
            continue

        if not human_prediction_path.exists():
            logger.info(f"Human prediction path {human_prediction_path} does not exist, skipping")

        if not orthomosaic_path.exists():
            logger.warning(f"Prediction path {orthomosaic_path} does not exist, skipping")
            continue

        dataset_name = human_prediction_path.stem

        # replace spaces and special characters with underscores
        dataset_name = dataset_name.replace(" ", "_").replace("-", "_").replace(".", "_")
        # to lowercase
        dataset_name = dataset_name.lower()

        c = GeospatialDatasetCorrectionConfig(
            dataset_name=dataset_name,
            type="points",
            geojson_prediction_path=prediction_a_path,
            geojson_reference_annotation_path=human_prediction_path,
            output_path=output_path,
            image_path=orthomosaic_path,
            hasty_reference_annotation_path=hasty_reference_annotation_path,
            box_size_x=box_size,
            box_size_y=box_size,
        )

        c_path = base_path / f"{c.dataset_name}_config.json"
        if not c_path.exists():
            c.save(c_path)
            logger.info(f"Config saved to {c_path}")
        else:
            logger.info(f"Config {c_path} already exists, skipping")

        yield c


if __name__ == "__main__":
    # base_path = Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021")
    # config = GeospatialDatasetCorrectionConfig(
    #     dataset_name=f"FLPC03_correction",
    #     type="points",
    #     geojson_prediction_path="/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLPC03_22012021/detections_Flo_FLPC03_22012021.geojson",
    #     output_path=base_path,
    #     image_path=Path("/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/cog/Flo_FLPC03_22012021.tif"),
    #     hasty_reference_annotation_path=Path("/raid/cwinkelmann/Manual_Counting/2025_08_13_iguana_reference.json")
    # )

    # Name of the
    dataset_name = "main_dataset_correction_2025_08_28"

    base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy")

    base_path_corrected = Path(
        "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/all_drone_deploy/corrected")

    hasty_reference_annotation_path = Path(
        "/Users/christian/data/training_data/2025_08_10_label_correction/fernandina_s_correction_hasty_corrected_1.json")

    # config = GeospatialDatasetCorrectionConfig(
    #     dataset_name=f"Fer_FNA01_02_20122021_ds_correction",
    #     type="points",
    #     geojson_prediction_path="/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/Fer/Fer_FNA01-02_20122021 counts.geojson",
    #     output_path=base_path,
    #     image_path=Path(
    #         "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNA01-02_20122021.tif"),
    #     hasty_reference_annotation_path=Path(
    #         "/Users/christian/data/training_data/2025_08_10_label_correction/fernandina_s_correction_hasty_corrected_1.json"),
    #     box_size_x=800,
    #     box_size_y=800,
    # )
    #
    # config.output_path.mkdir(exist_ok=True)
    # config_path = base_path / f"{config.dataset_name}_config.json"
    # config.save(config_path)
    #
    # config_b = GeospatialDatasetCorrectionConfig(
    #     dataset_name=f"Scris_SRL12_10012021",
    #     type="points",
    #     geojson_prediction_path="/Users/christian/Downloads/Scris_SRL12_10012021/Scris_SRL12_10012021 detections.shp",
    #     output_path=base_path,
    #     image_path=Path(
    #         "/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Scris/Scris_SRL12_10012021.tif"),
    #     hasty_reference_annotation_path=Path(
    #         "/Users/christian/data/training_data/2025_08_10_label_correction/fernandina_s_correction_hasty_corrected_1.json"),
    #     box_size_x=800,
    #     box_size_y=800,
    # )
    # config_path_b = base_path / f"{config_b.dataset_name}_config.json"
    # config_b.save(config_path_b)
    # config_b.output_path.mkdir(exist_ok=True)
    # config_path_b = base_path / f"{config_b.dataset_name}_config.json"
    # config_b.save(config_path_b)

    # df_mapping_csv = pd.read_csv(Path(
    #     '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/shapefile_orthomosaic_mapping.csv'))
    df_mapping_csv = pd.read_csv(Path(
        '/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/enriched_GIS_progress_report_with_stats_2025_08_30.csv'))
    # df_mapping_csv = df_mapping_csv[df_mapping_csv['island'] == 'Fernandina']
    # df_mapping_csv = df_mapping_csv[df_mapping_csv['island'] == 'Floreana']
    df_mapping_csv = df_mapping_csv[~df_mapping_csv['island'].isin(['Fernandina', 'Floreana', 'Genovesa'])]
    # df_mapping_csv = df_mapping_csv[df_mapping_csv['shp_name'] == 'Flo_FLBB01_28012023 counts.shp']
    # df_mapping_csv = df_mapping_csv[df_mapping_csv['shp_name'] == 'Fer_FWK01_20122021 counts']
    # df_mapping_csv = df_mapping_csv[df_mapping_csv['shp_name'] == 'Fer_FNA01-02_20122021 counts']

    logger.info(f"Processing {len(df_mapping_csv)} orthomosaics")

    cc = config_generator_human_ai_correction(
        df_mapping=df_mapping_csv,

        predictions_base_path=Path('/Volumes/2TB/work/training_data_sync/Manual_Counting/AI_detection'),
        human_prediction_base_path=Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations'),
        # prediction_path=Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Geospatial_Annotations/Fer/Fer_FNA01-02_20122021 counts.geojson'),
        # orthomosaic_path=Path("/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Fer/Fer_FNA01-02_20122021.tif"),
        output_path=Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/CVAT_temp"),
        hasty_reference_annotation_path=hasty_reference_annotation_path
    )

    cc = list(cc)

    configs = GeospatialDatasetCorrectionConfigCollection(configs=cc,
                                                          dataset_name=dataset_name,
                                                          output_path=base_path,
                                                          corrected_path=base_path_corrected,
                                                          organization="IguanasFromAbove",
                                                          project_name="Geospatial_Dataset_Correction_Batched_2025_08_31"
                                                          )

    vis_output_dir = base_path / "visualisation"
    vis_output_dir.mkdir(exist_ok=True)

    report_configs = batched_main(configs,
                                  output_dir=base_path,
                                  vis_output_dir=vis_output_dir,
                                  submit_to_CVAT=True,
                                  include_reference=True,
                                  delete_dataset_if_exists=True,
                                  radius=0.4
                                  )

    logger.info(f"Report saved to {report_configs}")
