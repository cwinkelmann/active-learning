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
from active_learning.types.Exceptions import OrthomosaicNotSetError, AnnotationFileNotSetError
from active_learning.util.hit.geospatial import batched_geospatial_correction_upload
import geopandas as gpd


def config_generator_human_ai_correction(

        df_mapping,
                                         orthomosaics_base_path: Path,
                                         predictions_a_base_path: None | Path,
                                         hasty_reference_annotation_path: Path,
                                         output_path: Path,
                                         box_size=800
                                         ):
    """
    helper script to generate config files for multiple orthomosaics
    :param df_mapping:
    :param predictions_base_path:
    :param prediction_b_base_path:
    :param hasty_reference_annotation_path:
    :param output_path:
    :param box_size:
    :return:
    """


    for idx, row in df_mapping.iterrows():

        mission_folder = row.mission_folder
        island_short = row.island_short
        orthomosaic_name = row.orthomosaic_name
        prediction_a_path = predictions_a_base_path / row.ai_prediction_name
        orthomosaic_path = orthomosaics_base_path / island_short / orthomosaic_name

        if not orthomosaic_path.exists():
            raise OrthomosaicNotSetError(f"{orthomosaic_path} does not exist")


        if not prediction_a_path.exists():
            raise AnnotationFileNotSetError(f"{prediction_a_path} does not exist")

        dataset_name = mission_folder
        predicton_b = None

        # replace spaces and special characters with underscores
        dataset_name = dataset_name.replace(" ", "_").replace("-", "_").replace(".", "_")
        # to lowercase
        dataset_name = dataset_name.lower()

        c = GeospatialDatasetCorrectionConfig(
            dataset_name=dataset_name,
            type="points",
            geojson_prediction_path=prediction_a_path,
            geojson_reference_annotation_path=predicton_b, # TODO this can be optional
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
    # input
    predictions_a_base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/AI_detection')
    # source of AI detections
    orthomosaics_base_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog')
    df_mapping_csv = pd.read_csv(Path(
        '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/documents/Studium/FIT/Master Thesis/mapping/Geospatial_Annotations/enriched_GIS_progress_report_with_stats_2025_08_30.csv'))
    hasty_reference_annotation_path = Path(
        "/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_10_11/2025_11_12_labels.json")

    # Name of the run
    dataset_name = "main_dataset_correction_2025_11_13"
    island_short_filter = "Isa"
    island_full_name = "Isabela"
    # island_full_name = "Fernandina"
    # base_path = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting_2/Analysis_of_counts/all_drone_deploy_uncorrected")
    base_path = Path(f"/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/Manual_Counting/Analysis_of_counts/correction_run_{island_full_name}")
    base_path.mkdir(parents=True, exist_ok=True)
    # base_path_corrected = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting_2/Analysis_of_counts/all_drone_deploy_corrected")
    # base_path_corrected = Path("deprecated")



    df_mapping_csv = df_mapping_csv[df_mapping_csv['island'] == island_full_name]
    output_path = base_path / f"CVAT_corrected_{island_full_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    ai_predictions = [f for f in predictions_a_base_path.glob("*.geojson") if not f.name.startswith("._")]
    mission_folders = []

    # ai_predictions = [a for a in ai_predictions if a.name.startswith("Fer_FPE03-04-05_18122021")]
    # ai_predictions = [a for a in ai_predictions if a.name.startswith("Isa_ISVB01_27012023")]
    ai_predictions = [a for a in ai_predictions if a.name.startswith("Isa_ISEB05_19012023")]
    ai_predictions = [a for a in ai_predictions if a.name.startswith(island_short_filter)]



    # mission_filter = ["ispp01_18012023", "ispda02_17012023", "ispce01_25012023", "ispc02_25012023", "ispc01_25012023",
    #                   "ispa11_12_13_15122021", "isncw02_25012023", "isncw01_25012023", "ismu01_04_05_16122021",
    #                   "islcb01_02_21012023", "isepe01_02_26012023", "iseb04_19012023", "iseb03_19012023",
    #                   "iscwn03_18012023", "iscwn02_18012023", "iscwn01_18012023", "iscwe01_27012023",
    #                   "iscw01_25012023", "iscna01_02_iscnb01_02_21012023",
    #                   "isbb03_22012023", "iscr02_26012023"
    #                   ]
    orthomosaic_mission_filter = [x.lower() for x in [
        # "FPE01-06_18122021",
        # "Fer_FPE03-04-05_18122021",
        # "Fer_FPM05_24012023"
        "Isa_ISEB05_19012023"
                      ]]
    # mission_filter = None
    # island_short_filter = "Fer"

    mapping = []
    for ai_prediction in ai_predictions:
        gdf_ai_pred = gpd.read_file(ai_prediction)
        island_short, flight_code, flight_date, suffix = ai_prediction.name.split("_")
        mission_folder = f"{flight_code}_{flight_date}"
        orthomosaic_name = f"{island_short}_{mission_folder}"
        dd, mm, year = flight_date[:2], flight_date[2:4], flight_date[4:] # '11012023'
        existing_flight_codes = list(df_mapping_csv["flight code"])

        # if island_short == island_short_filter and mission_folder not in existing_flight_codes and year == "2023":
        # if island_short == island_short_filter and mission_folder not in existing_flight_codes and mission_folder.lower() in mission_filter:
        if island_short == island_short_filter and orthomosaic_name.lower() in orthomosaic_mission_filter:
            mapping.append({
                    "mission_folder": mission_folder,
                    "island_short": island_short,
                    "orthomosaic_name": f"{island_short}_{mission_folder}.tif",
                    "ai_prediction_name": f"{island_short}_{Path(mission_folder)}_detections.geojson",
                }
                )
            pass

    df_mapping = pd.DataFrame(mapping)
    logger.info(f"Processing {len(df_mapping_csv)} orthomosaics")

    cc = config_generator_human_ai_correction(
        df_mapping=df_mapping,
        predictions_a_base_path=predictions_a_base_path,
        orthomosaics_base_path=orthomosaics_base_path,
        output_path=output_path,
        hasty_reference_annotation_path=hasty_reference_annotation_path,
        box_size=1024
    )

    cc = list(cc)

    configs = GeospatialDatasetCorrectionConfigCollection(configs=cc,
                                                          dataset_name=dataset_name,
                                                          output_path=base_path,
                                                          # corrected_path=base_path_corrected,
                                                          organization="IguanasFromAbove",
                                                          project_name=f"No_ref_{island_full_name}_2025_11_12"
                                                          )

    vis_output_dir = base_path / "visualisation"
    vis_output_dir.mkdir(exist_ok=True)

    report_configs = batched_geospatial_correction_upload(configs,
                                  output_dir=base_path,
                                  vis_output_dir=vis_output_dir,
                                  submit_to_CVAT=True,
                                  include_reference=False,
                                  delete_dataset_if_exists=True,
                                  radius=0.5,
                                  )

    logger.info(f"Report saved to {report_configs}")
