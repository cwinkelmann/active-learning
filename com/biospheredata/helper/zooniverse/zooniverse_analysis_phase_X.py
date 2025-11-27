import glob
import json
import math
import shutil
import statistics
from pathlib import Path

import pandas as pd
from loguru import logger

from com.biospheredata.helper.zooniverse.zooniverse_analysis import get_annotation_count_stats, stats_calculation
from com.biospheredata.helper.zooniverse.zooniverse_to_yolo import process_zooniverse_phases, get_mark_overview, reformat_marks, \
    get_estimated_iguana_count_localizations, get_metadata_partitions, get_estimated_SCAN_localizations, \
    plot_zooniverse_user_marks


## TODO

## turn the images so all are aligned
## draw a diagram to make it more clear what the datasources are

# Issues
## Silouette scoring has troubles with with less then 2 clusters. It can't find out if there is only a single cluster

# Done
## have a look at 2-T2-GS-results-5th-0s.csv to filter for images with 5 volunteers or more - this thresholding might not be necessary
## prefixes in third phase are Flo_ & Esp_
## play with EPS, min samples to see if DBSCAN gets better
## summarise the counts and compare
## filter the partials


def filter_func(df: pd.DataFrame):
    # remove partials

    def filter_marks(df_marks: list):
        df_marks = pd.DataFrame(df_marks)
        return df_marks[~df_marks["tool_label"].isin(["Partial iguana"])].to_dict(orient="records")

    df["marks"] = df["marks"].apply(filter_marks)

    return df

def compare_dbscan_hyp(phase_tag, eps, min_samples):
    # location for the everything
    input_path = Path("/Users/christian/data/zooniverse")
    output_path = Path(f"/Users/christian/data/zooniverse/output/{phase_tag}_{eps}_{min_samples}/")
    plot_path = output_path
    # don't plot
    plot_path = None
    configs = {}
    ds_stats = []
    ## first phase
    #goldstandard_data = Path(
    #    "/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/expert-GS-1stphase.csv")
    configs["Iguanas 1st launch"] = {
        # classifications
        "annotations_source": input_path.joinpath("IguanasFromAbove/iguanas-from-above-classifications_2023_08_03.csv"),

        # gold standard datatable with the expert count
        "goldstandard_data": Path(
        "/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/expert-GS-1stphase.csv"),

        # which images/subject ids to consider. filters the data. output from zooniverse
        "gold_standard_image_subset":
        input_path.joinpath("Images/Zooniverse_Goldstandard_images/1-T2-GS-results-5th-0s.csv"),

        # images for plot on them
        "image_source": input_path.joinpath("Images/Zooniverse_Goldstandard_images/1st launch")

    }

    ## second phase
    configs["Iguanas 2nd launch"] = {
        # classifications
        "annotations_source": input_path.joinpath("IguanasFromAbove/iguanas-from-above-classifications_2023_08_03.csv"),

        # gold standard datatable with the expert count
        "goldstandard_data": Path(
        "/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/expert-GS-2ndphase.csv"),

        # which images/subject ids to consider. filters the data. output from zooniverse
        "gold_standard_image_subset":
        input_path.joinpath("Images/Zooniverse_Goldstandard_images/2-T2-GS-results-5th-0s.csv"),

        # images for plot on them
        "image_source": input_path.joinpath("Images/Zooniverse_Goldstandard_images/2nd launch_without_prefix")

    }

    # third phase
    configs["Iguanas 3rd launch"] = {
        # classifications
        "annotations_source": input_path.joinpath("IguanasFromAbove/iguanas-from-above-classifications_2023_08_03.csv"),

        # gold standard datatable
        "goldstandard_data": Path(
        "/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/expert-GS-3rdphase_renamed.csv"),

        # which images/subject ids to consider. filters the data.
        "gold_standard_image_subset": input_path.joinpath("Images/Zooniverse_Goldstandard_images/3-T2-GS-results-5th-0s.csv"),

        # images for plot on them
        # "image_source": input_path.joinpath("Images/Zooniverse_Goldstandard_images/3rd launch")
        "image_source": None
    }

    config = configs[phase_tag]
    # images for plot on them
    image_source = config["image_source"]
    # image_source = None

    ## get all images in the launch folder, remove the prefix and use it as a mapping
    ## remove the images so they are not considered in the process
    if image_source is not None:
        image_names = [Path(i).name for i in glob.glob(str(image_source.joinpath("**/*.jpg")), recursive=True)]
        logger.info(f"found {len(image_names)} images in {image_source}")
        ds_stats.append({"image_source": image_source, "images": len(image_names)})
    else:
        image_names = None

    # image_names = ["ESCH02-1_323.jpg"]

    output_path.mkdir(exist_ok=True)
    cache_folder = input_path.joinpath(f"cache_{phase_tag}")

    annotations_source = config["annotations_source"]
    goldstandard_expert_count_data = config["goldstandard_data"]

    df_goldstandard_expert_count = pd.read_csv(goldstandard_expert_count_data, sep=";")
    logger.info(f"found {len(df_goldstandard_expert_count)} images in {goldstandard_expert_count_data}, the expert counts")
    ds_stats.append({"goldstandard_data": goldstandard_expert_count_data, "images": len(df_goldstandard_expert_count)})

    gold_standard_image_subset = config["gold_standard_image_subset"]
    df_gold_standard_image_subset = pd.read_csv(gold_standard_image_subset, sep=";")

    logger.info(f"working with {len(df_gold_standard_image_subset)} images in {gold_standard_image_subset}, the goldstandard file.")
    ds_stats.append({"gold_standard_image_subset": gold_standard_image_subset, "images": len(df_gold_standard_image_subset)})

    merged_dataset = process_zooniverse_phases(annotations_source=annotations_source,
                                               image_source=image_source,
                                               cache_folder=cache_folder,
                                               # image_names=image_list
                                               image_names=image_names,
                                               subject_ids=df_gold_standard_image_subset.subject_id.to_list(),
                                               filter_func=filter_func, phase_tag=phase_tag
                                               )
    logger.info(f"working with {len(merged_dataset)} records after process function 'process_zooniverse_phases'")
    logger.info(f"{len(merged_dataset.image_name.unique())} images 'process_zooniverse_phases' after the filtering")
    ds_stats.append({"process_zooniverse_phases": process_zooniverse_phases, "images": len(merged_dataset.image_name.unique())})

    merged_dataset.to_csv(output_path.joinpath("processed_zooniverse_classification.csv"))
    metadata_record = merged_dataset.to_dict(orient="records")

    imagename_subject_id_map = merged_dataset[["image_name", "subject_id"]].groupby("image_name").first().reset_index(drop=False)
    imagename_subject_id_map.to_csv(output_path.joinpath("imagename_subjectid_map.csv"))

    df_gold_standard_image_subset = imagename_subject_id_map.merge(df_gold_standard_image_subset, on="subject_id")
    df_gold_standard_and_expert = df_gold_standard_image_subset.merge(df_goldstandard_expert_count, on="subject_id")
    df_gold_standard_and_expert.to_csv(output_path.joinpath("gold_standard_and_expert.csv"))


    basic_stats = []
    localizations = []
    dbscan_localizations = []
    threshold = None
    metadata_record_partitions = get_metadata_partitions(metadata_record, threshold=threshold)
    logger.info(f"working with {len(metadata_record_partitions)} images with more that {threshold} user interactions each")

    ## filter if Tool 1: Is there an iguana was ansered with yes
    for metadata_record_partition in metadata_record_partitions:
    ## TODO is it necessary to apply a Tool 2 filter
        annotations_count = get_mark_overview(metadata_record_partition)

        df_marks = reformat_marks(metadata_record_partition)

        ## TODO this only works if the images are present
        # markers_plot_path = plot_zooniverse_user_marks(metadata_record_partition,
        #                                                image_path=metadata_record_partition[0]["image_path"],
        #                                                output_path=plot_path)

        annotations_count_stats = get_annotation_count_stats(annotations_count=annotations_count,
                                                             image_name=metadata_record_partition[0]["image_name"])

        SCAN_localization = get_estimated_SCAN_localizations(df_marks=df_marks,
                                                             output_path=plot_path,
                                                             # output_path=None,
                                                             image_name=metadata_record_partition[0]["image_name"],
                                                             params=(eps, min_samples)
                                                             )
        basic_stats.append(annotations_count_stats)
        dbscan_localizations.append(SCAN_localization)

        # logger.info(f"dbscan results: {SCAN_localization}")

    basic_stats = pd.DataFrame(basic_stats)
    df_localization = pd.DataFrame(localizations)
    df_dbscan_localization = pd.DataFrame(dbscan_localizations)

    df_comparison = basic_stats.merge(df_goldstandard_expert_count, on='image_name', how='left')
    df_comparison = df_comparison.merge(df_dbscan_localization, on='image_name', how='left')
    df_comparison = df_comparison.merge(df_gold_standard_image_subset, on='image_name', how='left')

    df_exp, df_exp_sum, mse_errors = stats_calculation(df_comparison)

    df_comparison.to_csv(output_path.joinpath(f"expert-GS-{phase_tag}_cl.csv"))
    df_exp.to_csv(output_path.joinpath(f"expert-GS-{phase_tag}_cl_metrics.csv"))
    df_exp_sum.to_csv(output_path.joinpath(f"expert-GS-{phase_tag}_cl_sum.csv"))
    mse_errors.to_csv(output_path.joinpath(f"expert-GS-{phase_tag}_cl_mse_errors.csv"))
    pd.DataFrame(ds_stats).to_csv(output_path.joinpath(f"stats-{phase_tag}.csv"))

    mse_errors["eps"] = eps
    mse_errors["min_samples"] = min_samples

    return mse_errors, df_exp_sum, df_exp, df_comparison



if __name__ == "__main__":

    eps = 0.01
    min_samples = 10
    output_path = Path(f"/Users/christian/data/zooniverse/")

    phase_tag = "Iguanas 1st launch"
    # phase_tag = "Iguanas 2nd launch"
    # phase_tag = "Iguanas 3rd launch"
    results = []
    eps_variants = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    min_samples_variants = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

    #eps_variants = [0.01]
    #min_samples_variants = [2]

    params = [(eps, min_samples) for eps in eps_variants for min_samples in min_samples_variants]

    for eps, min_samples in params:
        mse_errors, df_exp_sum, df_exp, df_comparison = compare_dbscan_hyp(phase_tag=phase_tag, eps=eps, min_samples=min_samples)
        ser_mse_erros = pd.Series(mse_errors)

        results.append(pd.concat([ser_mse_erros, df_exp_sum]))

    pd.DataFrame(results).to_csv(output_path.joinpath(f"{phase_tag}_hyperparam_grid.csv"))


