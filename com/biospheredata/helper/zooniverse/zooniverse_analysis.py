import glob
import json
import math
import shutil
import statistics
from pathlib import Path

import pandas as pd
from loguru import logger

from com.biospheredata.helper.zooniverse.zooniverse_to_yolo import process_zooniverse_phases, get_mark_overview, reformat_marks, \
    get_estimated_iguana_count_localizations, get_metadata_partitions, get_estimated_SCAN_localizations


def rename_2023_scheme_images_to_zooniverse(image_source: Path):
    ## get all images in the launch folder, remove the prefix and use it as a mapping

    image_list = glob.glob(str(image_source.joinpath("**/*.jpg")), recursive=True)

    new_name = [Path(i) for i in image_list] # the new name in the dataset is something like ABC_...
    old_name = [Path(i).parent.joinpath(Path(i).name[4:]) for i in image_list]

    df_name_mapping = pd.DataFrame.from_dict({"old_name": old_name, "new_name": new_name})

    return df_name_mapping


def copy(source, target):
    shutil.move(source, target)

def rename_from_schema(df_mapping: pd.DataFrame):
    assert sorted(list(df_mapping.columns)) == sorted(["old_name", "new_name"])

    df_mapping.apply(lambda row: copy(
        row['new_name'],
        row['old_name']),
                     axis=1)


def reformat_marks_v2(metadata_record: pd.DataFrame):
    """
    get one part from the flat structure

    :param metadata_record:
    :return:
    """

    metadata_record_groups = metadata_record.groupby("image_name")
    for group in metadata_record_groups:
        flat_struct = []

        for k in group[1]:
            for mark in group[1]["marks"]:

                marks = {
                    "x": int(mark["x"]),
                    "y": int(mark["y"]),
                }
                flat_struct.append(marks)

        df_structure = pd.DataFrame(flat_struct)

        yield df_structure


def stats_calculation(df_exp):
    """
    compare the ground truth from the gold standard
    :param df_comparison:
    :return:
    """

    df_exp = df_exp[~df_exp.median_count.isna()]

    df_exp["count_diff_median"] = df_exp.count_total - df_exp.median_count
    # df_exp["count_diff_kmeans"] = df_exp.count_total - df_exp.sillouette_count
    df_exp["count_diff_dbscan"] = df_exp.count_total - df_exp.dbscan_count

    df_exp.sort_values(by="median_count", ascending=False)

    df_exp_sum = df_exp[["count_total", "mean_count", "median_count",  # "sillouette_count",
            "dbscan_count",
            "count_diff_median",  # "count_diff_kmeans",
            "count_diff_dbscan"
            ]].sum()
    # %% md

    mse_errors = {}
    from sklearn.metrics import mean_squared_error
    mse_errors["rmse_median"] = mean_squared_error(df_exp.count_total, df_exp.median_count, squared=False)
    mse_errors["rmse_mean"] = mean_squared_error(df_exp.count_total, df_exp.mean_count, squared=False)
    # mse_errors["rmse_silloutte"] = mean_squared_error(df_exp.count_total, df_exp.sillouette_count, squared=False)
    mse_errors["rmse_dbscan"] = mean_squared_error(df_exp.count_total, df_exp.dbscan_count, squared=False)


    return df_exp, df_exp_sum, pd.Series(mse_errors)


def get_annotation_count_stats(annotations_count, image_name):
    return {
        "image_name": image_name,
        "median_count": statistics.median(annotations_count),
        "mean_count": round(statistics.mean(annotations_count), 2),
        "mode_count": statistics.mode(annotations_count)
    }


if __name__ == "__main__":
    ## TODO what is that for?
    MAX_IMAGES = 5
    eps = 0.01
    min_samples = 10
    phase_tag = "Iguanas 1st launch"

    # location for the everything
    input_path = Path("/Users/christian/data/zooniverse")
    output_path = Path(f"/Users/christian/data/zooniverse/output_{phase_tag}_{eps}_{min_samples}/")
    output_path.mkdir(exist_ok=True)
    cache_folder = input_path.joinpath("cache")

    # classifications
    annotations_source = input_path.joinpath("IguanasFromAbove/iguanas-from-above-classifications_2023_08_03.csv")

    # images from the goldstandard dataset
    image_source = input_path.joinpath("Images/Zooniverse_Goldstandard_images/2nd launch_without_prefix")
    # gold standard datatable
    goldstandard_data = Path("/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/expert-GS-1stphase.csv")
    # goldstandard_data = Path("/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/expert-GS-2ndphase.csv")
    # goldstandard_data = Path("/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/expert-GS-3rdphase.csv")
    ## TODO get the gold standard images in here
    df_goldstandard = pd.read_csv(goldstandard_data, sep=";")

    ## get all images in the launch folder, remove the prefix and use it as a mapping
    ## image_names = image_list_gold_standard = [Path(i).name for i in glob.glob(str(image_source.joinpath("**/*.jpg")), recursive=True)]
    image_names = None
    ## a preprocessed version from this zooniverse tool
    # image_subset = pd.read_csv("/Users/christian/data/zooniverse/2-T2-GS-vol-raw-data_5th.csv")
    ## alternatively use the second phase goldstandard subset
    #image_subset = pd.read_csv("/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/2-T2-GS-results-5th-0s.csv", sep=";")
    image_subset = pd.read_csv("/Users/christian/data/zooniverse/Images/Zooniverse_Goldstandard_images/1-T2-GS-results-5th-0s.csv", sep=";")
    image_subset.subject_id.to_list()
    logger.info(f"working with {len(image_subset)} images now")
    # image_names = ["FMO05-2_07.jpg", "FMO03-1_65.jpg", "FMO03-1_72.jpg", "FMO04-1_101.jpg"]
    # image_names = ["ESCG01_160.jpg"] #, "FMO03-1_65.jpg", "FMO05-2_07.jpg"]
    #image_name = "FMO03-1_65.jpg"

    def filter_func(df: pd.DataFrame):
        # remove partials

        def filter_marks(df_marks: list):
            df_marks = pd.DataFrame(df_marks)
            return df_marks[~df_marks["tool_label"].isin(["Partial iguana"])].to_dict(orient="records")

        df["marks"] = df["marks"].apply(filter_marks)

        return df



    merged_dataset = process_zooniverse_phases(annotations_source=annotations_source,
                                               image_source=image_source,
                                               cache_folder=cache_folder,
                                               # image_names=image_list
                                               image_names=image_names,
                                               subject_ids=image_subset.subject_id.to_list(),
                                               filter_func= filter_func, phase_tag=phase_tag
                                               )

    merged_dataset.to_csv(output_path.joinpath("processed_zooniverse_classification.csv"))
    pd.read_csv(output_path.joinpath("processed_zooniverse_classification.csv"))
    metadata_record = merged_dataset.to_dict(orient="records")

    basic_stats = []
    localizations = []
    dbscan_localizations = []
    for metadata_record_partition in get_metadata_partitions(metadata_record, threshold=3):
        if metadata_record_partition[0]["image_path"] is None or math.isnan(metadata_record_partition[0]["image_path"]):
            pass
        else:
            annotations_count = get_mark_overview(metadata_record_partition,
                                                  output_path=output_path
                                                  )
            df_marks = reformat_marks(metadata_record_partition)

            annotations_count_stats = get_annotation_count_stats(annotations_count=annotations_count, image_name=metadata_record_partition[0]["image_name"])

            # localization = get_estimated_iguana_count_localizations(df_marks=df_marks,
            #                                                         annotations_count=annotations_count,
            #                                                         output_path=output_path,
            #                                                         image_name=metadata_record_partition[0]["image_name"])

            SCAN_localization = get_estimated_SCAN_localizations(df_marks=df_marks,
                                                                    # output_path=output_path,
                                                                    output_path=None,
                                                                    image_name=metadata_record_partition[0]["image_name"],
                                                                    params = (eps, min_samples)
                                                                 #, (0.5, 5), (0.6, 5),
                                                                     #(0.2, 7), (0.5, 7), (0.6, 7),
                                                                     #(0.2, 9), (0.5, 9), (0.6, 9),
                                                                 #]
                                                                 )
            # localizations.append(localization)
            basic_stats.append(annotations_count_stats)
            dbscan_localizations.append(SCAN_localization)

    basic_stats = pd.DataFrame(basic_stats)
    df_localization = pd.DataFrame(localizations)
    df_dbscan_localization = pd.DataFrame(dbscan_localizations)
    print(f"result: {localizations}")

    df_comparison = basic_stats.merge(df_goldstandard, on='image_name', how='left')
    df_comparison = df_comparison.merge(df_dbscan_localization, on='image_name', how='left')


    # df_comparison = df_goldstandard.merge(basic_stats, on='image_name', how='left')
    # df_comparison = df_goldstandard.merge(df_localization, on='image_name', how='left')
    # df_comparison = df_comparison.merge(df_dbscan_localization, on='image_name', how='left')

    df_exp, df_exp_sum, mse_errors = stats_calculation(df_comparison)


    df_comparison.to_csv(output_path.joinpath("expert-GS-2ndphase_cl.csv"))
    df_exp.to_csv(output_path.joinpath("expert-GS-2ndphase_cl_metrics.csv"))
    df_exp_sum.to_csv(output_path.joinpath("expert-GS-2ndphase_cl_sum.csv"))
    mse_errors.to_csv(output_path.joinpath("expert-GS-2ndphase_cl_mse_errors.csv"))




