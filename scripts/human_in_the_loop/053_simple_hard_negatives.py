"""
Take Detections from a model, compare them with ground truth and display them in a FiftyOne Dataset
every detection which is not in the Ground Truth is considered a hard negative and can be used for
hard negative mining in the next training round.

"""
import numpy as np
import typing

import PIL.Image
import pandas as pd
from loguru import logger
from pathlib import Path


from active_learning.analyse_detections import analyse_point_detections_greedy
# import pytest

from active_learning.util.converter import herdnet_prediction_to_hasty
from active_learning.util.evaluation.evaluation import evaluate_in_fifty_one, Evaluator, plot_confidence_density, \
    plot_error_curve, plot_single_metric_curve, plot_comprehensive_curves, plot_species_detection_analysis
from active_learning.util.image_manipulation import crop_out_images_v3
from active_learning.util.visualisation.draw import draw_text, draw_thumbnail
from com.biospheredata.converter.Annotation import project_point_to_crop
import shapely
import fiftyone as fo

from com.biospheredata.types.status import AnnotationType, LabelingStatus
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2, AnnotatedImage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def simple_hard_negative(base_path, df_detections, herdnet_annotation_name,
                              images_path, suffix, radius, 
                              CONFIDENCE_THRESHOLD, dataset_name):
    
    visualisations_path = base_path / "visualisations"
    visualisations_path.mkdir(exist_ok=True, parents=True)
    IL_detections = herdnet_prediction_to_hasty(df_detections, images_path)

    image_list_all = [i.image_name for i in IL_detections]

    df_ground_truth = pd.read_csv(base_path / herdnet_annotation_name)

    images = list(images_path.glob(f"*.{suffix}"))
    if len(images) == 0:
        raise FileNotFoundError("No images found in: " + images_path)

    df_detections = df_detections[df_detections.scores > CONFIDENCE_THRESHOLD]

    df_false_positives, df_true_positives, df_false_negatives, gdf_ground_truth = analyse_point_detections_greedy(
        df_detections=df_detections,
        df_ground_truth=df_ground_truth,
        radius=radius,
        image_list=image_list_all
    )

    logger.info(f"False Positives: {len(df_false_positives)} True Positives: {len(df_true_positives)}, "
                f"False Negatives: {len(df_false_negatives)}, Ground Truth: {len(df_ground_truth)}")

    df_false_positives["species"] = "hard_negative"
    df_fp_hc = df_false_positives[df_false_positives.scores > CONFIDENCE_THRESHOLD]

    IL_fp_detections = herdnet_prediction_to_hasty(df_fp_hc, images_path)

    AI_fp_detections = []
    for il in IL_fp_detections:
        pred_image = AnnotatedImage(**il.model_dump(), dataset_name=dataset_name,
                                    image_status=LabelingStatus.COMPLETED)
        AI_fp_detections.append(pred_image)

    return AI_fp_detections

    # images_set = [images_path / i.image_name for i in hA_ground_truth.images]

    # evaluate_in_fifty_one(dataset_name,
    #                       images_set,
    #                       hA_ground_truth,
    #                       IL_fp_detections,
    #                       IL_fn_detections,
    #                       IL_tp_detections,
    #                       type=AnnotationType.KEYPOINT,)


if __name__ == "__main__":


    ds= "Genovesa"
    if ds == "Eikelboom2019":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/eikelboom2019/eikelboom_512_overlap_0_ebFalse/eikelboom_test/test/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-05/10-14-15/detections.csv') # dla102

        rename_species = {
            "Buffalo": "Elephant",
            "Alcelaphinae": "Giraffe",
            "Kob": "Zebra"
        }
        df_detections['species'] = df_detections['species'].replace(rename_species)
        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"

    elif ds == "Genovesa":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Genovesa_detection/val/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-19/13-35-05/detections.csv') # dla102
        hA_reference = Path(
            "/raid/cwinkelmann/training_data/iguana/2025_10_11/unzipped_hasty_annotation/2025_10_11_labels.json")

        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"
        
    elif ds == "Genovesa_crop":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/iguana/2025_10_11/Genovesa_detection/val/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-19/13-49-19/detections.csv') # dla102


        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "crops_512_numNone_overlap0"
        
    elif ds == "delplanque2023":
        # Path of the base directory where the images and annotations are stored which we want to correct
        base_path = Path(f'/raid/cwinkelmann/training_data/delplanque2023/')
        ## On full size original images
        df_detections = pd.read_csv('/raid/cwinkelmann/herdnet/outputs/2025-10-19/14-02-23/detections.csv')
        hA_reference = Path(
            "/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/delplanque_hasty.json")

        # hasty_annotation_name = 'hasty_format_full_size.json'
        herdnet_annotation_name = 'herdnet_format.csv'
        images_path = base_path / "Default"
    
    else:
        raise ValueError("Unknown dataset: " + ds)
        # /raid/cwinkelmann/herdnet/outputs/2025-10-25/10-11-12
    suffix = "JPG"

    # hA_ground_truth_path = base_path / hasty_annotation_name
    # hA_ground_truth = HastyAnnotationV2.from_file(hA_ground_truth_path)

    # ## On cropped images
    # df_detections = pd.read_csv('/Users/christian/PycharmProjects/hnee/HerdNet/data/inference_21-58-47/detections.csv')
    # hasty_annotation_name = 'hasty_format_crops_512_0.json'
    # herdnet_annotation_name = 'herdnet_format_512_0_crops.csv'
    # images_path = base_path / "crops_512_numNone_overlap0"
    # suffix = "jpg"


    radius = 150
    CONFIDENCE_THRESHOLD = 0.9
    dataset_name = "Genovesa"
    hard_negative_annotated_images = simple_hard_negative(base_path,
                                                          df_detections,
                                                          herdnet_annotation_name,
                                                          images_path,
                                                          suffix, radius,
                                                          CONFIDENCE_THRESHOLD, dataset_name=dataset_name)

    hA = HastyAnnotationV2.from_file(hA_reference)
    hA.get_image_by_id('712e7446-df34-4529-8727-820726fe3a1e')
    for ai in hard_negative_annotated_images:
        original_dataset_name, image_name = ai.image_name.split("___")

        uai = hA.get_image_by_name(dataset_name=dataset_name, image_name=image_name)
        uai.labels.extend(ai.labels)

        uai = hA.get_image_by_name(dataset_name=dataset_name, image_name=image_name)
        uai
    # merge the given hasty annotations with the generated hard negatives and save them


    hA.save("/raid/cwinkelmann/training_data/delplanque/general_dataset/hasty_style/delplanque_hasty.json")


