import gc
import json
import shutil
import yaml
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path

from active_learning.config.dataset_filter import DatasetFilterConfig, DataPrepReport
from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from active_learning.util.visualisation.annotation_vis import visualise_points_only, create_simple_histograms, \
    visualise_hasty_annotation_statistics, plot_bbox_sizes
from com.biospheredata.converter.HastyConverter import AnnotationType, LabelingStatus
from com.biospheredata.converter.HastyConverter import HastyConverter
from image_template_search.util.util import (visualise_image, visualise_polygons)

labels_path = Path(f"/Users/christian/data/training_data/2025_08_10_endgame")
hasty_annotations_labels_zipped = "full_label_correction_floreana_2025_07_10_train_correction_hasty_corrected_1.json.zip"
# hasty_annotations_labels_zipped = "label_correction_floreana_2025_07_10_review_hasty_corrected.json.zip"
hasty_annotations_images_zipped = "2025_07_10_images_final.zip"

# labels_name = "/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/corrections/label_correction_floreana_2025_07_10_review_hasty_corrected.json"

annotation_types = [AnnotationType.BOUNDING_BOX, AnnotationType.KEYPOINT]
class_filter = ["iguana", "iguana_point"]

# annotation_types = [AnnotationType.KEYPOINT]
# class_filter = ["iguana_point"]

label_mapping = {"iguana_point": 1, "iguana": 2}

crop_size = 512
overlap = 0
VISUALISE_FLAG = False
empty_fraction = 0
multiprocessing = False  # Fixme later, currently not working with multiprocessing
edge_black_out = True
crop = False
num = None

datasets = {
    "Floreana": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
                 'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO03', 'FMO04', 'FPA03 condor', 'FSCA02',
                 "floreana_FPE01_FECA01"],

    "Floreana_1": ['FMO03', 'FMO04', 'FPA03 condor', 'FSCA02', "floreana_FPE01_FECA01"],
    "Floreana_2": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
                   'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05'],

    "Floreana_best": ['Floreana_03.02.21_FMO06', "floreana_FPE01_FECA01"
                                                 'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO04', 'FMO03',
                      'FPA03 condor'],

    "Fernandina_s_1": ['Fer_FCD01-02-03_20122021_single_images'],
    "Fernandina_s_2": [
        'FPM01_24012023',
        'Fer_FPE02_07052024'
    ],
    "Genovesa": ['Genovesa'],

    "Fernandina_m": ['Fer_FCD01-02-03_20122021', 'Fer_FPM01-02_20122023'],
    "Fernandina_m_fcd": ['Fer_FCD01-02-03_20122021'],
    "Fernandina_m_fpm": ['Fer_FPM01-02_20122023'],

    "the_rest": [
        # "SRPB06 1053 - 1112 falcon_25.01.20", # orthomosaics contains nearly iguanas but not annotated
        "SCris_SRIL01_04022023",  # Orthomosaic
        "SCris_SRIL02_04022023",  # Orthomosaic
        "SCris_SRIL04_04022023",  # Orthomosaic, 4 iguanas

        "San_STJB01_12012023",  # Orthomosaic, 13
        "San_STJB02_12012023",  # Orthomosaic
        "San_STJB03_12012023",  # Orthomosaic
        "San_STJB04_12012023",  # Orthomosaic
        "San_STJB06_12012023",  # Orthomosaic

        "SCruz_SCM01_06012023"  # Orthomosaic
    ],

    "zooniverse_phase_2": ["Zooniverse_expert_phase_2"],
    "zooniverse_phase_3": ["Zooniverse_expert_phase_3"]
}

## Data preparation for a debugging sample
train_floreana_sample = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "floreana_sample",
    "dataset_filter": datasets["Floreana"],
    "images_filter": ["DJI_0906.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
})
train_genovesa = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Genovesa_detection",
    "dataset_filter": datasets["Genovesa"],
    "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop
})
val_genovesa = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Genovesa_detection",
    "dataset_filter": datasets["Genovesa"],  # Fer_FCD01-02-03_20122021_single_images
    "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop
})
train_floreana = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets["Floreana_1"],
    # "images_filter": [
    #     # "DJI_0064_FECA01.JPG",
    #     "DJI_0210_FPE01.JPG",
    #     # "DJI_0485_FPE01.JPG"
    # ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "remove_default_folder": False,
    "crop": False
})

floreana_all = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Floreana_detection_all",
    "dataset_filter": datasets["Floreana_1"] + datasets["Floreana_2"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "remove_default_folder": False,
    "crop": False
})

train_floreana_increasing_length = [DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": f"Floreana_detection_il_{x}",
    "dataset_filter": datasets["Floreana_1"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": x
}) for x in range(1, 36)]

train_fernandina_s1_increasing_length = [DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": f"Fernandina_s_detection_il_{x}",
    "dataset_filter": datasets["Fernandina_s_1"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": x
}) for x in range(1, 25)]

val_floreana = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets["Floreana_2"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "remove_default_folder": False,
    "crop": False
})
## Fernandina Mosaic
train_fernandina_m = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_m_detection",
    "dataset_filter": datasets["Fernandina_m_fcd"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
    "remove_default_folder": False,
})
val_fernandina_m = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Fernandina_m_detection",
    "dataset_filter": datasets["Fernandina_m_fpm"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
    "remove_default_folder": False,
})
# Fernandina single images
train_fernandina_s1 = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_s_detection",
    "dataset_filter": datasets["Fernandina_s_1"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
    "remove_default_folder": False,
})

train_fernandina_s1_floreana = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Floreana_Fernandina_s_detection",
    "dataset_filter": datasets["Fernandina_s_1"] + datasets["Floreana_1"] + datasets["Floreana_2"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
    "remove_default_folder": False,
})
val_fernandina_s2 = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Fernandina_s_detection",
    "dataset_filter": datasets["Fernandina_s_2"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
    "remove_default_folder": False,
})
# All other datasets which are just out of Orthomosaics
train_rest = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Rest_detection",
    "dataset_filter": datasets["the_rest"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
"remove_default_folder": False,
})
# All single images from all datasets
train_single_all = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection_single",
    "dataset_filter": datasets["Floreana_1"] + datasets["Fernandina_s_1"] + datasets["Genovesa"] + datasets["the_rest"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop
})
val_single_all = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "All_detection_single",
    "dataset_filter": datasets["Floreana_2"] + datasets["Fernandina_s_2"] + datasets["Genovesa"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop
})
# All datasets combined
train_all = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection_2",
    "dataset_filter": datasets["the_rest"] + datasets["Floreana_1"] + datasets["Floreana_2"] + datasets["Fernandina_m_fpm"] + datasets["Fernandina_m_fcd"] + datasets["Genovesa"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "edge_black_out": edge_black_out,
    "num": num,
    "crop": crop,
"remove_default_folder": False,
})

datasets = [
    # train_floreana_sample,
    # train_floreana,
    # val_floreana,
    # floreana_all,
    # train_fernandina_s1, val_fernandina_s2,
    # train_fernandina_s1_floreana,
    # train_fernandina_m, val_fernandina_m,
    # train_genovesa, val_genovesa,
    # train_rest,
    # train_all,
    # train_single_all,
    # val_single_all
    train_all
]
# datasets += train_floreana_increasing_length
# datasets += train_fernandina_s1_increasing_length
