from loguru import logger
from pathlib import Path

from active_learning.config.dataset_filter import DatasetFilterConfig
from active_learning.types.filter import SampleStrategy
from com.biospheredata.types.status import LabelingStatus, AnnotationType

crop_size = 512
# labels_path = Path(f"/raid/cwinkelmann/training_data/iguana/2025_07_10_classic")
# hasty_annotations_labels_zipped = "2025_07_10_labels_final.zip"
# hasty_annotations_images_zipped = "2025_07_10_images_final.zip"


# # endgame case
# hasty_annotations_images_zipped = "2025_07_10_images_final.zip"
# hasty_annotations_labels_zipped = "full_label_correction_floreana_2025_07_10_train_correction_hasty_corrected_1.json.zip"
# labels_path = Path(f"/raid/cwinkelmann/training_data/iguana/2025_08_10_endgame")


# added CVAT corrected missions
hasty_annotations_images_zipped = "2025_09_19_images_final.zip"
hasty_annotations_labels_zipped = "2025_09_19_orthomosaic_data_combined_corrections_2.json.zip"
labels_path = Path(f"/home/cwinkelmann/work/Herdnet/data/2025_09_28_orthomosaic_data/")

# hasty_annotations_labels_zipped = "2025_07_10_labels_final.zip"
# hasty_annotations_labels_zipped = "fernandina_s_correction_hasty_corrected_1.json.zip"
# hasty_annotations_labels_zipped = "2025_07_10_labels_final.zip"


# labels_name = "/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/corrections/label_correction_floreana_2025_07_10_review_hasty_corrected.json"
# labels_name = "/raid/cwinkelmann/training_data/iguana/2025_08_10_label_correction/fernandina_s_correction_hasty_corrected_1.json"

annotation_types = [AnnotationType.KEYPOINT]
class_filter = ["iguana_point"]
label_mapping = {"iguana_point": 1, "iguana": 2}

# annotation_types = [AnnotationType.BOUNDING_BOX]
# class_filter = ["iguana"]


overlap = 0
VISUALISE_FLAG = False
empty_fraction = 0
use_multiprocessing = True
unpack = True
flatten = True

datasets_mapping = {
    "Floreana": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
                 'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO03', 'FMO04', 'FPA03 condor', 'FSCA02',
                 ],

    "Floreana_1": ['Floreana_02.02.21_FMO01', 'FMO02', 'FLMO02_28012023', 'FMO03',
                   'FMO04', 'FMO05', 'Floreana_03.02.21_FMO06'],

    "Floreana_3": ['FMO03', 'FMO04',],

    "Floreana_4": ['FPA03 condor', 'FSCA02'],

    "Floreana_2": ['Floreana_22.01.21_FPC07',
                   'FLBB01_28012023'],

    "Floreana_best": ['Floreana_03.02.21_FMO06',
                      'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO04', 'FMO03',
                      'FPA03 condor'],

    "Fernandina_s_1": ['Fer_FCD01-02-03_20122021_single_images', "floreana_FPE01_FECA01"],
    # yes that is Fernandina despite the name
    "Fernandina_s_2": [
        'FPM01_24012023',
        'Fer_FPE02_07052024'
    ],
    "Genovesa": ['Genovesa'],

    "Fernandina_CVAT_train_corr": ['ha_corrected_fer_fna01_02_20122021', 'ha_corrected_fer_fef01_02_20012023', 'ha_corrected_fer_fnd02_19122021'],
    "Fernandina_CVAT_val_corr": ['ha_corrected_fer_fni03_04_19122021', 'ha_corrected_fer_fnj01_19122021'],

    "Fernandina_CVAT_train_class": ['ha_fer_fna01_02_20122021', 'ha_fer_fef01_02_20012023', 'ha_fer_fnd02_19122021', ],
    "Fernandina_CVAT_val_class": ['ha_fer_fni03_04_19122021', 'ha_fer_fnj01_19122021'],

    "Fernandina_m": ['Fer_FCD01-02-03_20122021', 'Fer_FPM01-02_20122023'],
    "Fernandina_m_fcd": ['Fer_FCD01-02-03_20122021'],
    "Fernandina_m_fpm": ['Fer_FPM01-02_20122023'],

    "Fernanna_m_corr": ['ha_corrected_fer_fna01_02_20122021', 'ha_corrected_fer_fef01_02_20012023'],

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
    "dataset_filter": datasets_mapping["Floreana_1"],
    # "images_filter": ["FCD01-02-03_20122021_Fernandina_m_3_8.jpg"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "num": 3
})
train_genovesa = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Genovesa_detection",
    "dataset_filter": datasets_mapping["Genovesa"],
    "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,

})
val_genovesa = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Genovesa_detection",
    "dataset_filter": datasets_mapping["Genovesa"],  # Fer_FCD01-02-03_20122021_single_images
    "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})

## Floreana
train_floreana = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets_mapping["Floreana_1"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})
train_floreana_2 = DatasetFilterConfig(**{
    "dset": "train_2",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets_mapping["Floreana_2"] + datasets_mapping["Floreana_1"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})
train_floreana_2_overlap = DatasetFilterConfig(**{
    "dset": "train_2",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets_mapping["Floreana_2"] + datasets_mapping["Floreana_1"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": 350,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})
train_floreana_increasing_length = [DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": f"Floreana_detection_il_{x}",
    "dataset_filter": datasets_mapping["Floreana_1"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "num": x,
    "sample_strategy": SampleStrategy.ORDERED_ASC
}) for x in range(1, 125)]



val_floreana = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets_mapping["Floreana_3"] + datasets_mapping["Floreana_4"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})

val_1_floreana = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets_mapping["Floreana_3"],  # Fer_FCD01-02-03_20122021_single_images
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})

val_2_floreana = DatasetFilterConfig(**{
    "dset": "val_2",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets_mapping["Floreana_4"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})

test_floreana = DatasetFilterConfig(**{
    "dset": "test",
    "dataset_name": "Floreana_detection",
    "dataset_filter": datasets_mapping["Floreana_3"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})

## Fernandina Mosaic
train_fernandina_m = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_m_detection",
    "dataset_filter": datasets_mapping["Fernandina_m_fcd"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
val_fernandina_m = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Fernandina_m_detection",
    "dataset_filter": datasets_mapping["Fernandina_m_fpm"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
# Fernandina single images
train_fernandina_s1 = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_s_detection",
    "dataset_filter": datasets_mapping["Fernandina_s_1"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})
train_fernandina_s1_increasing_length = [DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": f"Fernandina_s_detection_il_{x}",
    "dataset_filter": datasets_mapping["Fernandina_s_1"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "num": x,

    "sample_strategy": SampleStrategy.ORDERED_ASC,
    "remove_Default": True,
    "remove_padding": True
}) for x in range(1, 39)]

val_fernandina_s2 = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Fernandina_s_detection",
    "dataset_filter": datasets_mapping["Fernandina_s_2"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "remove_Default": False
})
# All other datasets_mapping which are just out of Orthomosaics
train_rest = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Rest_detection",
    "dataset_filter": datasets_mapping["the_rest"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
# All single images from all datasets_mapping
train_single_all = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection_single",
    "dataset_filter": datasets_mapping["Floreana_1"] + datasets_mapping["Fernandina_s_1"] + datasets_mapping[
        "Genovesa"] + datasets_mapping["the_rest"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
val_single_all = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "All_detection_single",
    "dataset_filter": datasets_mapping["Floreana_2"] + datasets_mapping["Fernandina_s_2"] + datasets_mapping[
        "Genovesa"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
# All datasets_mapping combined
train_all = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection",
    "dataset_filter": datasets_mapping["the_rest"] + datasets_mapping["Floreana_1"]
                      + datasets_mapping["Fernandina_s_2"] + datasets_mapping["Fernandina_s_1"]
                      + datasets_mapping["Fernandina_m_fpm"] + datasets_mapping["Fernandina_m_fcd"]
                      + datasets_mapping["Genovesa"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

# the wining datasets_mapping combination
train_winning = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection_winning",
    "dataset_filter": datasets_mapping["the_rest"] + datasets_mapping["Floreana_1"]
                      + datasets_mapping["Floreana_2"]
                      + datasets_mapping["Fernandina_m_fpm"]
                      + datasets_mapping["Fernandina_s_2"]  # fpm and fpe
                      + datasets_mapping["Genovesa"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})

# the wining datasets_mapping combination
val_winning = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "All_detection_winning",
    "dataset_filter": datasets_mapping["Fernandina_s_1"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})

# the wining datasets_mapping combination
val_winning_fcd_m = DatasetFilterConfig(**{
    "dset": "val_fcd_m",
    "dataset_name": "All_detection_winning",
    "dataset_filter": datasets_mapping["Fernandina_m_fcd"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})

All = DatasetFilterConfig(**{
    "dset": "all",
    "dataset_name": "All_detection_winning",
    "dataset_filter": None,
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})



### CVAT corrected and not corrected datasets_mapping

#     Fernandina_CVAT_train_corr, Fernandina_CVAT_val_corr,
#     Fernandina_CVAT_train_class, Fernandina_CVAT_val_class,
Fernandina_CVAT_train_corr = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_CVAT_corr",
    "dataset_filter": datasets_mapping["Fernandina_CVAT_train_corr"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})
Fernandina_CVAT_val_corr = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Fernandina_CVAT_corr",
    "dataset_filter": datasets_mapping["Fernandina_CVAT_val_corr"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})

train_all_corr = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection",
    "dataset_filter": datasets_mapping["the_rest"] + datasets_mapping["Floreana_1"]
                      + datasets_mapping["Fernandina_s_2"] + datasets_mapping["Fernandina_s_1"]
                      + datasets_mapping["Fernandina_m_fpm"] + datasets_mapping["Fernandina_m_fcd"]
                      + datasets_mapping["Genovesa"] + datasets_mapping["Fernandina_CVAT_train_corr"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

Fernandina_CVAT_train_class = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_CVAT_class",
    "dataset_filter": datasets_mapping["Fernandina_CVAT_train_class"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})
Fernandina_CVAT_val_class = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Fernandina_CVAT_class",
    "dataset_filter": datasets_mapping["Fernandina_CVAT_val_class"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    # "edge_black_out": edge_black_out,
    # "num": num,
    # "crop": False,
    "remove_default_folder": False,
})

datasets = [
# All,
# train_floreana_sample,
# train_floreana,
# val_floreana,
# val_1_floreana,
# val_2_floreana,
# test_floreana,
# #
# train_fernandina_s1,
# val_fernandina_s2,
# 
# train_genovesa, val_genovesa,


    # # ,
    # 
    # , 
    # train_floreana_2_overlap,
    # val_floreana,
    # val_2_floreana,
    # test_floreana,
    # # train_winning, val_winning,
    # val_winning_fcd_m,
    # train_fernandina_m, val_fernandina_m,
    # train_fernandina_s1, val_fernandina_s2,
    # train_genovesa, val_genovesa,
    # # train_rest,
    
    # train_all_corr,

    # train_single_all,
    # val_single_all,

    # Fernandina_CVAT_train_corr, Fernandina_CVAT_val_corr,
    # Fernandina_CVAT_train_class, Fernandina_CVAT_val_class,

]
datasets += train_floreana_increasing_length
# datasets += train_fernandina_s1_increasing_length
