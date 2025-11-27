from pathlib import Path

from dataset_configs_hasty_point_iguanas import datasets_mapping
from active_learning.config.dataset_filter import DatasetFilterConfig
from com.biospheredata.types.status import LabelingStatus, AnnotationType

hasty_annotations_images_zipped = "2025_10_11_images.zip"
hasty_annotations_labels_zipped = "2025_10_11_labels.json.zip"
labels_path = Path(f"/raid/cwinkelmann/training_data/iguana/2025_10_11")

# labels_name = "/Users/christian/data/training_data/2025_07_10_final_point_detection/Floreana_detection/val/corrections/label_correction_floreana_2025_07_10_review_hasty_corrected.json"
# images_path = base_path / "unzipped_images"

annotation_types = [AnnotationType.BOUNDING_BOX]
class_filter = ["iguana"]

crop_size = 512
empty_fraction = 0.0
overlap = 0
VISUALISE_FLAG = False
use_multiprocessing = False
edge_black_out = False

unpack = True


# datasets = {
#     "Floreana": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
#                  'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO03', 'FMO04', 'FPA03 condor', 'FSCA02',
#                  "floreana_FPE01_FECA01"],
# 
#     "Floreana_1": ['FMO03', 'FMO04', 'FPA03 condor', 'FSCA02', "floreana_FPE01_FECA01"],
#     "Floreana_2": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
#                    'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05'],
# 
#     "Floreana_best": ['Floreana_03.02.21_FMO06', "floreana_FPE01_FECA01"
#                                                  'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO04', 'FMO03',
#                       'FPA03 condor'],
# 
#     "Fernandina_s_1": ['Fer_FCD01-02-03_20122021_single_images'],
#     "Fernandina_s_2": [
#         'FPM01_24012023',
#         'Fer_FPE02_07052024'
#     ],
#     "Genovesa": ['Genovesa'],
# 
#     "Fernandina_m": ['Fer_FCD01-02-03_20122021', 'Fer_FPM01-02_20122023'],
#     "Fernandina_m_fcd": ['Fer_FCD01-02-03_20122021'],
#     "Fernandina_m_fpm": ['Fer_FPM01-02_20122023'],
# 
#     "the_rest": [
#         # "SRPB06 1053 - 1112 falcon_25.01.20", # orthomosaics contains nearly iguanas but not annotated
#         "SCris_SRIL01_04022023",  # Orthomosaic
#         "SCris_SRIL02_04022023",  # Orthomosaic
#         "SCris_SRIL04_04022023",  # Orthomosaic, 4 iguanas
# 
#         "San_STJB01_12012023",  # Orthomosaic, 13
#         "San_STJB02_12012023",  # Orthomosaic
#         "San_STJB03_12012023",  # Orthomosaic
#         "San_STJB04_12012023",  # Orthomosaic
#         "San_STJB06_12012023",  # Orthomosaic
# 
#         "SCruz_SCM01_06012023"  # Orthomosaic
#     ],
# 
#     "zooniverse_phase_2": ["Zooniverse_expert_phase_2"],
#     "zooniverse_phase_3": ["Zooniverse_expert_phase_3"]
# }


train_floreana_sample = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "floreana_sample",
    "dataset_filter": datasets_mapping["Floreana_best"],
    "images_filter": ["DJI_0514.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    # "num": 10
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter, 
    "crop_size": crop_size,
})

train_floreana = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Floreana_classification",
    "dataset_filter": datasets_mapping["Floreana_1"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
val_floreana = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Floreana_classification",
    "dataset_filter": datasets_mapping["Floreana_2"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
train_floreana_increasing_length = [DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": f"Floreana_classification_il_{x}",
    "dataset_filter": datasets_mapping["Floreana_1"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
    "num": x
}) for x in range(1, 36)]
## Fernandina Mosaic
train_fernandina_m = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_m_classification",
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
    "dataset_name": "Fernandina_m_classification",
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
# classification Fernandina single images
train_fernandina_s1 = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Fernandina_s_classification",
    "dataset_filter": datasets_mapping["Fernandina_s_1"],
    # "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
val_fernandina_s2 = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "Fernandina_s_classification",
    "dataset_filter": datasets_mapping["Fernandina_s_2"],  # Fer_FCD01-02-03_20122021_single_images
    # "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

train_genovesa = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Genovesa_classification",
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
    "dataset_name": "Genovesa_classification",
    "dataset_filter": datasets_mapping["Genovesa"],  # Fer_FCD01-02-03_20122021_single_images
    "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})


# All other datasets which are just out of Orthomosaics
train_rest = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "Rest_m_detection",
    "dataset_filter": datasets_mapping["the_rest"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})
# All single images from all datasets BUT Floreana 2 to see if they are useful
train_single_all = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection_single_butfloreana_2",
    "dataset_filter": datasets_mapping["Floreana_1"] + datasets_mapping["Fernandina_s_2"] + datasets_mapping["Fernandina_s_1"] + datasets_mapping[
        "Genovesa"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

# All datasets combined
train_all = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "All_detection",
    "dataset_filter": datasets_mapping["the_rest"] + datasets_mapping["Floreana_1"] + datasets_mapping["Fernandina_s_2"] + datasets_mapping[
        "Fernandina_s_1"] + datasets_mapping["Fernandina_m_fpm"] + datasets_mapping["Fernandina_m_fcd"] + datasets_mapping["Genovesa"],
    "output_path": labels_path,
    "empty_fraction": empty_fraction,
    "overlap": overlap,
    "status_filter": [LabelingStatus.COMPLETED],
    "annotation_types": annotation_types,
    "class_filter": class_filter,
    "crop_size": crop_size,
})

datasets = [
    # train_floreana_sample,
    # train_floreana, val_floreana,
    train_fernandina_m, val_fernandina_m,
    train_fernandina_s1, val_fernandina_s2,
    train_genovesa, val_genovesa,
    # train_rest,
    # train_all,
    #
    # train_single_all
]
# datasets += train_floreana_increasing_length
