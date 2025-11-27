"""
read the annotations from the hasty json and transform it

"""
from pathlib import Path

from loguru import logger
import json

import yaml

from com.biospheredata.converter.HastyToYoloConverter import HastyToYOLOConverter
from com.biospheredata.converter.YoloTiler import YoloTiler



# def prepare_output_folder_str(base_path: Path, yolo_base_path: Path):
#     """
#
#     :param base_path:
#     :return:
#     """
#     # base_path.joinpath("test/images").mkdir(exist_ok=True, parents=True)
#     # base_path.joinpath("test/labels").mkdir(exist_ok=True, parents=True)
#     base_path.joinpath("train/images").mkdir(exist_ok=True, parents=True)
#     base_path.joinpath("train/labels").mkdir(exist_ok=True, parents=True)
#     base_path.joinpath("valid/images").mkdir(exist_ok=True, parents=True)
#     base_path.joinpath("valid/labels").mkdir(exist_ok=True, parents=True)
#
#     with open(yolo_base_path.joinpath("class_names.txt"), 'r+') as f:
#         class_names = json.load(f)
#
#     data_yaml = {
#         "train": "./train/images",
#         "val": "./valid/images",
#
#         "nc": len(class_names),
#         "names": class_names
#     }
#     with open(base_path.joinpath("data.yaml"), 'w+') as outfile:
#         yaml.dump(data_yaml, outfile, default_flow_style=False)
#
#
# def whole_workflow(base_path: Path,
#                    hasty_annotations_labels_zipped,
#                    hasty_annotations_images_zipped,
#                    max_training_images=(5000000,),
#                    backgrounds=(0,),
#                    fulls=False,
#                    folds=5,
#                    slice_size=632,
#                    prefix="rtu_tds_hnee"
#                    ):
#     """
#     with downloaded annotations prepare them to training data for yolo
#
#     @type base_path: Path
#     @param base_path
#     @param hasty_annotations_labels_zipped:
#     @param hasty_annotations_images_zipped:
#     @return:
#     :param prefix:
#     :param slice_size:
#     :param folds:
#     :param fulls:
#     :param backgrounds:
#     :param max_training_images:
#     """
#
#     tiled_path = base_path.joinpath("tiled")
#
#     hYC = HastyToYOLOConverter(base_path=base_path,
#                                hasty_annotations_labels_zipped=hasty_annotations_labels_zipped,
#                                hasty_annotations_images_zipped=hasty_annotations_images_zipped
#                                )
#     if fulls:
#         hYC.prepare_zip_files()
#         hYC.merge_image_datasets()
#     hYC.transform_annotation()
#
#     if fulls:
#         results = list(YoloTiler.tile_folders(images_path=hYC.yolo_image_path,
#                                               labels_path=hYC.yolo_label_path,
#                                               tiled_path=tiled_path,
#                                               slice_size=slice_size))
#         results = [r for r in results]
#
#     for background in backgrounds:
#         for n in max_training_images:
#             DATASET_NAME = f"{prefix}_{n}_{background}"
#             splitted_tiled_path = base_path.joinpath(DATASET_NAME)
#
#             if background:
#                 YoloTiler.splitter2(source=tiled_path.joinpath(YoloTiler.IMAGES_WITHOUT_OBJECTS),
#                                     target=splitted_tiled_path,
#                                     consistent_ids=hYC.consistent_ids,
#                                     ext="JPG",
#                                     max_train_images=background,
#                                     folds=folds)
#
#             k_folds = YoloTiler.splitter2(source=tiled_path.joinpath(YoloTiler.IMAGES_WITH_OBJECTS),
#                                           target=splitted_tiled_path,
#                                           consistent_ids=hYC.consistent_ids,
#                                           ext="JPG",
#                                           max_train_images=n,
#                                           folds=folds)
#
#             logger.info(f"Everything is ready. The Data can be found at {splitted_tiled_path}")
#             i = 1
#             for fold in k_folds:
#                 yield {"ds_sz": n, "bg": background, "ds": DATASET_NAME, "fold": i}
#                 i += 1


if __name__ == '__main__':
    base_path = Path("/home/christian/data/iguanas_2022_11_02")
    hasty_annotations_labels_zipped = "labels.zip"
    hasty_annotations_images_zipped = "images.zip"

    whole_workflow(base_path,
                   hasty_annotations_labels_zipped,
                   hasty_annotations_images_zipped,
                   max_training_images=[10, 20, 30, 40, 50, 75, 100, 150, 200],
                   # max_training_images=[250],
                   backgrounds=[0]
                   )

