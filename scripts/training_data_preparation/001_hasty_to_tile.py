# """
# Create patches from images and labels from hasty to be used in CVAT/training
# """
# import shutil
# from loguru import logger
# from pathlib import Path
#
# from active_learning.config.dataset_filter import DatasetFilterConfig
# from active_learning.filter import ImageFilterConstantNum
# from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
# from com.biospheredata.converter.HastyConverter import AnnotationType
# from com.biospheredata.converter.HastyConverter import HastyConverter
#
# if __name__ == "__main__":
#     """ This only works if the input is a hasty zip file which is very constraining. """
#
#     # labels_path = Path("/Users/christian/data/training_data/2025_01_11")
#     # hasty_annotations_labels_zipped = "labels_segments.zip"
#     # hasty_annotations_images_zipped = "images_segments.zip"
#     # annotation_types = [AnnotationType.POLYGON]
#     #
#     # labels_path = Path("/Users/christian/data/training_data/2024_12_09")
#     # hasty_annotations_labels_zipped = "FMO02_03_05_labels.zip"
#     # hasty_annotations_images_zipped = "FMO02_03_05_images.zip"
#     # annotation_types = [AnnotationType.BOUNDING_BOX]
#
#     # class_filter = ["iguana"]
#
#
#     ## Meeting presentation
#     labels_path = Path("/Users/christian/data/training_data/2024_12_09_debug")
#     hasty_annotations_labels_zipped = "FMO02_03_05_labels.zip"
#     hasty_annotations_images_zipped = "FMO02_03_05_images.zip"
#     annotation_types = [AnnotationType.KEYPOINT]
#     class_filter = ["iguana_point", "iguana"]
#
#     ## Segmentation masks
#     labels_path = Path("/Users/christian/data/training_data/2025_02_03_segments")
#     hasty_annotations_labels_zipped = "labels_segments.zip"
#     hasty_annotations_images_zipped = "images_segments.zip"
#     annotation_types = [AnnotationType.POLYGON]
#
#     class_filter = ["iguana"]
#
#     crop_size = 224
#     # overlap = 0
#     # amount of empty images in the dataset
#
#
#     # train_fmo05 = DatasetFilterConfig(**{
#     #     "dset": "train",
#     #     # "images_filter": ["DJI_0935.JPG", "DJI_0972.JPG", "DJI_0863.JPG"],
#     #     # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
#     #     "dataset_filter": ["FMO05"],
#     #     # "dataset_filter": None,
#     #     # "num": 1,
#     #     "output_path": labels_path,
#     #
#     #     "empty_fraction": 0.0,
#     #
#     # })
#
#     # train_floreana = DatasetFilterConfig(**{
#     #     "dset": "train",
#     #     "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
#     #     # "num": 1,
#     #     "output_path": labels_path,
#     #     "empty_fraction": 0.0
#     # })
#     #
#     # val_fmo03 = DatasetFilterConfig(**{
#     #         "dset": "val",
#     #         # "images_filter": ["DJI_0465.JPG"],
#     #         "dataset_filter": ["FMO03"],
#     #         # "dataset_filter": None,
#     #         "output_path": labels_path,
#     #         "empty_fraction": 0.0
#     #
#     #     })
#     #
#     # fernandina_ds = DatasetFilterConfig(**{
#     #         "dset": "test",
#     #         # "images_filter": ["DJI_0465.JPG"],
#     #         "dataset_filter": ["Fer_FCD01-02-03_20122021", "Fer_FCD01-02-03_20122021_single_images"],
#     #         # "dataset_filter": None,
#     #         "output_path": labels_path,
#     #         "empty_fraction": 0.0
#     #
#     #     })
#     #
#     # test_fmo02 = DatasetFilterConfig(**{
#     #         "dset": "test",
#     #         # "images_filter": ["DJI_0554.JPG"],
#     #         "dataset_filter": ["FMO02"],
#     #         "output_path": labels_path,
#     #         "empty_fraction": 0.0
#     #
#     #     })
#
#
#     # datasets = [
#     #     {
#     #         "dset": "train",
#     #         # "images_filter": ["DJI_0935.JPG", "DJI_0972.JPG", "DJI_0863.JPG"],
#     #         # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
#     #         "dataset_filter": ["FMO05"],
#     #         # "dataset_filter": None,
#     #         "num": 1,
#     #         "output_path": labels_path,
#     #         "empty_fraction": 0.0
#     #     },
#     #     {
#     #         "dset": "val",
#     #         # "images_filter": ["DJI_0465.JPG"],
#     #         "dataset_filter": ["FMO03"],
#     #         # "dataset_filter": None,
#     #         "output_path": labels_path,
#     #         "empty_fraction": 0.0
#     #
#     #     },
#     #     {
#     #         "dset": "test",
#     #         # "images_filter": ["DJI_0554.JPG"],
#     #         "dataset_filter": ["FMO02"],
#     #         "output_path": labels_path,
#     #         "empty_fraction": 0.0
#     #
#     #     }
#     # ]
#
#     # ## Data preparation based on segmentation masks
#     # train_segments = DatasetFilterConfig(**{
#     #     "dset": "train",
#     #     "images_filter": ["STJB06_12012023_Santiago_m_2_7_DJI_0128.JPG", "DJI_0079_FCD01.JPG", "DJI_0924.JPG", "DJI_0942.JPG",
#     #                       "DJI_0097.JPG", "SRPB06 1053 - 1112 falcon_dem_translate_0_0.jpg", "DJI_0097.JPG", "DJI_0185.JPG",
#     #                       "DJI_0195.JPG", "DJI_0237.JPG", "DJI_0285.JPG", "DJI_0220.JPG",
#     #                       ],
#     #     "output_path": labels_path,
#     #     "empty_fraction": 1.0,
#     #     "image_tags": ["segment"]
#     # })
#     #
#     # val_segments = DatasetFilterConfig(**{
#     #     "dset": "val",
#     #     "images_filter": ["DJI_0395.JPG", "DJI_0009.JPG", "DJI_0893.JPG", "DJI_0417.JPG"],
#     #     "output_path": labels_path,
#     #     "empty_fraction": 1.0,
#     #     "image_tags": ["segment"]
#     # })
#
#     ## Data preparation based on segmentation masks
#     train_segments = DatasetFilterConfig(**{
#         "dset": "train",
#         "images_filter": ["STJB06_12012023_Santiago_m_2_7_DJI_0128.JPG", "DJI_0079_FCD01.JPG", "DJI_0924.JPG", "DJI_0942.JPG",
#                           "DJI_0097.JPG", "SRPB06 1053 - 1112 falcon_dem_translate_0_0.jpg", "DJI_0097.JPG", "DJI_0185.JPG",
#                           "DJI_0195.JPG", "DJI_0237.JPG", "DJI_0285.JPG", "DJI_0220.JPG",
#                           ],
#         "output_path": labels_path,
#         "empty_fraction": 1.0,
#         "image_tags": ["segment"],
#         "num": 2,
#         "overlap": 0.5
#     })
#
#     val_segments = DatasetFilterConfig(**{
#         "dset": "val",
#         "images_filter": ["DJI_0395.JPG", "DJI_0009.JPG", "DJI_0893.JPG", "DJI_0417.JPG"],
#         "output_path": labels_path,
#         "empty_fraction": 1.0,
#         "image_tags": ["segment"]
#     })
#
#
#     # datasets = [train_fmo05, val_fmo03, test_fmo02]
#     # datasets = [train_floreana, val_fmo03, fernandina_ds]
#     datasets = [train_segments, val_segments]
#     report = {}
#     # datasets = [{
#     #     "dset": "train",
#     #     # "images_filter": ["DJI_0432.JPG"],
#     #     # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"],
#     #     "dataset_filter": ["FMO05"],
#     #     "num": n
#     # } for n in range(11, 12)]
#     #
#     # datasets.append( {"dset": "val",
#     #     # "images_filter": ["DJI_0465.JPG"],
#     #      "dataset_filter": ["FMO03"],
#     #      })
#     # datasets.append(
#     #     {"dset": "test",
#     #      # "images_filter": ["DJI_0554.JPG"],
#     #      "dataset_filter": ["FMO02"]
#     #      })
#
#     for dataset in datasets:  # , "val", "test"]:
#         logger.info(f"Starting {dataset.dset}")
#         dset = dataset.dset
#         num = dataset.num
#         ifcn = ImageFilterConstantNum(num=num, dataset_config = dataset)
#         # output_path = dataset["output_path"]
#
#         uA = UnpackAnnotations()
#         hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped=labels_path / hasty_annotations_labels_zipped,
#                        hasty_annotations_images_zipped = labels_path / hasty_annotations_images_zipped)
#         logger.info(f"Unzipped {len(hA.images)} images.")
#         output_path_dset = labels_path / dset
#         output_path_dset.mkdir(exist_ok=True)
#
#         # TODO flatten the images here
#         hA_flat = hA.get_flat_df()
#         logger.info(f"Flattened annotations {hA_flat} annotations.")
#
#         dp = DataprepPipeline(annotations_labels=hA,
#                               images_path=images_path,
#                               crop_size=crop_size,
#                               overlap=overlap,
#                               output_path=output_path_dset,
#                               )
#
#         dp.dataset_filter = dataset.dataset_filter
#
#         dp.images_filter = dataset.images_filter
#         dp.images_filter_func = [ifcn]
#         dp.class_filter = class_filter
#         dp.annotation_types = annotation_types
#         dp.empty_fraction = dataset.empty_fraction
#
#         # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
#         dp.run(flatten=True)
#
#         hA_filtered = dp.get_hA_filtered()
#         # full size annotations
#         HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dset / "herdnet_format.csv")
#
#         hA_crops = dp.get_hA_crops()
#         aI = AnnotationsIntermediary()
#         logger.info(f"After processing {len(hA_crops.images)} images remain")
#         if len(hA_crops.images) == 0:
#             raise ValueError("No images left after filtering")
#
#         # if visualise_image:
#         #     # TODO visualise these images
#         #     visualise_image(hA_crops, images_path, output_path_dset / "visualisation")
#
#         aI.set_hasty_annotations(hA=hA_crops)
#         coco_path = aI.coco(output_path_dset / "coco_format.json")
#         images_list = dp.get_images()
#
#
#         logger.info(f"Finished {dset} at {output_path_dset}")
#         # TODO before uploading anything to CVAT labels need to be converted when necessary
#         hA_crops.save(output_path_dset / "hasty_format_crops.json")
#
#         # TODO check if the conversion from polygon to point is correct
#         HastyConverter.convert_to_herdnet_format(hA_crops, output_file=output_path_dset / "herdnet_format_crops.csv")
#
#         if AnnotationType.BOUNDING_BOX in annotation_types or AnnotationType.POLYGON in annotation_types:
#             HastyConverter.convert_deep_forest(hA_crops, output_file=output_path_dset / "deep_forest_format_crops.csv")
#
#             class_names = aI.to_YOLO_annotations(output_path=output_path_dset / "yolo")
#             report[f"yolo_box_path_{dset}"] = output_path_dset / "yolo" / "yolo_boxes"
#             report[f"yolo_segments_path_{dset}"] = output_path_dset / "yolo" / "yolo_segments"
#             report[f"class_names"] = class_names
#
#         # TODO move the crops to a new folder for YOLO
#         # TODO move the crops to a new folder for classification
#
#         stats = dp.get_stats()
#         logger.info(f"Stats {dset}: {stats}")
#         destination_path = output_path_dset / f"crops_{crop_size}_num{num}_overlap{overlap}"
#
#         try:
#             shutil.rmtree(destination_path)
#             logger.warning(f"Removed {destination_path}")
#         except FileNotFoundError:
#             pass
#         shutil.move(output_path_dset / f"crops_{crop_size}", destination_path)
#
#         logger.info(f"Moved to {destination_path}")
#
#         report[f"destination_path_{dset}"] = destination_path
#
#
#
#     # TODO This YOLO data.yaml sucks
#     HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
#                                                   images_train_path=report["destination_path_train"],
#                                                   images_val_path=report["destination_path_val"],
#                                                   labels_train_path=report["yolo_box_path_train"],
#                                                     labels_val_path=report["yolo_box_path_val"], class_names=report["class_names"],
#                                                   data_yaml_path=labels_path / "data_boxes.yaml")
#
#     HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
#                                                   images_train_path=report["destination_path_train"],
#                                                   images_val_path=report["destination_path_val"],
#                                                   labels_train_path=report["yolo_box_path_train"],
#                                                     labels_val_path=report["yolo_box_path_val"], class_names=report["class_names"],
#                                                   data_yaml_path=labels_path / "data_segments.yaml")
