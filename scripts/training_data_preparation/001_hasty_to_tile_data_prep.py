# """
# Create patches from images and labels from hasty which can be used for training a model
#
# """
# import shutil
# from loguru import logger
# from matplotlib import pyplot as plt
# from pathlib import Path
#
# from active_learning.filter import ImageFilterConstantNum
# from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
# from active_learning.util.visualisation.annotation_vis import visualise_points_only
# from data_configs import val_fmo05, \
#     train_big
# from com.biospheredata.converter.HastyConverter import AnnotationType
# from com.biospheredata.converter.HastyConverter import HastyConverter
# from image_template_search.util.util import visualise_image
#
# if __name__ == "__main__":
#     visualise_crops = False
#
#     # ## fixed test data
#     labels_path = Path("/Users/christian/data/training_data/2025_06_08/all_completed")
#     hasty_annotations_labels_zipped = "labels_all_completed.zip"
#     hasty_annotations_images_zipped = "all_images.zip"
#
#     datasets = [train_big, val_fmo05]
#
#     annotation_types = [AnnotationType.KEYPOINT]
#     class_filter = ["iguana_point"]
#
#     # labels_path = Path("/Users/christian/data/training_data/2025_06_08/all_completed")
#     # hasty_annotations_labels_zipped = "labels_all_completed.zip"
#     # hasty_annotations_images_zipped = "image_all_completed.zip"
#     # annotation_types = [AnnotationType.BOUNDING_BOX]
#     # class_filter = ["iguana"]
#     # datasets = [train_floreana_big, val_fmo05]
#
#     # ## Segmentation masks
#     # labels_path = Path("/Users/christian/data/training_data/2025_02_22_HIT/01_segment_pretraining")
#     # hasty_annotations_labels_zipped = "labels_segments_completed.zip"
#     # hasty_annotations_images_zipped = "images_segments_completed.zip"
#     # annotation_types = [AnnotationType.POLYGON]
#     # class_filter = ["iguana"]
#     # datasets = [train_segments_fernanandina_12, val_segments_fernandina_1, test_segments_fernandina_1]
#     # datasets = [val_segments_fernandina_1]
#     overlap = 320
#     # amount of empty images in the dataset
#
#
#     report = {}
#
#     for dataset in datasets:
#         logger.info(f"Starting {dataset.dataset_name}")
#         dataset_name = dataset.dataset_name
#         dset = dataset.dset
#         crop_size = dataset.crop_size
#         num = dataset.num
#
#         ifcn = ImageFilterConstantNum(num=num, dataset_config=dataset)
#
#         uA = UnpackAnnotations()
#         hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped=labels_path / hasty_annotations_labels_zipped,
#                        hasty_annotations_images_zipped = labels_path / hasty_annotations_images_zipped)
#         logger.info(f"Unzipped {len(hA.images)} images.")
#         output_path_dataset_name = labels_path / dataset_name / dset
#         output_path_dataset_name.mkdir(exist_ok=True, parents=True)
#
#         dp = DataprepPipeline(annotations_labels=hA,
#                               images_path=images_path,
#                               crop_size=crop_size,
#                               overlap=overlap,
#                               output_path=output_path_dataset_name,
#                               )
#
#         dp.dataset_filter = dataset.dataset_filter
#         dp.images_filter = dataset.images_filter
#
#
#         dp.images_filter_func.append(ifcn) # TODO fix this
#         dp.class_filter = class_filter
#         dp.annotation_types = annotation_types
#         dp.empty_fraction = dataset.empty_fraction
#         dp.tag_filter = dataset.image_tags
#
#
#         # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
#         dp.run(flatten=True)
#
#         # get just the filtered annotations
#         hA_filtered = dp.get_hA_filtered()
#         hA_filtered.save(output_path_dataset_name / "hasty_format.json")
#         aI = AnnotationsIntermediary()
#         aI.set_hasty_annotations(hA=hA_filtered)
#         coco_path = aI.coco(output_path_dataset_name / "coco_format.json")
#         HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dataset_name / "herdnet_format.csv")
#
#         # get the filtered and croped
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
#         coco_path = aI.coco(output_path_dataset_name / "coco_format_crops.json")
#         images_list = dp.get_images()
#
#
#         logger.info(f"Finished {dataset_name} at {output_path_dataset_name}")
#         hA_crops.save(output_path_dataset_name / "hasty_format_crops.json")
#
#         # TODO check if the conversion from polygon to point is correct
#         HastyConverter.convert_to_herdnet_format(hA_crops, output_file=output_path_dataset_name / "herdnet_format_crops.csv")
#
#         HastyConverter.convert_to_classification(source_path=output_path_dataset_name / f"crops_{crop_size}",
#                                                  hA_crops=hA_crops, output_file=output_path_dataset_name / "classification")
#
#         if AnnotationType.BOUNDING_BOX in annotation_types or AnnotationType.POLYGON in annotation_types:
#             HastyConverter.convert_deep_forest(hA_crops, output_file=output_path_dataset_name / "deep_forest_format_crops.csv")
#             HastyConverter.convert_to_herdnet_box_format(hA_crops, output_file=output_path_dataset_name / "herdnet_boxes_format_crops.csv")
#             logger.info(f"Wrote herdnet_box {output_path_dataset_name / 'herdnet_boxes_format_crops.csv'}, dataset_nmae {dataset_name} at {output_path_dataset_name}")
#             class_names = aI.to_YOLO_annotations(output_path=output_path_dataset_name / "yolo")
#             report[f"yolo_box_path_{dset}"] = output_path_dataset_name / "yolo" / "yolo_boxes"
#             report[f"yolo_segments_path_{dset}"] = output_path_dataset_name / "yolo" / "yolo_segments"
#             report[f"class_names"] = class_names
#
#
#         stats = dp.get_stats()
#         logger.info(f"Stats {dataset_name}: {stats}")
#         destination_path = output_path_dataset_name / f"crops_{crop_size}_num{num}_overlap{overlap}"
#
#         try:
#             shutil.rmtree(destination_path)
#             logger.warning(f"Removed {destination_path}")
#         except FileNotFoundError:
#             pass
#         shutil.move(output_path_dataset_name / f"crops_{crop_size}", destination_path)
#
#         logger.info(f"Moved to {destination_path}")
#
#         report[f"destination_path_{dset}"] = destination_path
#
#         if visualise_crops:
#             vis_path = output_path_dataset_name / f"visualisations"
#             vis_path.mkdir(exist_ok=True, parents=True)
#             for image in hA_crops.images:
#                 logger.info(f"Visualising {image.image_name}")
#                 assert destination_path.joinpath(image.image_name).exists(), f"{destination_path.joinpath(image.image_name)} does not exist"
#
#                 ax_s = visualise_image(image_path = destination_path / image.image_name, show=False, title=f"Visualisation of {len([p.polygon_s for p in image.labels])} labels in {image.image_name}")
#
#                 filename = vis_path / f"{image.image_name}.png"
#                 # TODO check which type of visualisation is needed
#                 # visualise_polygons(polygons=[p.polygon_s for p in image.labels],
#                 #                    labels=[p.class_name for p in image.labels],  ax=ax_s, show=False, linewidth=2,
#                 #                    filename=filename)
#
#                 # # Or use the simplified points-only function:
#                 ax = visualise_points_only(
#                     points=[p.incenter_centroid for p in image.labels],
#                     labels=[p.class_name for p in image.labels],
#                     markersize=15,
#                     ax=ax_s,
#                     show=False,
#                     filename=filename, title=f"Visualisation of {len([p.polygon_s for p in image.labels])} labels in {image.image_name}"
#                 )
#
#
#                 plt.close()
#     logger.warning(f"If points are used then the label has to chagen from 8 to 1")
#
#     # This YOLO data.yaml sucks
#     HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
#                                                   images_train_path=report["destination_path_train"],
#                                                   images_val_path=report["destination_path_val"],
#                                                   labels_train_path=report["yolo_box_path_train"],
#                                                   labels_val_path=report["yolo_box_path_val"],
#                                                   # images_test_path=report["destination_path_test"],
#                                                   # labels_test_path=report["yolo_box_path_test"],
#                                                   class_names=report["class_names"],
#                                                   data_yaml_path=labels_path / "data_boxes.yaml")
#
#     HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
#                                                   images_train_path=report["destination_path_train"],
#                                                   images_val_path=report["destination_path_val"],
#                                                   # images_test_path=report["destination_path_test"],
#                                                   labels_train_path=report["yolo_segments_path_train"],
#                                                   labels_val_path=report["yolo_segments_path_val"],
#                                                   # labels_test_path=report["yolo_segments_path_test"],
#                                                   class_names=report["class_names"],
#                                                   data_yaml_path=labels_path / "data_segments.yaml")
