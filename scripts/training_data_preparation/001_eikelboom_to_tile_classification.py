# """
# Create patches from images and labels from hasty annotation files to be used in CVAT/training
# """
# import json
#
# import gc
#
# import shutil
# import yaml
# from loguru import logger
# from pathlib import Path
# from matplotlib import pyplot as plt
#
# from active_learning.config.dataset_filter import DatasetFilterConfig, DataPrepReport
# from active_learning.filter import ImageFilterConstantNum
# from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
# from active_learning.util.visualisation.annotation_vis import visualise_points_only
# from com.biospheredata.converter.HastyConverter import AnnotationType, LabelingStatus
# from com.biospheredata.converter.HastyConverter import HastyConverter
# from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
# from image_template_search.util.util import (visualise_image, visualise_polygons)
#
#
# from dataset_configs_eikelboom import *
# if __name__ == "__main__":
#
#
#
#
#
#     for dataset in datasets:  # , "val", "test"]:
#         dataset_dict = dataset.model_dump()
#
#         # Add the new required fields
#         dataset_dict.update({
#             'labels_path': labels_path,
#         })
#         report = DataPrepReport(**dataset_dict)
#
#         logger.info(f"Starting {dataset.dset}")
#         dset = dataset.dset
#         num = dataset.num
#         overlap = dataset.overlap
#         ifcn = ImageFilterConstantNum(num=num, dataset_config=dataset)
#         # output_path = dataset["output_path"]
#
#         uA = UnpackAnnotations()
#         hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped=labels_path / hasty_annotations_labels_zipped,
#                                          hasty_annotations_images_zipped=labels_path / hasty_annotations_images_zipped)
#
#         logger.info(f"Unzipped {len(hA.images)} images.")
#         output_path_dset = labels_path / dataset.dataset_name / f"detection_{dset}_{overlap}_{crop_size}"
#         output_path_classifcation_dset = labels_path / dataset.dataset_name /  f"classification_{dset}_{overlap}_{crop_size}"
#
#         output_path_dset.mkdir(exist_ok=True, parents=True)
#         output_path_classifcation_dset.mkdir(exist_ok=True, parents=True)
#
#         vis_path = labels_path / f"visualisations" / f"{dataset.dataset_name}_{dset}_{overlap}_{crop_size}"
#         vis_path.mkdir(exist_ok=True, parents=True)
#
#         # hA_flat = hA.get_flat_df()
#         # logger.info(f"Flattened annotations {hA_flat} annotations.")
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
#         dp.visualise_path = vis_path
#         dp.use_multiprocessing = use_multiprocessing
#         dp.edge_black_out = edge_black_out
#
#         # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
#         dp.run(flatten=True)
#
#         hA_filtered = dp.get_hA_filtered()
#
#         report.num_images_filtered = len(hA_filtered.images)
#         hA_crops = dp.get_hA_crops()
#         report.num_labels_crops = sum(len(i.labels) for i in hA_crops.images)
#
#         # full size annotations
#         HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dset / f"herdnet_format.csv")
#
#         hA_crops = dp.get_hA_crops()
#         aI = AnnotationsIntermediary()
#         logger.info(f"After processing {len(hA_crops.images)} images remain")
#         if len(hA_crops.images) == 0:
#             raise ValueError("No images left after filtering")
#
#         report.num_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
#
#         hA_crops = dp.get_hA_crops()
#         report.num_labels_crops = sum(len(i.labels) for i in hA_crops.images)
#         report.num_images_crops = len(hA_crops.images)
#
#         if VISUALISE_FLAG:
#
#             vis_path.mkdir(exist_ok=True, parents=True)
#             for image in hA_crops.images:
#                 logger.info(f"Visualising {image}")
#                 ax_s = visualise_image(image_path=output_path_dset / f"crops_{crop_size}" / image.image_name, show=False)
#
#
#                 filename = vis_path / f"cropped_animal_{image.image_name}.png"
#                 visualise_polygons(polygons=[p.bbox_polygon for p in image.labels],
#                                    labels=[p.class_name for p in image.labels], ax=ax_s,
#                                    show=False, linewidth=2,
#                                    filename=filename,
#                                    title=f"Cropped #{len([p.class_name for p in image.labels])} Objects  {image.image_name} polygons")
#                 plt.close()
#
#                 ax_s = visualise_image(image_path=output_path_dset / f"crops_{crop_size}" / image.image_name, show=False)
#
#
#                 filename = vis_path / f"cropped_animal_{image.image_name}_point.png"
#                 visualise_points_only(points=[p.incenter_centroid for p in image.labels],
#                                       labels=[p.class_name for p in image.labels], ax=ax_s,
#                                       text_buffer=True, font_size=15,
#                                       show=False, markersize=10,
#                                       filename=filename,
#                                       title=f"Cropped {len([p.class_name for p in image.labels])} Objects  {image.image_name} Points")
#                 plt.close()
#
#         aI.set_hasty_annotations(hA=hA_crops)
#         coco_path = aI.coco(output_path_dset / f"coco_format_{crop_size}_{overlap}.json")
#         images_list = dp.get_images()
#
#         logger.info(f"Finished {dset} at {output_path_dset}")
#         # TODO before uploading anything to CVAT labels need to be converted when necessary
#         hA_crops.save(output_path_dset / f"hasty_format_crops_{crop_size}_{overlap}.json")
#
#         # TODO check if the conversion from polygon to point is correct
#         HastyConverter.convert_to_herdnet_format(hA_crops, output_file=output_path_dset / f"herdnet_format_{crop_size}_{overlap}_crops.csv")
#
#         if AnnotationType.BOUNDING_BOX in annotation_types or AnnotationType.POLYGON in annotation_types:
#             HastyConverter.convert_deep_forest(hA_crops, output_file=output_path_dset / f"deep_forest_format__{crop_size}_{overlap}_crops.csv")
#
#
#             # TODO convert to YOLO format later from COCO
#             # class_names = aI.to_YOLO_annotations(output_path=output_path_dset / "yolo")
#             # report[f"yolo_box_path_{dset}"] = output_path_dset / "yolo" / f"yolo_boxes"
#             # report[f"yolo_segments_path_{dset}"] = output_path_dset / "yolo" / "yolo_segments"
#             # report[f"class_names"] = class_names
#
#         # TODO move the crops to a new folder for YOLO
#
#         output_path_classifcation_dset.joinpath("Elephant").mkdir(exist_ok=True)
#         output_path_classifcation_dset.joinpath("Giraffe").mkdir(exist_ok=True)
#         output_path_classifcation_dset.joinpath("Zebra").mkdir(exist_ok=True)
#         output_path_classifcation_dset.joinpath("empty").mkdir(exist_ok=True)
#
#         # TODO move the crops to a new folder for classification
#
#         for hA_cropped_image in hA_crops.images:
#             if len(hA_cropped_image.labels) > 0:
#                 if len(set([label.class_name for label in hA_cropped_image.labels])) == 1:
#
#                     shutil.copy(output_path_dset / f"crops_{crop_size}" / hA_cropped_image.image_name, output_path_classifcation_dset / hA_cropped_image.labels[0].class_name / hA_cropped_image.image_name)
#                 else:
#                     logger.warning(f"There are species in the image {hA_cropped_image.image_name} Skipping.")
#             else:
#                 shutil.copy(output_path_dset / f"crops_{crop_size}" / hA_cropped_image.image_name, output_path_classifcation_dset / "empty" / hA_cropped_image.image_name)
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
#         report.destination_path = destination_path
#         report.edge_black_out = edge_black_out
#
#         report_dict = json.loads(report.model_dump_json())
#         with open(labels_path / dataset.dataset_name / f"datapreparation_report_{dset}.yaml", 'w', encoding='utf-8') as f:
#             yaml.dump(report_dict, f, default_flow_style=False, indent=2)
#
#         logger.info(f"Saved report to {labels_path / dataset.dataset_name / f'datapreparation_report_{dset}.yaml'}")
#
#         shutil.rmtree(output_path_dset.joinpath(HastyConverter.DEFAULT_DATASET_NAME))
#         shutil.rmtree(output_path_dset.joinpath("padded_images"))
#
#
#     # # YOLO Box data
#     # HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
#     #                                               images_train_path=report["destination_path_train"],
#     #                                               images_val_path=report["destination_path_val"],
#     #                                               labels_train_path=report["yolo_box_path_train"],
#     #                                               labels_val_path=report["yolo_box_path_val"],
#     #                                               class_names=report["class_names"],
#     #                                               data_yaml_path=labels_path / "data_boxes.yaml")
#     #
#     # # YOLO Segmentation Data
#     # HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
#     #                                               images_train_path=report["destination_path_train"],
#     #                                               images_val_path=report["destination_path_val"],
#     #                                               labels_train_path=report["yolo_box_path_train"],
#     #                                               labels_val_path=report["yolo_box_path_val"],
#     #                                               class_names=report["class_names"],
#     #                                               data_yaml_path=labels_path / "data_segments.yaml")
#
#     gc.collect()