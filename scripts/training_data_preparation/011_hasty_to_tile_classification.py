"""
Create patches from images and labels from hasty annotation files to be used in CVAT/training
"""
import json

import gc
import pandas as pd

import shutil
import yaml
from loguru import logger
from pathlib import Path
from matplotlib import pyplot as plt

from active_learning.config.dataset_filter import DatasetFilterConfig, DataPrepReport
from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from active_learning.util.visualisation.annotation_vis import visualise_points_only
from com.biospheredata.converter.HastyConverter import AnnotationType, LabelingStatus
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.visualization.visualize_result import (visualise_image, visualise_polygons)
from active_learning.filter import ImageFilterConstantNumByLabel
from active_learning.util.visualisation.annotation_vis import visualise_hasty_annotation_statistics, plot_bbox_sizes

# import dataset_configs_weinstein as dataset_configs
# import dataset_configs_hasty_iguanas as dataset_configs

# import dataset_configs_eikelboom as dataset_configs
import dataset_configs_delplanque as dataset_configs

empty_class_name = "empty"

if __name__ == "__main__":

    for dataset in dataset_configs.datasets:  # , "val", "test"]:

        logger.info(f"Processing dataset {dataset.dataset_name} for {dataset.dset}")

        dataset_dict = dataset.model_dump()

        # Add the new required fields
        dataset_dict.update({
            'labels_path': dataset_configs.labels_path,
        })
        report = DataPrepReport(**dataset_dict)

        logger.info(f"Starting Dataset {dataset.dataset_name} for {dataset.dset}")
        dset = dataset.dset
        num = dataset.num
        overlap = dataset.overlap
        ifcn = ImageFilterConstantNum(num=num, dataset_config=dataset)
        ifcln = ImageFilterConstantNumByLabel(num_labels=dataset.num_labels)
        # output_path = dataset["output_path"]

        hA = HastyAnnotationV2.from_file(dataset_configs.labels_name)

        output_path_dset = dataset_configs.labels_path / dataset.dataset_name / f"detection_{dset}_{overlap}_{dataset_configs.crop_size}"
        output_path_classifcation_dset = dataset_configs.labels_path / dataset.dataset_name / f"classification_{dset}_{overlap}_{dataset_configs.crop_size}"

        output_path_dset.mkdir(exist_ok=True, parents=True)
        output_path_classifcation_dset.mkdir(exist_ok=True, parents=True)

        vis_path = dataset_configs.labels_path / f"visualisations" / f"{dataset.dataset_name}_{dset}_{overlap}_{dataset_configs.crop_size}"
        vis_path.mkdir(exist_ok=True, parents=True)

        df_hA_flat = hA.get_flat_df()
        logger.info(f"Flattened annotations {df_hA_flat} annotations.")

        def label_statistics(df_hA_flat: pd.DataFrame) -> pd.DataFrame:
            """
            Calculate label statistics from the flattened Hasty Annotation DataFrame.
            :param df_hA_flat: Flattened DataFrame of Hasty Annotations.
            :return: DataFrame with label statistics.
            """
            label_stats = df_hA_flat.groupby('class_name').agg(
                num_labels=('label_id', 'count'),
                num_images=('image_name', 'nunique')
            ).reset_index()

            total_label_count = label_stats['num_labels'].sum()

            label_stats['fraction'] = (label_stats['num_labels'] / total_label_count)

            return label_stats.to_dict(orient='records')

        hA_labels_stats = label_statistics(df_hA_flat)

        dp = DataprepPipeline(annotations_labels=hA,
                              images_path=dataset_configs.images_path,
                              crop_size=dataset_configs.crop_size,
                              overlap=overlap,
                              output_path=output_path_dset,
                              config=dataset
                              )

        dp.dataset_filter = dataset.dataset_filter

        dp.images_filter = dataset.images_filter
        dp.images_filter_func = [ifcn, ifcln]
        dp.class_filter = dataset_configs.class_filter
        dp.annotation_types = dataset_configs.annotation_types
        dp.empty_fraction = dataset.empty_fraction
        dp.num_labels = dataset.num_labels
        if dataset_configs.VISUALISE_FLAG:
            dp.visualise_path = vis_path

        dp.use_multiprocessing = dataset_configs.use_multiprocessing
        dp.edge_black_out = dataset_configs.edge_black_out

        dp.run(flatten=True)

        hA_filtered = dp.get_hA_filtered()
        bbox_statistics = plot_bbox_sizes(hA_filtered.images, suffix=f"{dataset.dataset_name}_{dset}",
                                          plot_name=vis_path / f"box_sizes_{dataset.dataset_name}_{dset}.png")
        # create_simple_histograms(hA.images)
        visualise_hasty_annotation_statistics(hA_filtered.images)
        plot_bbox_sizes(hA_filtered.images, suffix=dataset.dataset_name,
                        plot_name=vis_path / f"box_sizes_{dataset.dataset_name}.png")


        # find labels which have 0 height or width
        def has_valid_bbox_size(label, min_size=5):
            bounds = label.bbox  # (minx, miny, maxx, maxy)
            width = bounds[2] - bounds[0]  # maxx - minx
            height = bounds[3] - bounds[1]  # maxy - miny
            return width > min_size and height > min_size


        hA_filtered.images = [
            image for image in hA_filtered.images
            if all(has_valid_bbox_size(label) for label in image.labels)
        ]
        hA_filtered.save(output_path_dset / f"hasty_format_full_size.json")

        HastyConverter.convert_deep_forest(hA_filtered,
                                           output_file=output_path_dset / f"deep_forest_format.csv")

        # full size annotations
        HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dset / f"herdnet_format.csv")

        hA_crops = dp.get_hA_crops()
        aI = AnnotationsIntermediary()
        logger.info(f"After processing {len(hA_crops.images)} images remain")
        if len(hA_crops.images) == 0:
            raise ValueError("No images left after filtering")

        report.num_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
        report.num_images_filtered = len(hA_filtered.images)
        report.label_statistics = label_statistics(hA_filtered.get_flat_df())

        def get_label_mapping(hA: HastyAnnotationV2) -> dict:
            """
            Get a mapping of class names to their corresponding IDs.
            :param hA: HastyAnnotationV2 object.
            :return: Dictionary mapping class names to IDs.
            """
            return {mapping.class_id: mapping.class_name for mapping in hA.label_classes}

        report.label_mapping = get_label_mapping(hA_filtered)

        hA_crops = dp.get_hA_crops()
        report.num_labels_crops = sum(len(i.labels) for i in hA_crops.images)
        report.num_images_crops = len(hA_crops.images)

        if dataset_configs.VISUALISE_FLAG:

            vis_path.mkdir(exist_ok=True, parents=True)
            for image in hA_crops.images:
                logger.info(f"Visualising {image}")
                ax_s = visualise_image(image_path=output_path_dset / f"crops_{dataset_configs.crop_size}" / image.image_name,
                                       show=False)

                filename = vis_path / f"cropped_animal_{image.image_name}.png"
                visualise_polygons(polygons=[p.bbox_polygon for p in image.labels],
                                   labels=[p.class_name for p in image.labels], ax=ax_s,
                                   show=False, linewidth=2,
                                   filename=filename,
                                   title=f"Cropped #{len([p.class_name for p in image.labels])} Objects  {image.image_name} polygons")
                plt.close()

                ax_s = visualise_image(image_path=output_path_dset / f"crops_{dataset_configs.crop_size}" / image.image_name,
                                       show=False)

                filename = vis_path / f"cropped_animal_{image.image_name}_point.png"
                visualise_points_only(points=[p.incenter_centroid for p in image.labels],
                                      labels=[p.class_name for p in image.labels], ax=ax_s,
                                      text_buffer=True, font_size=15,
                                      show=False, markersize=10,
                                      filename=filename,
                                      title=f"Cropped {len([p.class_name for p in image.labels])} Objects  {image.image_name} Points")
                plt.close()

        aI.set_hasty_annotations(hA=hA_crops)
        coco_path = aI.coco(output_path_dset / f"coco_format_{dataset_configs.crop_size}_{overlap}.json")
        images_list = dp.get_images()

        logger.info(f"Finished {dset} at {output_path_dset}")
        hA_crops.save(output_path_dset / f"hasty_format_crops_{dataset_configs.crop_size}_{overlap}.json")

        HastyConverter.convert_to_herdnet_format(hA_crops,
                                                 output_file=output_path_dset / f"herdnet_format_{dataset_configs.crop_size}_{overlap}_crops.csv")

        if AnnotationType.BOUNDING_BOX in dataset_configs.annotation_types or AnnotationType.POLYGON in dataset_configs.annotation_types:
            HastyConverter.convert_deep_forest(hA_crops,
                                               output_file=output_path_dset / f"deep_forest_format__{dataset_configs.crop_size}_{overlap}_crops.csv")

            # TODO convert to YOLO format later from COCO
            # class_names = aI.to_YOLO_annotations(output_path=output_path_dset / "yolo")
            # report[f"yolo_box_path_{dset}"] = output_path_dset / "yolo" / f"yolo_boxes"
            # report[f"yolo_segments_path_{dset}"] = output_path_dset / "yolo" / "yolo_segments"
            # report[f"class_names"] = class_names

        # TODO move the crops to a new folder for YOLO

        output_path_classifcation_dset.joinpath(empty_class_name).mkdir(exist_ok=True)
        for acn in dataset_configs.class_filter:
            output_path_classifcation_dset.joinpath(acn).mkdir(exist_ok=True)

        # TODO move the crops to a new folder for classification

        for hA_cropped_image in hA_crops.images:
            if len(hA_cropped_image.labels) > 0:
                if len(set([label.class_name for label in hA_cropped_image.labels])) == 1:

                    shutil.copy(output_path_dset / f"crops_{dataset_configs.crop_size}" / hA_cropped_image.image_name,
                                output_path_classifcation_dset / hA_cropped_image.labels[
                                    0].class_name / hA_cropped_image.image_name)
                else:
                    logger.warning(f"There are multiple species in the image {hA_cropped_image.image_name} Skipping.")
            else:
                shutil.copy(output_path_dset / f"crops_{dataset_configs.crop_size}" / hA_cropped_image.image_name,
                            output_path_classifcation_dset / empty_class_name / hA_cropped_image.image_name)

        stats = dp.get_stats()
        logger.info(f"Stats {dset}: {stats}")
        destination_path = output_path_dset / f"crops_{dataset_configs.crop_size}_num{num}_overlap{overlap}"

        try:
            shutil.rmtree(destination_path, ignore_errors=True)
            logger.warning(f"Removed {destination_path}")
        except FileNotFoundError as e:
            logger.error(f"Could not remove {destination_path}, because of {e}")
        shutil.move(output_path_dset / f"crops_{dataset_configs.crop_size}", destination_path)

        logger.info(f"Moved to {destination_path}")

        report.destination_path = destination_path
        report.edge_black_out = dataset_configs.edge_black_out

        report_dict = json.loads(report.model_dump_json())
        with open(dataset_configs.labels_path / dataset.dataset_name / f"datapreparation_report_{dset}.yaml", 'w',
                  encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved report to {dataset_configs.labels_path / dataset.dataset_name / f'datapreparation_report_{dset}.yaml'}")

        if dataset.remove_default_folder:
            shutil.rmtree(output_path_dset.joinpath(HastyConverter.DEFAULT_DATASET_NAME), ignore_errors=True)
        if dataset.remove_padding_folder:
            shutil.rmtree(output_path_dset.joinpath("padded_images"), ignore_errors=True)

    # # YOLO Box data
    # HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
    #                                               images_train_path=report["destination_path_train"],
    #                                               images_val_path=report["destination_path_val"],
    #                                               labels_train_path=report["yolo_box_path_train"],
    #                                               labels_val_path=report["yolo_box_path_val"],
    #                                               class_names=report["class_names"],
    #                                               data_yaml_path=labels_path / "data_boxes.yaml")
    #
    # # YOLO Segmentation Data
    # HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
    #                                               images_train_path=report["destination_path_train"],
    #                                               images_val_path=report["destination_path_val"],
    #                                               labels_train_path=report["yolo_box_path_train"],
    #                                               labels_val_path=report["yolo_box_path_val"],
    #                                               class_names=report["class_names"],
    #                                               data_yaml_path=labels_path / "data_segments.yaml")

    gc.collect()
