"""
Takes Boxes of Iguanas blacks them out if they are on the edge of an image.
Then keep only the points outside that area
"""

# from dataset_configs_hasty_box_point import *

import dataset_configs_hasty_box_point as dataset_configs

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
from com.biospheredata.visualization.visualize_result import visualise_image

if __name__ == "__main__":


    for dataset in dataset_configs.datasets:  # , "val", "test"]:
        dataset_dict = dataset.model_dump()

        # Add the new required fields
        dataset_dict.update({
            'labels_path': dataset_configs.labels_path,
        })
        report = DataPrepReport(**dataset_dict)

        logger.info(f"Starting {dataset.dataset_name}, split: {dataset.dset}")
        dset = dataset.dset
        num = dataset.num
        overlap = dataset.overlap
        ifcn = ImageFilterConstantNum(num=num, dataset_config=dataset)
        # output_path = dataset["output_path"]

        uA = UnpackAnnotations()
        hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped=dataset_configs.labels_path / dataset_configs.hasty_annotations_labels_zipped,
                                         hasty_annotations_images_zipped=dataset_configs.labels_path / dataset_configs.hasty_annotations_images_zipped)

        logger.info(f"Unzipped {len(hA.images)} images.")
        output_path_dset = dataset_configs.labels_path / dataset.dataset_name / dset

        output_path_dset.mkdir(exist_ok=True, parents=True)

        vis_path = dataset_configs.labels_path / f"visualisations" / f"{dataset.dataset_name}_{overlap}_{dataset_configs.crop_size}_{dset}"
        vis_path.mkdir(exist_ok=True, parents=True)

        hA_flat = hA.get_flat_df()
        logger.info(f"Flattened annotations {hA_flat} annotations.")

        dp = DataprepPipeline(annotations_labels=hA,
                              images_path=images_path,
                              crop_size=dataset_configs.crop_size,
                              overlap=overlap,
                              output_path=output_path_dset,
                              config=dataset
                              )

        dp.dataset_filter = dataset.dataset_filter
        dp.status_filter = dataset.status_filter
        dp.images_filter = dataset.images_filter
        dp.images_filter_func = [ifcn]
        dp.class_filter = dataset_configs.class_filter
        dp.annotation_types = dataset_configs.annotation_types
        dp.empty_fraction = dataset.empty_fraction
        dp.visualise_path = vis_path
        dp.use_multiprocessing = dataset_configs.multiprocessing
        dp.edge_black_out = dataset.edge_black_out

        dp.run(flatten=True)

        hA_filtered = dp.get_hA_filtered()

        # create_simple_histograms(hA.images)
        visualise_hasty_annotation_statistics(hA_filtered.images)
        bbox_statistics = plot_bbox_sizes(hA_filtered.images,
                        suffix=f"{dataset.dataset_name}_{dset}",
                        plot_name=vis_path / f"box_sizes_{dataset.dataset_name}_{dataset.dset}.png")


        hA_filtered.save(output_path_dset / f"hasty_format_full_size.json")
        # full size annotations


        df_herdnet_boxcenter = HastyConverter.convert_to_herdnet_format(hA_filtered,
                                                 output_file=output_path_dset / f"herdnet_format_boxcenter.csv",
                                                 label_mapping=dataset_configs.label_mapping, label_filter=[AnnotationType.BOUNDING_BOX])
        df_herdnet_points = HastyConverter.convert_to_herdnet_format(hA_filtered,
                                                 output_file=output_path_dset / f"herdnet_format_points.csv",
                                                 label_mapping=dataset_configs.label_mapping, label_filter=[AnnotationType.KEYPOINT])


        report.num_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
        report.num_point_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
        report.num_boxcenter_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
        report.num_images_filtered = len(hA_filtered.images)

        if dataset.crop:
            hA_crops = dp.get_hA_crops()

            annotated_images_split = hA_crops.images
            create_simple_histograms(hA_crops.images, dataset_name=dataset.dataset_name)

            # visualise_hasty_annotation_statistics(hA_crops.images)

            report.num_labels_crops = sum(len(i.labels) for i in hA_crops.images)
            report.num_images_crops = len(hA_crops.images)
            report.bbox_statistics = bbox_statistics

            aI = AnnotationsIntermediary()
            logger.info(f"After processing {len(hA_crops.images)} images remain")
            if len(hA_crops.images) == 0:
                raise ValueError("No images left after filtering")


            if dataset_configs.VISUALISE_FLAG:

                vis_path.mkdir(exist_ok=True, parents=True)
                for image in hA_crops.images:
                    logger.info(f"Visualising {image}")
                    ax_s = visualise_image(image_path=output_path_dset / f"crops_{dataset_configs.crop_size}" / image.image_name, show=False)

                    # if image.image_name == "FMO04___DJI_0906_x3072_y1024.jpg":
                    #     if len(image.labels) == 0:
                    #         raise ValueError("No labels but there should be one full iguana and a blacked out edge partial")
                    if image.image_name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg":
                        if len(image.labels) == 0:
                            raise ValueError("No labels but there should be one full iguana and some blacked out edge partial")

                    filename = vis_path / f"cropped_iguana_{image.image_name}.png"
                    # if AnnotationType.BOUNDING_BOX in annotation_types:
                    #     ax_s = visualise_polygons(polygons=[p.bbox_polygon for p in image.labels],
                    #                               labels=[p.class_name for p in image.labels], ax=ax_s,
                    #                               show=False, linewidth=2,
                    #                               filename=filename, title=f"Cropped Objects  {image.image_name} polygons")
                    if AnnotationType.KEYPOINT in dataset_configs.annotation_types:
                        visualise_points_only(points=[p.incenter_centroid for p in image.labels],
                                              labels=[p.class_name for p in image.labels], ax=ax_s,
                                              text_buffer=True, font_size=15,
                                              show=False, markersize=10,
                                              filename=filename,
                                              title=f"Cropped Objects  {image.image_name} Points")
                    plt.close()


            aI.set_hasty_annotations(hA=hA_crops)
            coco_path = aI.coco(output_path_dset / f"coco_format_{dataset_configs.crop_size}_{overlap}.json")
            images_list = dp.get_images()

            logger.info(f"Finished {dset} at {output_path_dset}")
            # TODO before uploading anything to CVAT labels need to be converted when necessary
            hA_crops.save(output_path_dset / f"hasty_format_crops_{dataset_configs.crop_size}_{overlap}.json")

            # TODO check if the conversion from polygon to point is correct
            HastyConverter.convert_to_herdnet_format(hA_crops,
                                                     output_file=output_path_dset / f"herdnet_format_{dataset_configs.crop_size}_{overlap}_crops.csv",
                                                     label_mapping=dataset_configs.label_mapping)


            stats = dp.get_stats()
            logger.info(f"Stats {dset}: {stats}")
            destination_path = output_path_dset / f"crops_{dataset_configs.crop_size}_num{num}_overlap{overlap}"

            try:
                shutil.rmtree(destination_path)
                logger.warning(f"Removed {destination_path}")
            except FileNotFoundError:
                pass
            shutil.move(output_path_dset / f"crops_{dataset_configs.crop_size}", destination_path)

            logger.info(f"Moved to {destination_path}")

            report.destination_path = destination_path
            report.edge_black_out = dataset_configs.edge_black_out

        # TODO add to the report: Datset statistiscs, number of images, number of annotations, number of classes, geojson of location
        # Save the report
        report_dict = json.loads(report.model_dump_json())
        with open(dataset_configs.labels_path / dataset.dataset_name / f"datapreparation_report_{dset}.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved report to {dataset_configs.labels_path / dataset.dataset_name / f'datapreparation_report_{dset}.yaml'}")

        shutil.rmtree(output_path_dset.joinpath(f"crops_{dataset_configs.crop_size}"), ignore_errors=True)
        if dataset.remove_default_folder:
            shutil.rmtree(output_path_dset.joinpath(HastyConverter.DEFAULT_DATASET_NAME))
        if dataset.remove_padding_folder:
            shutil.rmtree(output_path_dset.joinpath("padded_images"), ignore_errors=True)

    gc.collect()
