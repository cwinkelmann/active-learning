"""
Create patches from images and labels from hasty annotation files to be used in CVAT/training
"""

from dataset_configs_hasty_point_iguanas import *

import json

import gc

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
from image_template_search.util.util import (visualise_image, visualise_polygons)

if __name__ == "__main__":



    for dataset in datasets:  # , "val", "test"]:
        dataset_dict = dataset.model_dump()

        # Add the new required fields
        dataset_dict.update({
            'labels_path': labels_path,
        })
        report = DataPrepReport(**dataset_dict)

        logger.info(f"Starting Dataset {dataset.dataset_name}, split: {dataset.dset}")
        dset = dataset.dset
        num = dataset.num
        overlap = dataset.overlap
        ifcn = ImageFilterConstantNum(num=num, dataset_config=dataset)
        # output_path = dataset["output_path"]

        uA = UnpackAnnotations()
        hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped=labels_path / hasty_annotations_labels_zipped,
                                         hasty_annotations_images_zipped=labels_path / hasty_annotations_images_zipped)

        logger.info(f"Unzipped {len(hA.images)} images.")
        output_path_dset = labels_path / dataset.dataset_name / dset

        output_path_dset.mkdir(exist_ok=True, parents=True)

        vis_path = labels_path / f"visualisations" / f"{dataset.dataset_name}_{overlap}_{crop_size}_{dset}"
        vis_path.mkdir(exist_ok=True, parents=True)

        # hA_flat = hA.get_flat_df()
        # logger.info(f"Flattened annotations {hA_flat} annotations.")

        dp = DataprepPipeline(annotations_labels=hA,
                              images_path=images_path,
                              crop_size=crop_size,
                              overlap=overlap,
                              output_path=output_path_dset,
                              )

        dp.dataset_filter = dataset.dataset_filter
        dp.status_filter = dataset.status_filter
        dp.images_filter = dataset.images_filter
        dp.images_filter_func = [ifcn]
        dp.class_filter = class_filter
        dp.annotation_types = annotation_types
        dp.empty_fraction = dataset.empty_fraction
        if VISUALISE_FLAG:
            dp.visualise_path = vis_path
        dp.use_multiprocessing = multiprocessing
        dp.edge_black_out = dataset.edge_black_out

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        dp.run(flatten=True)



        hA_filtered = dp.get_hA_filtered()
        hA_filtered.save(output_path_dset / f"hasty_format_full_size.json")
        # full size annotations
        HastyConverter.convert_to_herdnet_format(hA_filtered,
                                                 output_file=output_path_dset / f"herdnet_format.csv",
                                                 label_mapping=label_mapping)

        report.num_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
        report.num_images_filtered = len(hA_filtered.images)

        hA_crops = dp.get_hA_crops()
        report.num_labels_crops = sum(len(i.labels) for i in hA_crops.images) 
        report.num_images_crops = len(hA_crops.images)

        aI = AnnotationsIntermediary()
        logger.info(f"After processing {len(hA_crops.images)} images remain")
        if len(hA_crops.images) == 0:
            raise ValueError("No images left after filtering")


        if VISUALISE_FLAG:

            vis_path.mkdir(exist_ok=True, parents=True)
            for image in hA_crops.images:
                logger.info(f"Visualising {image}")
                ax_s = visualise_image(image_path=output_path_dset / f"crops_{crop_size}" / image.image_name, show=False)

                if image.image_name == "FMO03___DJI_0514_x3200_y2560.jpg":
                    if len(image.labels) == 0:
                        raise ValueError("No labels but there should be one full iguana and a blacked out edge partial")
                if image.image_name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg":
                    if len(image.labels) == 0:
                        raise ValueError("No labels but there should be one full iguana and some blacked out edge partial")

                filename = vis_path / f"cropped_iguana_{image.image_name}.png"
                if AnnotationType.BOUNDING_BOX in annotation_types:
                    ax_s = visualise_polygons(polygons=[p.bbox_polygon for p in image.labels],
                                              labels=[p.class_name for p in image.labels], ax=ax_s,
                                              show=False, linewidth=2,
                                              filename=filename, title=f"Cropped Objects  {image.image_name} polygons")
                if AnnotationType.KEYPOINT in annotation_types:
                    visualise_points_only(points=[p.incenter_centroid for p in image.labels],
                                          labels=[p.class_name for p in image.labels], ax=ax_s,
                                          text_buffer=True, font_size=15,
                                          show=False, markersize=10,
                                          filename=filename,
                                          title=f"Cropped Objects  {image.image_name} Points")
                plt.close()


        aI.set_hasty_annotations(hA=hA_crops)
        coco_path = aI.coco(output_path_dset / f"coco_format_{crop_size}_{overlap}.json")
        images_list = dp.get_images()

        logger.info(f"Finished {dset} at {output_path_dset}")
        # TODO before uploading anything to CVAT labels need to be converted when necessary
        hA_crops.save(output_path_dset / f"hasty_format_crops_{crop_size}_{overlap}.json")

        # TODO check if the conversion from polygon to point is correct
        HastyConverter.convert_to_herdnet_format(hA_crops,
                                                 output_file=output_path_dset / f"herdnet_format_{crop_size}_{overlap}_crops.csv",
                                                 label_mapping=label_mapping)


        stats = dp.get_stats()
        logger.info(f"Stats {dset}: {stats}")
        destination_path = output_path_dset / f"crops_{crop_size}_num{num}_overlap{overlap}"

        try:
            shutil.rmtree(destination_path)
            logger.warning(f"Removed {destination_path}")
        except FileNotFoundError:
            pass
        shutil.move(output_path_dset / f"crops_{crop_size}", destination_path)

        logger.info(f"Moved to {destination_path}")

        report.destination_path = destination_path


        # TODO add to the report: Datset statistiscs, number of images, number of annotations, number of classes, geojson of location
        # Save the report
        report_dict = json.loads(report.model_dump_json())
        with open(labels_path / dataset.dataset_name / f"datapreparation_report_{dset}.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved report to {labels_path / dataset.dataset_name / f'datapreparation_report_{dset}.yaml'}")

        if dataset.remove_Default:
            shutil.rmtree(output_path_dset.joinpath(HastyConverter.DEFAULT_DATASET_NAME), ignore_errors=True)
        if dataset.remove_Padding:
            shutil.rmtree(output_path_dset.joinpath("padded_images"), ignore_errors=True)


    gc.collect()

    # logger.info(f"Finished all datasets, reports saved to {labels_path}")