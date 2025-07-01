"""
Create patches from images and labels from hasty annotation files to be used in CVAT/training
"""
import shutil
from loguru import logger
from pathlib import Path
from matplotlib import pyplot as plt

from active_learning.config.dataset_filter import DatasetFilterConfig
from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.converter.HastyConverter import HastyConverter
from image_template_search.util.util import (visualise_image, visualise_polygons)
from image_template_search.util.visualisation import visualise_annotated_image

if __name__ == "__main__":

    ## Meeting presentation
    labels_path = Path("/Users/christian/data/training_data/2025_06_30")
    hasty_annotations_labels_zipped = "labels_completed_20250630.zip"
    hasty_annotations_images_zipped = "images_completed_20250630.zip"
    annotation_types = [AnnotationType.BOUNDING_BOX]
    class_filter = ["iguana"]

    crop_size = 224
    # overlap = 0
    VISUALISE_FLAG = False

    ## Data preparation based on segmentation masks
    train_segments = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images"],
        # "images_filter": ["DJI_0126.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })

    val_segments = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_filter": ["Fer_FCD01-02-03_20122021"],  # Fer_FCD01-02-03_20122021_single_images

        "output_path": labels_path,
        "empty_fraction": 1,
    })

    train_segments_120 = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images"],
        #"images_filter": ["DJI_0126.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 120,
    })

    val_segments_120 = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_filter": ["Fer_FCD01-02-03_20122021"],  # Fer_FCD01-02-03_20122021_single_images

        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 120
    })

    train_segments_180 = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images"],
        #"images_filter": ["DJI_0126.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 180,
    })

    val_segments_180 = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_filter": ["Fer_FCD01-02-03_20122021"],  # Fer_FCD01-02-03_20122021_single_images

        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 180
    })

    datasets = [train_segments, train_segments_120, train_segments_180, val_segments, val_segments_120, val_segments_180]
    # datasets = [train_segments, val_segments]
    # datasets = [train_segments]
    report = {}

    for dataset in datasets:  # , "val", "test"]:
        logger.info(f"Starting {dataset.dset}")
        dset = dataset.dset
        num = dataset.num
        overlap = dataset.overlap
        ifcn = ImageFilterConstantNum(num=num, dataset_config=dataset)
        # output_path = dataset["output_path"]

        uA = UnpackAnnotations()
        hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped=labels_path / hasty_annotations_labels_zipped,
                                         hasty_annotations_images_zipped=labels_path / hasty_annotations_images_zipped)

        logger.info(f"Unzipped {len(hA.images)} images.")
        output_path_dset = labels_path / dset

        output_path_classifcation_dset = labels_path / f"classification_{dset}"
        output_path_dset.mkdir(exist_ok=True)
        output_path_classifcation_dset.mkdir(exist_ok=True)
        vis_path = output_path_classifcation_dset / f"visualisations_{overlap}_{crop_size}"
        vis_path.mkdir(exist_ok=True)

        # hA_flat = hA.get_flat_df()
        # logger.info(f"Flattened annotations {hA_flat} annotations.")

        dp = DataprepPipeline(annotations_labels=hA,
                              images_path=images_path,
                              crop_size=crop_size,
                              overlap=overlap,
                              output_path=output_path_dset,
                              )

        dp.dataset_filter = dataset.dataset_filter

        dp.images_filter = dataset.images_filter
        dp.images_filter_func = [ifcn]
        dp.class_filter = class_filter
        dp.annotation_types = annotation_types
        dp.empty_fraction = dataset.empty_fraction
        dp.visualise_path = vis_path

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        dp.run(flatten=True)

        hA_filtered = dp.get_hA_filtered()
        # full size annotations
        HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dset / "herdnet_format.csv")

        hA_crops = dp.get_hA_crops()
        aI = AnnotationsIntermediary()
        logger.info(f"After processing {len(hA_crops.images)} images remain")
        if len(hA_crops.images) == 0:
            raise ValueError("No images left after filtering")


        if VISUALISE_FLAG:

            vis_path.mkdir(exist_ok=True, parents=True)
            for image in hA_crops.images:
                logger.info(f"Visualising {image}")
                ax_s = visualise_image(image_path=output_path_dset / f"crops_{crop_size}" / image.image_name, show=False)
                if image.image_name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x2464_y1120.jpg":
                    if len(image.labels) == 0:
                        raise ValueError("No labels but there should be one full iguana and a blacked out edge partial")
                if image.image_name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg":
                    if len(image.labels) == 0:
                        raise ValueError("No labels but there should be one full iguana and some blacked out edge partial")

                filename = vis_path / f"cropped_iguana_{image.image_name}.png"
                visualise_polygons(polygons=[p.bbox_polygon for p in image.labels],
                                   labels=[p.class_name for p in image.labels], ax=ax_s,
                                   show=False, linewidth=2,
                                   filename=filename, title=f"Cropped Objects  {image.image_name} polygons")
                plt.close()

        aI.set_hasty_annotations(hA=hA_crops)
        coco_path = aI.coco(output_path_dset / "coco_format.json")
        images_list = dp.get_images()

        logger.info(f"Finished {dset} at {output_path_dset}")
        # TODO before uploading anything to CVAT labels need to be converted when necessary
        hA_crops.save(output_path_dset / "hasty_format_crops.json")

        # TODO check if the conversion from polygon to point is correct
        HastyConverter.convert_to_herdnet_format(hA_crops, output_file=output_path_dset / "herdnet_format_crops.csv")

        if AnnotationType.BOUNDING_BOX in annotation_types or AnnotationType.POLYGON in annotation_types:
            HastyConverter.convert_deep_forest(hA_crops, output_file=output_path_dset / "deep_forest_format_crops.csv")

            class_names = aI.to_YOLO_annotations(output_path=output_path_dset / "yolo")
            report[f"yolo_box_path_{dset}"] = output_path_dset / "yolo" / "yolo_boxes"
            report[f"yolo_segments_path_{dset}"] = output_path_dset / "yolo" / "yolo_segments"
            report[f"class_names"] = class_names

        # TODO move the crops to a new folder for YOLO

        output_path_classifcation_dset.joinpath("iguana").mkdir(exist_ok=True)
        output_path_classifcation_dset.joinpath("empty").mkdir(exist_ok=True)

        # TODO move the crops to a new folder for classification

        for hA_cropped_image in hA_crops.images:
            if len(hA_cropped_image.labels) > 0:
                shutil.copy(output_path_dset / f"crops_{crop_size}" / hA_cropped_image.image_name, output_path_classifcation_dset / "iguana" / hA_cropped_image.image_name)
            else:
                shutil.copy(output_path_dset / f"crops_{crop_size}" / hA_cropped_image.image_name, output_path_classifcation_dset / "empty" / hA_cropped_image.image_name)

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

        report[f"destination_path_{dset}"] = destination_path


    # YOLO Box data
    HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
                                                  images_train_path=report["destination_path_train"],
                                                  images_val_path=report["destination_path_val"],
                                                  labels_train_path=report["yolo_box_path_train"],
                                                  labels_val_path=report["yolo_box_path_val"],
                                                  class_names=report["class_names"],
                                                  data_yaml_path=labels_path / "data_boxes.yaml")

    # YOLO Segmentation Data
    HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
                                                  images_train_path=report["destination_path_train"],
                                                  images_val_path=report["destination_path_val"],
                                                  labels_train_path=report["yolo_box_path_train"],
                                                  labels_val_path=report["yolo_box_path_val"],
                                                  class_names=report["class_names"],
                                                  data_yaml_path=labels_path / "data_segments.yaml")
