"""
Create patches from images and labels from hasty to be used in CVAT
"""
import shutil
from loguru import logger
from pathlib import Path

from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from active_learning.scripts.hasty_to_tile.data_configs import val_segments_fernandina_1, \
    test_segments_fernandina_1, train_segments_fernanandina_12, train_segments_points_fernanandina
from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.converter.HastyConverter import HastyConverter



if __name__ == "__main__":
    """ Take orthomosaics and geojsons to create trainign data from this"""


    # annotation_types = [AnnotationType.POLYGON]
    # class_filter = ["iguana", Cl]

    datasets = [train_FER_1]
    crop_size = 5000
    overlap = 0
    # amount of empty images in the dataset


    report = {}

    for dataset in datasets:
        logger.info(f"Starting {dataset.dataset_name}")
        dataset_name = dataset.dataset_name
        dset = dataset.dset
        num = dataset.num
        ifcn = ImageFilterConstantNum(num=num, dataset_config= dataset)

        hA =
        images_path =

        output_path_dataset_name = labels_path / dataset_name / dset
        output_path_dataset_name.mkdir(exist_ok=True, parents=True)



        dp = DataprepPipeline(annotations_labels=hA,
                              images_path=images_path,
                              crop_size=crop_size,
                              overlap=overlap,
                              output_path=output_path_dataset_name,
                              )

        dp.dataset_filter = dataset.dataset_filter
        dp.images_filter = dataset.images_filter
        dp.images_exclude = dataset.images_exclude
        dp.images_filter_func = ifcn
        dp.class_filter = dataset.class_filter
        dp.annotation_types = dataset.annotation_types
        dp.empty_fraction = dataset.empty_fraction
        dp.tag_filter = dataset.image_tags
        dp.rename_dictionary = {"iguana_point": "iguana"} # TODO add this to the config

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        dp.run(flatten=True)

        # get just the filtered annotations
        hA_filtered = dp.get_hA_filtered()
        hA_filtered.save(output_path_dataset_name / "hasty_format.json")
        aI = AnnotationsIntermediary()
        aI.set_hasty_annotations(hA=hA_filtered)
        coco_path = aI.coco(output_path_dataset_name / "coco_format.json")
        HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dataset_name / "herdnet_format.csv")

        # get the filtered and croped
        hA_crops = dp.get_hA_crops()
        aI = AnnotationsIntermediary()
        logger.info(f"After processing {len(hA_crops.images)} images remain")
        if len(hA_crops.images) == 0:
            raise ValueError("No images left after filtering")

        # if visualise_image:
        #     # TODO visualise these images
        #     visualise_image(hA_crops, images_path, output_path_dset / "visualisation")

        aI.set_hasty_annotations(hA=hA_crops)
        coco_path = aI.coco(output_path_dataset_name / "coco_format_crops.json")
        images_list = dp.get_images()


        logger.info(f"Finished {dataset_name} at {output_path_dataset_name}")
        hA_crops.save(output_path_dataset_name / "hasty_format_crops.json")

        # TODO check if the conversion from polygon to point is correct
        HastyConverter.convert_to_herdnet_format(hA_crops, output_file=output_path_dataset_name / "herdnet_format_crops.csv")


        # FIXME readd this to keep the pipeline consistent
        # if AnnotationType.BOUNDING_BOX in dataset.annotation_types or AnnotationType.POLYGON in dataset.annotation_types:
        #     HastyConverter.convert_deep_forest(hA_crops, output_file=output_path_dataset_name / "deep_forest_format_crops.csv")
        #
        #     class_names = aI.to_YOLO_annotations(output_path=output_path_dataset_name / "yolo")
        #     report[f"yolo_box_path_{dset}"] = output_path_dataset_name / "yolo" / "yolo_boxes"
        #     report[f"yolo_segments_path_{dset}"] = output_path_dataset_name / "yolo" / "yolo_segments"
        #     report[f"class_names"] = class_names


        stats = dp.get_stats()
        logger.info(f"================ Stats {dataset_name}: {stats} =========================")
        destination_path = output_path_dataset_name / f"crops_{crop_size}_num{num}_overlap{overlap}"

        try:
            shutil.rmtree(destination_path)
            logger.warning(f"Removed {destination_path}")
        except FileNotFoundError:
            pass
        shutil.move(output_path_dataset_name / f"crops_{crop_size}", destination_path)

        logger.info(f"Moved to {destination_path}")

        report[f"destination_path_{dset}"] = destination_path



    # # This YOLO data.yaml sucks
    # HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
    #                                                images_train_path=report["destination_path_train"],
    #                                               images_val_path=report["destination_path_val"],
    #                                               labels_train_path=report["yolo_box_path_train"],
    #                                                 labels_val_path=report["yolo_box_path_val"],
    #                                               images_test_path=report["destination_path_test"],
    #                                               labels_test_path=report["yolo_box_path_test"],
    #                                               class_names=report["class_names"],
    #                                               data_yaml_path=labels_path / "data_boxes.yaml")
    #
    # HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
    #                                               images_train_path=report["destination_path_train"],
    #                                               images_val_path=report["destination_path_val"],
    #                                               images_test_path=report["destination_path_test"],
    #                                               labels_train_path=report["yolo_segments_path_train"],
    #                                                 labels_val_path=report["yolo_segments_path_val"],
    #                                                 labels_test_path=report["yolo_segments_path_test"],
    #                                               class_names=report["class_names"],
    #                                               data_yaml_path=labels_path / "data_segments.yaml")

