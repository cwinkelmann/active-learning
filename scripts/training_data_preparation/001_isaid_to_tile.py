"""
TODO repair this
Create patches from images and labels from hasty to be used in CVAT
convert the iSAID data in coco format to usable patches of px size 1024x1024
"""
import json

import shutil

from loguru import logger
from pathlib import Path

from matplotlib import pyplot as plt

from active_learning.config.dataset_filter import DatasetFilterConfig, DataPrepReport
from active_learning.filter import ImageFilterConstantNum, SampleStrategy, sample_coco
from active_learning.pipelines.data_prep import DataprepPipeline, AnnotationsIntermediary
from active_learning.util.converter import coco2yolo
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.types.status import AnnotationType
from com.biospheredata.visualization.visualize_result import visualise_image, visualise_polygons


# from com.biospheredata.types.serialisation import save_model_to_file


def get_data(iSAID_annotations_path, iSAID_images_path, dataset, sample=None, ):
    aI = AnnotationsIntermediary()
    with open(iSAID_annotations_path, "r") as f:
        coco_data = json.load(f)
        if sample is not None:
            coco_data = sample_coco(coco_data, sample)

    aI.set_coco_annotations(coco_data=coco_data, images_path=iSAID_images_path, project_name="iSAID", dataset_name=dataset)

    return aI


if __name__ == "__main__":

    # TODO add the unpacking of the zip files up here
    num = 1
    visualise_crops = True

    # TODO the sampling here is inconsistent but saves time
    # iSAID_images_path_train = Path("/Users/christian/data/training_data/2025_01_08_isaid/DOTA/train/images")
    iSAID_images_path_train = Path("/Users/christian/data/training_data/2025_01_08_isaid/isaid/train")
    datasets_hA_train = get_data(iSAID_images_path = iSAID_images_path_train,
                           iSAID_annotations_path = Path("/Users/christian/data/training_data/2025_01_08_isaid/isaid/iSAID_train.json"),
                                 sample=num, dataset="train").get_hasty_annotations()

    # iSAID_images_path_val = Path("/Users/christian/data/training_data/2025_01_08_isaid/DOTA/val/images")
    iSAID_images_path_val = Path("/Users/christian/data/training_data/2025_01_08_isaid/isaid/val")
    datasets_hA_val = get_data(iSAID_images_path=iSAID_images_path_val,
                           iSAID_annotations_path = Path("/Users/christian/data/training_data/2025_01_08_isaid/isaid/iSAID_val.json"),
                               sample=num, dataset="val").get_hasty_annotations()

    annotation_types = [AnnotationType.POLYGON]
    # class_filter = ["iguana"]
    class_filter = None

    crop_size = 224
    overlap = 0
    report = {}
    labels_path = Path("/Users/christian/data/training_data/2025_01_08_isaid")

    train_json = {
        "dset": "train",
        "annotation_data": datasets_hA_train,
        "images_path": iSAID_images_path_train,
       "num": num,
        "image_path": iSAID_images_path_train,
        "output_path": labels_path / "processed/train",
        "empty_fraction": 0.0,
        "crop_size": crop_size
        # "type": "iSAID"
    }

    val_json = {
            "dset": "val",
            "annotation_data": datasets_hA_val,
            "images_path": iSAID_images_path_val,
            "num": num,
            "image_path": iSAID_images_path_val,
            "output_path": labels_path / "processed/val",
            "empty_fraction": 0.0,
            "crop_size": crop_size
        # "type": "iSAID"

    }

    train_segments = DataPrepReport(**train_json)
    val_segments = DataPrepReport(**val_json)

    # datasets = [train_segments, val_segments]
    datasets = [train_segments]


    for dataset in datasets:  # , "val", "test"]:
        logger.info(f"Starting {dataset.dset} dataset")
        dset = dataset.dset
        dataset_name = dataset.dataset_name
        num = dataset.num
        crop_size = dataset.crop_size

        output_path_dataset_name = labels_path / dataset_name / dset
        output_path_dataset_name.mkdir(exist_ok=True, parents=True)

        images_path = dataset.images_path

        ifcn = ImageFilterConstantNum(num=num, sample_strategy=SampleStrategy.FIRST, dataset_config=dataset)

        dp = DataprepPipeline(
            annotations_labels=dataset.annotation_data,
            images_path=images_path,
              crop_size=crop_size,
              overlap=overlap,
              output_path= output_path_dataset_name,
                              )

        dp.images_filter = dataset.images_filter
        dp.class_filter = class_filter
        dp.status_filter = None
        dp.annotation_types = None
        dp.empty_fraction = 0.0
        dp.images_filter_func = ifcn

        dp.run(flatten=False)

        hA = dp.get_hA_crops()
        aI = AnnotationsIntermediary()

        aI.set_hasty_annotations(hA = hA)

        coco_path = aI.coco(output_path_dataset_name / "coco_format.json")
        images_list = dp.get_images()

        logger.info(f"Finished {dset} at {output_path_dataset_name}")
        HastyConverter.convert_to_herdnet_format(hA, output_file=output_path_dataset_name / "herdnet_format.csv")

        logger.info(f"Finished converting to herdnet, {dset} at {output_path_dataset_name}")

        if AnnotationType.BOUNDING_BOX in annotation_types or AnnotationType.POLYGON in annotation_types:
            HastyConverter.convert_deep_forest(hA, output_file=output_path_dataset_name / "deep_forest_format_crops.csv")

            class_names = aI.to_YOLO_annotations(output_path=output_path_dataset_name / "yolo")
            report[f"yolo_box_path_{dset}"] = output_path_dataset_name / "yolo" / "yolo_boxes"
            report[f"yolo_segments_path_{dset}"] = output_path_dataset_name / "yolo" / "yolo_segments"
            report[f"class_names"] = class_names

        stats = dp.get_stats()
        logger.info(f"Stats {dset}: {stats}")
        destination_path = output_path_dataset_name / f"crops_{crop_size}_num{num}_overlap{overlap}"

        try:
            shutil.move(output_path_dataset_name / f"crops_{crop_size}", destination_path )
        except:
            logger.error(f"Could not move {output_path_dataset_name / f'crops_{crop_size}'} to {destination_path}")

        logger.info(f"Moved to {destination_path}")
        report[f"destination_path_{dset}"] = destination_path


        if visualise_crops:
            vis_path = output_path_dataset_name / f"visualisations"
            vis_path.mkdir(exist_ok=True, parents=True)
            for image in hA.images:
                ax_s = visualise_image(image_path = destination_path / image.image_name, show=False)

                filename = vis_path / f"{image.image_name}.png"
                visualise_polygons(polygons=[p.polygon_s for p in image.labels],
                                   labels=[p.class_name for p in image.labels],  ax=ax_s, show=False, linewidth=2,
                                   filename=filename)
                plt.close()



    HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
                                                  images_train_path=report["destination_path_train"],
                                                  images_val_path=report["destination_path_val"],
                                                  labels_train_path=report["yolo_box_path_train"],
                                                    labels_val_path=report["yolo_box_path_val"],
                                                  class_names=report["class_names"],
                                                  data_yaml_path=labels_path / "processed" / "data_boxes.yaml")

    HastyConverter.prepare_YOLO_output_folder_str(base_path=labels_path,
                                                  images_train_path=report["destination_path_train"],
                                                  images_val_path=report["destination_path_val"],
                                                  labels_train_path=report["yolo_box_path_train"],
                                                    labels_val_path=report["yolo_box_path_val"],
                                                  class_names=report["class_names"],
                                                  data_yaml_path=labels_path / "processed" / "data_segments.yaml")