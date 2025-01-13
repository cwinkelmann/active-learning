"""
Create patches from images and labels from hasty to be used in CVAT
convert the iSAID data in coco format to usable patches of px size 1024x1024
"""
import json

import shutil

from loguru import logger
from pathlib import Path

from active_learning.filter import ImageFilterConstantNum, SampleStrategy, sample_coco
from active_learning.pipelines.data_prep import DataprepPipeline, AnnotationsIntermediary
from active_learning.util.converter import coco2yolo
from com.biospheredata.converter.HastyConverter import HastyConverter


# from com.biospheredata.types.serialisation import save_model_to_file


def get_data(iSAID_annotations_path, iSAID_images_path, sample=None):
    aI = AnnotationsIntermediary()
    with open(iSAID_annotations_path, "r") as f:
        coco_data = json.load(f)
        if sample is not None:
            coco_data = sample_coco(coco_data, sample)

    aI.set_coco_annotations(coco_data=coco_data, images_path=iSAID_images_path)

    return aI


if __name__ == "__main__":

    # TODO add the unpacking of the zip files up here
    num = 2
    iSAID_images_path_train = Path("/Users/christian/data/training_data/2025_01_08_isaid/DOTA/train/images")
    datasets_hA_train = get_data(iSAID_images_path = iSAID_images_path_train,
                           iSAID_annotations_path = Path("/Users/christian/data/training_data/2025_01_08_isaid/train/Annotations/iSAID_train.json"), sample=num)

    iSAID_images_path_val = Path("/Users/christian/data/training_data/2025_01_08_isaid/DOTA/val/images")
    datasets_hA_val = get_data(iSAID_images_path=iSAID_images_path_val,
                           iSAID_annotations_path = Path("/Users/christian/data/training_data/2025_01_08_isaid/val/Annotations/iSAID_val.json"), sample=num)

    # annotation_types = ["segmentation"]
    # class_filter = ["iguana"]
    class_filter = None

    crop_size = 1024
    overlap = 0

    datasets = [{
        "dset": "train",
        "data": datasets_hA_train,
        "images_path": iSAID_images_path_train,
        # "images_filter": ["DJI_0935.JPG", "DJI_0972.JPG", "DJI_0863.JPG"],
        # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
        # "dataset_filter": ["FMO05"],
        "num": num,
        "output_path": Path("/Users/christian/data/training_data/2025_01_08_isaid/processed/train"),
    },
        {
            "dset": "val",
            "data": datasets_hA_val,
            "images_path": iSAID_images_path_val,

            # "images_filter": ["DJI_0465.JPG"],
             # "dataset_filter": ["FMO03"],
            "num": num,
            "output_path": Path("/Users/christian/data/training_data/2025_01_08_isaid/processed/val"),
        },

    ]


    for dataset in datasets:  # , "val", "test"]:
        dset = dataset["dset"]
        data = dataset["data"]
        num = dataset.get("num", None)
        output_path = dataset["output_path"]
        images_path = dataset["images_path"]

        ifcn = ImageFilterConstantNum(num=num, sample_strategy=SampleStrategy.FIRST)

        # TODO a config would be better than passing all these parameters
        dp = DataprepPipeline(
            annotations_labels=data.get_hasty_annotations(),
            images_path=images_path,
              crop_size=crop_size,
              overlap=overlap,
              output_path= output_path,
                              )

        dp.images_filter = dataset.get("images_filter", None)
        dp.class_filter = class_filter
        dp.status_filter = None
        dp.annotation_types = None
        dp.empty_fraction = 0.0
        dp.images_filter_func = ifcn

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        # dp.hA = hA

        dp.run()

        hA = dp.get_hA()
        aI = AnnotationsIntermediary()

        # TODO do need the images path?
        aI.set_hasty_annotations(hA = hA)

        #$$$$$$$$ TODO convert annotation type

        # TODO make boxes out of masks

        coco_path = aI.coco(output_path / "coco_crops" / "coco_format.json")
        images_list = dp.get_images()

        # aI.to_YOLO_annotations(output_path=output_path.parent / "yolo", images_list=images_list, coco_path=coco_path)

        logger.info(f"Finished {dset} at {output_path}")
        # TODO before uploading anything to CVAT labels need to be converted when necessary

        HastyConverter.convert_to_herdnet_format(hA, output_file=output_path / "herdnet_format.csv")

        HastyConverter.convert_deep_forest(hA, output_file=output_path / "deep_forest_format.csv")

        stats = dp.get_stats()
        logger.info(f"Stats {dset}: {stats}")
        destination_path = output_path / f"crops_{crop_size}_num{num}_overlap{overlap}"
        shutil.move(output_path / f"crops_{crop_size}", destination_path )

        logger.info(f"Moved to {destination_path}")
