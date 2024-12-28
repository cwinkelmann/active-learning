import shutil

from loguru import logger
from pathlib import Path

from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline
from com.biospheredata.converter.HastyConverter import HastyConverter, hasty2coco, coco2yolo
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2
from com.biospheredata.types.serialisation import save_model_to_file

## TODO Download annotations from hasty






label_schema = {
  "ground_truth": {
    "type": "detection",  # Bounding box annotations
    "classes": ["person", "car", "dog"],  # Classes for bounding boxes
    "attributes": {
      "iscrowd": {"type": "boolean", "default": False},
      "confidence": {"type": "float", "default": 0.0}
    }
  },
  "landmarks": {
    "type": "keypoints",  # Point-based annotations
    "classes": ["eye_left", "eye_right", "nose", "mouth_left", "mouth_right"],  # Point labels
    "attributes": {
      "visibility": {"type": "enum", "values": ["visible", "occluded", "not_visible"], "default": "visible"}
    }
  }
}





if __name__ == "__main__":
    labels_path = Path("/Users/christian/data/training_data/2024_12_16")
    hasty_annotations_labels_zipped = "labels_2024_12_16.zip"
    hasty_annotations_images_zipped = "images_2024_12_16.zip"

    annotation_types = ["box"]
    class_filter = ["iguana"]

    # annotation_types = ["point"]
    # class_filter = ["iguana_point"]

    # annotation_types = ["point", "box"]
    # class_filter = ["iguana_point", "iguana"]

    crop_size = 640
    overlap = 0



    # datasets = [{
    #     "dset": "train",
    #     "images_filter": ["DJI_0432.JPG"],
    #     "dataset_filter": ["FMO03"],
    # },
    #     {"dset": "val",
    #      "images_filter": ["DJI_0465.JPG"],
    #      "dataset_filter": ["FMO03"],
    #      },
    #     {"dset": "test",
    #      "images_filter": ["DJI_0554.JPG"],
    #      "dataset_filter": ["FMO03"]
    #      }
    # ]

    datasets = [{
        "dset": "train",
        # "images_filter": ["DJI_0935.JPG", "DJI_0972.JPG", "DJI_0863.JPG"],
        # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
        "dataset_filter": ["FMO05"],
        # "num": 56,
    },
        {"dset": "val",
        # "images_filter": ["DJI_0465.JPG"],
         "dataset_filter": ["FMO03"],
         },
        # {"dset": "test",
        #  # "images_filter": ["DJI_0554.JPG"],
        #  "dataset_filter": ["FMO02"]
        #  }
    ]

    datasets = [{
        "dset": "train",
        # "images_filter": ["DJI_0432.JPG"],
        # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"],
        "dataset_filter": ["FMO05"],
        "num": n
    } for n in range(1, 12)]

    datasets.append( {"dset": "val",
        # "images_filter": ["DJI_0465.JPG"],
         "dataset_filter": ["FMO03"],
         })
    # datasets.append(
    #     {"dset": "test",
    #      # "images_filter": ["DJI_0554.JPG"],
    #      "dataset_filter": ["FMO02"]
    #      })

    for dataset in datasets:  # , "val", "test"]:
        dset = dataset["dset"]
        num = dataset.get("num", None)
        ifcn = ImageFilterConstantNum(num=num)


        output_path_train = labels_path / dset
        output_path_train.mkdir(exist_ok=True)

        # TODO a config would be better than passing all these parameters
        dp = DataprepPipeline(hasty_annotations_labels_zipped=hasty_annotations_labels_zipped,
                              hasty_annotations_images_zipped=hasty_annotations_images_zipped,
                              crop_size=crop_size,
                              overlap=overlap,
                              labels_path=labels_path,
                              stage=dset,
                              output_path_train = output_path_train,
                              )
        dp.dataset_filter = dataset["dataset_filter"]

        dp.images_filter = dataset.get("images_filter", None)
        dp.images_filter_func = ifcn
        dp.class_filter = class_filter
        dp.annotation_types = annotation_types
        dp.empty_fraction = 0.0

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too

        dp.run()

        hA = dp.get_hA()
        dp.get_images()

        logger.info(f"Finished {dset} at {dp.output_path}")
        # TODO before uploading anything to CVAT labels need to be converted when necessary

        # HastyConverter.convert_to_herdnet_format(hA, output_file=dp.output_path / "herdnet_format.csv")

        HastyConverter.convert_deep_forest(hA, output_file=dp.output_path / "deep_forest_format.csv")
        coco_annotations = hasty2coco(hA)
        coco_path = dp.output_path / "coco"
        yolo_path = dp.output_path / "yolo"
        coco_path.mkdir(exist_ok=True, parents=True)
        yolo_path.mkdir(exist_ok=True, parents=True)

        save_model_to_file(coco_annotations, coco_path / "coco_format.json")

        coco2yolo(yolo_path=yolo_path, source_dir=dp.output_path,
                  coco_annotations=coco_path, images=dp.get_images())
        # HastyConverter.convert_to_yolo(hA, output_file=dp.output_path / "cvat_format.xml")

        stats = dp.get_stats()
        logger.info(f"Stats {dset}: {stats}")
        destination_path = output_path_train / f"crops_{crop_size}_num{num}_overlap{overlap}"
        shutil.move(output_path_train / f"crops_{crop_size}", destination_path )

        logger.info(f"Moved to {destination_path}")
