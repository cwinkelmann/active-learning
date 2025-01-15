"""
Create patches from images and labels from hasty to be used in CVAT
"""
import shutil

from loguru import logger
from pathlib import Path

from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.serialisation import save_model_to_file
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2
## TODO Download annotations from hasty



if __name__ == "__main__":
    """ This only works if the input is a hasty zip file which is very constraining. """

    # labels_path = Path("/Users/christian/data/training_data/2025_01_11")
    # hasty_annotations_labels_zipped = "labels_segments.zip"
    # hasty_annotations_images_zipped = "images_segments.zip"
    # annotation_types = [AnnotationType.POLYGON]
    #
    # labels_path = Path("/Users/christian/data/training_data/2024_12_09")
    # hasty_annotations_labels_zipped = "FMO02_03_05_labels.zip"
    # hasty_annotations_images_zipped = "FMO02_03_05_images.zip"
    # annotation_types = [AnnotationType.BOUNDING_BOX]

    # class_filter = ["iguana"]

    labels_path = Path("/Users/christian/data/training_data/2024_12_09_debug")
    hasty_annotations_labels_zipped = "FMO02_03_05_labels.zip"
    hasty_annotations_images_zipped = "FMO02_03_05_images.zip"
    annotation_types = [AnnotationType.KEYPOINT]

    class_filter = ["iguana_point"]

    # annotation_types = ["box"]
    # annotation_types = ["point"]


    # annotation_types = ["point", "box"]
    # class_filter = ["iguana_point", "iguana"]

    crop_size = 512
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
        # "dataset_filter": None,
        "num": 1,
        "output_path": labels_path,
    },
        {"dset": "val",
        # "images_filter": ["DJI_0465.JPG"],
         "dataset_filter": ["FMO03"],
         # "dataset_filter": None,
         "output_path": labels_path,

         },
        {"dset": "test",
         # "images_filter": ["DJI_0554.JPG"],
         "dataset_filter": ["FMO02"],
         "output_path": labels_path,
         }
    ]

    # datasets = [{
    #     "dset": "train",
    #     # "images_filter": ["DJI_0432.JPG"],
    #     # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"],
    #     "dataset_filter": ["FMO05"],
    #     "num": n
    # } for n in range(11, 12)]
    #
    # datasets.append( {"dset": "val",
    #     # "images_filter": ["DJI_0465.JPG"],
    #      "dataset_filter": ["FMO03"],
    #      })
    # datasets.append(
    #     {"dset": "test",
    #      # "images_filter": ["DJI_0554.JPG"],
    #      "dataset_filter": ["FMO02"]
    #      })

    for dataset in datasets:  # , "val", "test"]:
        dset = dataset["dset"]
        num = dataset.get("num", None)
        ifcn = ImageFilterConstantNum(num=num)
        # output_path = dataset["output_path"]

        uA = UnpackAnnotations()
        hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped= labels_path / hasty_annotations_labels_zipped,
                       hasty_annotations_images_zipped = labels_path / hasty_annotations_images_zipped)
        logger.info(f"Unzipped {len(hA.images)} images.")
        output_path_dset = labels_path / dset
        output_path_dset.mkdir(exist_ok=True)

        # TODO a config would be better than passing all these parameters
        dp = DataprepPipeline(annotations_labels=hA,
                              images_path=images_path,
                              crop_size=crop_size,
                              overlap=overlap,
                              output_path=output_path_dset,
                              )
        dp.dataset_filter = dataset["dataset_filter"]

        dp.images_filter = dataset.get("images_filter", None)
        dp.images_filter_func = ifcn
        dp.class_filter = class_filter
        dp.annotation_types = annotation_types
        dp.empty_fraction = 0.0

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        dp.run()

        hA_filtered = dp.get_hA_filtered()
        HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dset / "herdnet_format.csv")

        hA_crops = dp.get_hA_crops()
        aI = AnnotationsIntermediary()
        logger.info(f"After processing {len(hA_crops.images)} images remain")
        if len(hA_crops.images) == 0:
            raise ValueError("No images left after filtering")

        aI.set_hasty_annotations(hA=hA_crops)
        coco_path = aI.coco(output_path_dset / "coco_format.json")
        images_list = dp.get_images()

        # aI.to_YOLO_annotations(output_path=output_path.parent / "yolo", images_list=images_list, coco_path=coco_path)

        logger.info(f"Finished {dset} at {output_path_dset}")
        # TODO before uploading anything to CVAT labels need to be converted when necessary
        hA_crops.save(output_path_dset / "hasty_format_crops.json")

        HastyConverter.convert_to_herdnet_format(hA_crops, output_file=output_path_dset / "herdnet_format_crops.csv")

        if annotation_types == [AnnotationType.BOUNDING_BOX]:
            HastyConverter.convert_deep_forest(hA_crops, output_file=output_path_dset / "deep_forest_format_crops.csv")

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
