"""
Create patches from images and labels from hasty annotation files to be used in CVAT/training
"""
import gc

import shutil
from loguru import logger
from pathlib import Path
from matplotlib import pyplot as plt

from active_learning.config.dataset_filter import DatasetFilterConfig
from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from image_template_search.util.util import (visualise_image, visualise_polygons)
from image_template_search.util.visualisation import visualise_annotated_image

if __name__ == "__main__":

    ## Meeting presentation
    labels_path = Path("/Users/christian/data/training_data/2025-07-02")
    hasty_annotations_labels_zipped = "labels_2025_07_02.zip"
    hasty_annotations_images_zipped = "images_2025_07_02.zip"
    annotation_types = [AnnotationType.BOUNDING_BOX]
    class_filter = ["iguana"]

    crop_size = 512
    # overlap = 0
    VISUALISE_FLAG = False

    # island = "Fernandina_s"  # or "Fernandina", "Santiago", "San Cristobal", "Santa Cruz", "Genovesa"

    datasets = {
        "Floreana": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
                     'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO03', 'FMO04', 'FPA03 condor', 'FSCA02',
                     'floreana'],

        "Floreana_1": ['FMO03', 'FMO04', 'FPA03 condor', 'FSCA02', 'floreana'],
        "Floreana_2": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
                     'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05'],

        "Floreana_best": ['Floreana_03.02.21_FMO06',
                          'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO04', 'FMO03', 'FPA03 condor'],

        "Fernandina_s_1": ['Fer_FCD01-02-03_20122021_single_images'],
        "Fernandina_s_2": [
                         'FPM01_24012023',
                         'Fer_FPE02_07052024'
        ],
        "Genovesa": ['Genovesa'],
        "Fernandina_m": ['Fer_FCD01-02-03_20122021', 'Fer_FPM01-02_20122023'],
        "Fernandina_m_fcd": ['Fer_FCD01-02-03_20122021'],
        "Fernandina_m_fpm": ['Fer_FPM01-02_20122023'],
    }

    # dataset_filter = datasets[island]


    ## Data preparation for a debugging sample
    train_floreana_sample = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "floreana_sample",
        "dataset_filter": datasets["Floreana_best"],
        "images_filter": ["DJI_0514.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
        # "num": 10
    })


    train_genovesa = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Genovesa_classification",
        "dataset_filter": datasets["Genovesa"],
        "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })
    val_genovesa = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Genovesa_classification",
        "dataset_filter": datasets["Genovesa"],  # Fer_FCD01-02-03_20122021_single_images
        "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })


    ## Data preparation based on segmentation masks
    train_floreana = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Floreana_classification",
        "dataset_filter": datasets["Floreana_1"],
        #"images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })
    val_floreana = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Floreana_classification",
        "dataset_filter": datasets["Floreana_2"],  # Fer_FCD01-02-03_20122021_single_images
        #"images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })


    ## Fernandina Mosaic
    train_fernandina_m = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Fernandina_m_classification",
        "dataset_filter": datasets["Fernandina_m_fcd"],
        #"images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })
    val_fernandina_m = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Fernandina_m_classification",
        "dataset_filter": datasets["Fernandina_m_fpm"],  # Fer_FCD01-02-03_20122021_single_images
        #"images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })


    # classification Fernandina single images
    train_fernandina_s1 = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Fernandina_s_classification",
        "dataset_filter": datasets["Fernandina_s_1"],
        #"images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })
    val_fernandina_s2 = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Fernandina_s_classification",
        "dataset_filter": datasets["Fernandina_s_2"],  # Fer_FCD01-02-03_20122021_single_images
        #"images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": 1,
        "overlap": 0,
    })



    datasets = [
        train_floreana_sample,
        # train_floreana, val_floreana,
        # train_fernandina_m, val_fernandina_m,
        # train_fernandina_s1, val_fernandina_s2,
        # train_genovesa, val_genovesa
    ]

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
        output_path_dset = labels_path / dataset.dataset_name / dset

        output_path_classifcation_dset = labels_path / dataset.dataset_name /  f"classification_{dset}"
        output_path_dset.mkdir(exist_ok=True, parents=True)
        output_path_classifcation_dset.mkdir(exist_ok=True, parents=True)

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
        dp.visualise_path = vis_path
        dp.use_multiprocessing = False
        dp.edge_black_out = True

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        dp.run(flatten=True)

        hA_filtered = dp.get_hA_filtered()
        # full size annotations
        HastyConverter.convert_to_herdnet_format(hA_filtered, output_file=output_path_dset / f"herdnet_format.csv")

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
                if image.image_name == "FMO03___DJI_0514_x3200_y2560.jpg":
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
        coco_path = aI.coco(output_path_dset / f"coco_format_{crop_size}_{overlap}.json")
        images_list = dp.get_images()

        logger.info(f"Finished {dset} at {output_path_dset}")
        # TODO before uploading anything to CVAT labels need to be converted when necessary
        hA_crops.save(output_path_dset / f"hasty_format_crops_{crop_size}_{overlap}.json")

        # TODO check if the conversion from polygon to point is correct
        HastyConverter.convert_to_herdnet_format(hA_crops, output_file=output_path_dset / f"herdnet_format_{crop_size}_{overlap}_crops.csv")

        if AnnotationType.BOUNDING_BOX in annotation_types or AnnotationType.POLYGON in annotation_types:
            HastyConverter.convert_deep_forest(hA_crops, output_file=output_path_dset / f"deep_forest_format__{crop_size}_{overlap}_crops.csv")

            class_names = aI.to_YOLO_annotations(output_path=output_path_dset / "yolo")
            report[f"yolo_box_path_{dset}"] = output_path_dset / "yolo" / f"yolo_boxes"
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

    gc.collect()