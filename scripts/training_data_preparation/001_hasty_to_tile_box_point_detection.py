"""
This becomes cool now
"""
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
from image_template_search.util.util import (visualise_image, visualise_polygons)




if __name__ == "__main__":

    ## Meeting presentation
    labels_path = Path("/Users/christian/data/training_data/2025_07_10_final_point_detection_edge_black")
    hasty_annotations_labels_zipped = "2025_07_10_labels_final.zip"
    hasty_annotations_images_zipped = "2025_07_10_images_final.zip"

    annotation_types = [AnnotationType.BOUNDING_BOX, AnnotationType.KEYPOINT]
    class_filter = ["iguana", "iguana_point"]

    # annotation_types = [AnnotationType.KEYPOINT]
    # class_filter = ["iguana_point"]

    label_mapping = {"iguana_point": 1, "iguana": 2}
    logger.warning(f"Later this should work with Box first to remove edge partials then point to mark")

# def main(labels_path: Path):

    crop_size = 512
    overlap = 0
    VISUALISE_FLAG = False
    empty_fraction = 0
    multiprocessing = False # Fixme later, currently not working with multiprocessing
    edge_black_out = True

    datasets = {
        "Floreana": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
                     'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO03', 'FMO04', 'FPA03 condor', 'FSCA02',
                     "floreana_FPE01_FECA01"],

        "Floreana_1": ['FMO03', 'FMO04', 'FPA03 condor', 'FSCA02', "floreana_FPE01_FECA01"],
        "Floreana_2": ['Floreana_22.01.21_FPC07', 'Floreana_03.02.21_FMO06', 'FLMO02_28012023', 'FLBB01_28012023',
                       'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05'],

        "Floreana_best": ['Floreana_03.02.21_FMO06', "floreana_FPE01_FECA01"
                                                     'Floreana_02.02.21_FMO01', 'FMO02', 'FMO05', 'FMO04', 'FMO03',
                          'FPA03 condor'],

        "Fernandina_s_1": ['Fer_FCD01-02-03_20122021_single_images'],
        "Fernandina_s_2": [
            'FPM01_24012023',
            'Fer_FPE02_07052024'
        ],
        "Genovesa": ['Genovesa'],

        "Fernandina_m": ['Fer_FCD01-02-03_20122021', 'Fer_FPM01-02_20122023'],
        "Fernandina_m_fcd": ['Fer_FCD01-02-03_20122021'],
        "Fernandina_m_fpm": ['Fer_FPM01-02_20122023'],

        "the_rest": [
            # "SRPB06 1053 - 1112 falcon_25.01.20", # orthomosaics contains nearly iguanas but not annotated
            "SCris_SRIL01_04022023",  # Orthomosaic
            "SCris_SRIL02_04022023",  # Orthomosaic
            "SCris_SRIL04_04022023",  # Orthomosaic, 4 iguanas

            "San_STJB01_12012023",  # Orthomosaic, 13
            "San_STJB02_12012023",  # Orthomosaic
            "San_STJB03_12012023",  # Orthomosaic
            "San_STJB04_12012023",  # Orthomosaic
            "San_STJB06_12012023",  # Orthomosaic

            "SCruz_SCM01_06012023"  # Orthomosaic
        ],

        "zooniverse_phase_2": ["Zooniverse_expert_phase_2"],
        "zooniverse_phase_3": ["Zooniverse_expert_phase_3"]
    }

    ## Data preparation for a debugging sample
    train_floreana_sample = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "floreana_sample",
        "dataset_filter": datasets["Floreana"],
        "images_filter": ["DJI_0906.JPG"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,

        # "num": 1
    })
    train_genovesa = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Genovesa_detection",
        "dataset_filter": datasets["Genovesa"],
        "images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,

    })
    val_genovesa = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Genovesa_detection",
        "dataset_filter": datasets["Genovesa"],  # Fer_FCD01-02-03_20122021_single_images
        "images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
    })
    train_floreana = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Floreana_detection",
        "dataset_filter": datasets["Floreana_1"],
        # "images_filter": [
        #     # "DJI_0064_FECA01.JPG",
        #     "DJI_0210_FPE01.JPG",
        #     # "DJI_0485_FPE01.JPG"
        # ],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        # "num": 5
        "edge_black_out": edge_black_out,
    })
    train_floreana_increasing_length = [DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": f"Floreana_detection_il_{x}",
        "dataset_filter": datasets["Floreana_1"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
        "num": x
    }) for x in range(1, 36)]

    train_fernandina_s1_increasing_length = [DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": f"Fernandina_s_detection_il_{x}",
        "dataset_filter": datasets["Fernandina_s_1"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
        "num": x
    }) for x in range(1, 25)]

    val_floreana = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Floreana_detection",
        "dataset_filter": datasets["Floreana_2"],  # Fer_FCD01-02-03_20122021_single_images
        #"images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    ## Fernandina Mosaic
    train_fernandina_m = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Fernandina_m_detection",
        "dataset_filter": datasets["Fernandina_m_fcd"],
        #"images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    val_fernandina_m = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Fernandina_m_detection",
        "dataset_filter": datasets["Fernandina_m_fpm"],  # Fer_FCD01-02-03_20122021_single_images
        #"images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    # Fernandina single images
    train_fernandina_s1 = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Fernandina_s_detection",
        "dataset_filter": datasets["Fernandina_s_1"],
        #"images_filter": ["DJI_0043_GES06.JPG", "DJI_0168_GES06.JPG", "DJI_0901_GES06.JPG", "DJI_0925_GES06.JPG"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    val_fernandina_s2 = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "Fernandina_s_detection",
        "dataset_filter": datasets["Fernandina_s_2"],  # Fer_FCD01-02-03_20122021_single_images
        #"images_filter": ["DJI_0474_GES07.JPG", "DJI_0474_GES07.JPG", "DJI_0703_GES13.JPG" ],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    # All other datasets which are just out of Orthomosaics
    train_rest = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "Rest_detection",
        "dataset_filter": datasets["the_rest"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    # All single images from all datasets
    train_single_all = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "All_detection_single",
        "dataset_filter": datasets["Floreana_1"] + datasets["Fernandina_s_1"]  + datasets["Genovesa"] + datasets["the_rest"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    val_single_all = DatasetFilterConfig(**{
        "dset": "val",
        "dataset_name": "All_detection_single",
        "dataset_filter": datasets["Floreana_2"] + datasets["Fernandina_s_2"]  + datasets["Genovesa"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })
    # All datasets combined
    train_all = DatasetFilterConfig(**{
        "dset": "train",
        "dataset_name": "All_detection",
        "dataset_filter": datasets["the_rest"] + datasets["Floreana_1"] + datasets["Fernandina_s_2"] + datasets["Fernandina_s_1"] + datasets["Fernandina_m_fpm"] + datasets["Fernandina_m_fcd"] + datasets["Genovesa"],
        "output_path": labels_path,
        "empty_fraction": empty_fraction,
        "overlap": overlap,
        "status_filter": [LabelingStatus.COMPLETED],
        "annotation_types": annotation_types,
        "class_filter": class_filter,
        "crop_size": crop_size,
        "edge_black_out": edge_black_out,
    })


    datasets = [
        train_floreana_sample,
        # train_floreana, val_floreana,
        # train_fernandina_m, val_fernandina_m,
        # train_fernandina_s1, val_fernandina_s2,
        # train_genovesa, val_genovesa,
        # train_rest,
        # train_all,
        # train_single_all,
        # val_single_all
    ]
    # datasets += train_floreana_increasing_length
    # datasets += train_fernandina_s1_increasing_length

    for dataset in datasets:  # , "val", "test"]:
        dataset_dict = dataset.model_dump()

        # Add the new required fields
        dataset_dict.update({
            'labels_path': labels_path,
        })
        report = DataPrepReport(**dataset_dict)

        logger.info(f"Starting {dataset.dataset_name}, split: {dataset.dset}")
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

        hA_flat = hA.get_flat_df()
        logger.info(f"Flattened annotations {hA_flat} annotations.")

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
        dp.use_multiprocessing = multiprocessing
        dp.edge_black_out = dataset.edge_black_out

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        dp.run(flatten=True)

        hA_filtered = dp.get_hA_filtered()

        # create_simple_histograms(hA.images)
        visualise_hasty_annotation_statistics(hA_filtered.images)
        plot_bbox_sizes(hA_filtered.images, dataset_name=dataset.dataset_name, plot_name=f"box_sizes_{dataset.dataset_name}.png")


        hA_filtered.save(output_path_dset / f"hasty_format_full_size.json")
        # full size annotations

        HastyConverter.convert_to_herdnet_format(hA_filtered,
                                                 output_file=output_path_dset / f"herdnet_format.csv",
                                                 label_mapping=label_mapping)

        report.num_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
        report.num_images_filtered = len(hA_filtered.images)

        hA_crops = dp.get_hA_crops()

        annotated_images_split = hA_crops.images
        create_simple_histograms(hA_crops.images, dataset_name=dataset.dataset_name)
        bbox_statistics = plot_bbox_sizes(hA_crops.images, dataset_name=dataset.dataset_name, plot_name=f"box_sizes_{dataset.dataset_name}.png")
        # visualise_hasty_annotation_statistics(hA_crops.images)

        report.num_labels_crops = sum(len(i.labels) for i in hA_crops.images) 
        report.num_images_crops = len(hA_crops.images)
        report.bbox_statistics = bbox_statistics

        aI = AnnotationsIntermediary()
        logger.info(f"After processing {len(hA_crops.images)} images remain")
        if len(hA_crops.images) == 0:
            raise ValueError("No images left after filtering")


        if VISUALISE_FLAG:

            vis_path.mkdir(exist_ok=True, parents=True)
            for image in hA_crops.images:
                logger.info(f"Visualising {image}")
                ax_s = visualise_image(image_path=output_path_dset / f"crops_{crop_size}" / image.image_name, show=False)

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
        report.edge_black_out = edge_black_out

        # TODO add to the report: Datset statistiscs, number of images, number of annotations, number of classes, geojson of location
        # Save the report
        report_dict = json.loads(report.model_dump_json())
        with open(labels_path / dataset.dataset_name / f"datapreparation_report_{dset}.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved report to {labels_path / dataset.dataset_name / f'datapreparation_report_{dset}.yaml'}")

        # shutil.rmtree(output_path_dset.joinpath(HastyConverter.DEFAULT_DATASET_NAME))
        # shutil.rmtree(output_path_dset.joinpath("padded_images"))

    gc.collect()

    # logger.info(f"Finished all datasets, reports saved to {labels_path}")