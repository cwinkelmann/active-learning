"""
Create patches from images and labels from hasty annotation files to be used in CVAT/training
"""
from pathlib import Path

import gc
import json
import pandas as pd
import shutil
import yaml
from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd

from active_learning.config.dataset_filter import DataPrepReport
from active_learning.filter import ImageFilterConstantNum
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from active_learning.util.visualisation.annotation_vis import visualise_points_only
from com.biospheredata.types.status import AnnotationType
from com.biospheredata.converter.HastyConverter import HastyConverter
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.visualization.visualize_result import visualise_polygons, visualise_image

iguana_special_stats = True
import dataset_configs_hasty_point_iguanas as dataset_configs

# import dataset_configs_AED as dataset_configs
# import dataset_configs_delplanque as dataset_configs
# import dataset_configs_eikelboom as dataset_configs



if __name__ == "__main__":
    if iguana_special_stats:
        # this only works if there the data is georeferenced
        # pd.DataFrame(dataset_configs.datasets_mapping)
        rows = []
        for group, datasets in dataset_configs.datasets_mapping.items():
            for dataset in datasets:
                rows.append({'Dataset_Group': group, 'Dataset_ID': dataset})

        df = pd.DataFrame(rows)
        df_grouped = df.groupby('Dataset_Group')['Dataset_ID'].agg(lambda x: ', '.join(x)).reset_index()
        df_grouped.to_latex("dataset_mapping.tex", index=False)
        base_path = Path("/raid/cwinkelmann/work/active_learning/mapping/database/")
        base_path_mapping = base_path / "mapping"
        base_path_mapping.mkdir(parents=True, exist_ok=True)
        flight_database_path = Path(base_path / "2020_2021_2022_2023_2024_database_analysis_ready.parquet")
        CRS_utm_zone_15 = "32715"
        import geopandas as gpd
        flight_database = gpd.read_parquet(flight_database_path).to_crs(epsg=CRS_utm_zone_15)
        mission_database = gdf = gpd.read_file('path/to/your_file.gpkg', layer='iguana_missions')


    for dataset in dataset_configs.datasets:
        dataset_dict = dataset.model_dump()

        logger.info(f"Starting Dataset {dataset.dataset_name}, split: {dataset.dset}")
        dset = dataset.dset
        num = dataset.num
        overlap = dataset.overlap
        ifcn = ImageFilterConstantNum(num=num, dataset_config=dataset)

        if dataset_configs.unpack:
            uA = UnpackAnnotations()
            hA, images_path = uA.unzip_hasty(
                hasty_annotations_labels_zipped=dataset_configs.labels_path / dataset_configs.hasty_annotations_labels_zipped,
                hasty_annotations_images_zipped=dataset_configs.labels_path / dataset_configs.hasty_annotations_images_zipped)

            # Add the new required fields
            dataset_dict.update({
                'labels_path': dataset_configs.hasty_annotations_images_zipped,
            })

        else:
            hA = HastyAnnotationV2.from_file(dataset_configs.labels_path / dataset_configs.labels_name)
            images_path = dataset_configs.images_path

            # Add the new required fields
            dataset_dict.update({
                'labels_path': dataset_configs.labels_path,
            })
        report = DataPrepReport(**dataset_dict)
        report.label_mapping = dataset_configs.label_mapping
        # report.images = [i.name for i in hA.images]
        
        
        logger.info(f"Unzipped {len(hA.images)} images.")
        output_path_dset = dataset_configs.labels_path / dataset.dataset_name / dset
        # output_path_dset = dataset_configs.labels_path / dataset.dataset_name / f"detection_{dset}_{overlap}_{dataset_configs.crop_size}"

        output_path_dset.mkdir(exist_ok=True, parents=True)

        vis_path = dataset_configs.labels_path / f"visualisations" / f"{dataset.dataset_name}_{overlap}_{dataset_configs.crop_size}_{dset}"
        vis_path.mkdir(exist_ok=True, parents=True)

        # hA_flat = hA.get_flat_df()
        # logger.info(f"Flattened annotations {hA_flat} annotations.")

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
        if dataset_configs.VISUALISE_FLAG:
            dp.visualise_path = vis_path
        dp.use_multiprocessing = dataset_configs.use_multiprocessing
        dp.edge_black_out = dataset.edge_black_out
        # dp.flight_database = flight_database if iguana_special_stats else None

        dp.run(flatten=dataset_configs.flatten)
        logger.info(f"done filtering")
        hA_filtered = dp.get_hA_filtered()
        hA_filtered.save(output_path_dset / f"hasty_format_full_size.json")
        # full size annotations
        HastyConverter.convert_to_herdnet_format(hA_filtered,
                                                 output_file=output_path_dset / f"herdnet_format.csv",
                                                 label_mapping=dataset_configs.label_mapping)
        aIf = AnnotationsIntermediary()
        aIf.set_hasty_annotations(hA=hA_filtered)
        coco_path = aIf.coco(output_path_dset / f"coco_format_full_size.json")
        logger.info(f"Saved coco to {coco_path}")
        report.num_labels_filtered = sum(len(i.labels) for i in hA_filtered.images)
        report.num_images_filtered = len(hA_filtered.images)

        logger.info(f"Creating crops of size {dataset_configs.crop_size} with overlap {overlap}")
        hA_crops = dp.get_hA_crops()
        report.num_labels_crops = sum(len(i.labels) for i in hA_crops.images)
        report.num_images_crops = len(hA_crops.images)

        aI = AnnotationsIntermediary()
        logger.info(f"After processing {len(hA_crops.images)} images remain")
        if len(hA_crops.images) == 0:
            raise ValueError("No images left after filtering")

        if dataset_configs.VISUALISE_FLAG:

            vis_path.mkdir(exist_ok=True, parents=True)
            for image in hA_crops.images:

                if len(image.labels) == 0:
                    logger.warning(f"No labels for {image.image_name}, skipping visualisation")
                    continue
                logger.info(f"Visualising {image}")
                ax_s = visualise_image(
                    image_path=output_path_dset / f"crops_{dataset_configs.crop_size}" / image.image_name,
                    show=False, figsize=(9,9))

                # if image.image_name == "FMO03___DJI_0514_x3200_y2560.jpg":
                #     if len(image.labels) == 0:
                #         raise ValueError("No labels but there should be one full iguana and a blacked out edge partial")
                # if image.image_name == "Fer_FCD01-02-03_20122021_single_images___DJI_0126_x4480_y1120.jpg":
                #     if len(image.labels) == 0:
                #         raise ValueError(
                #             "No labels but there should be one full iguana and some blacked out edge partial")

                filename = vis_path / (f"cropped_"
                                       f"{image.image_name}.png")

                if AnnotationType.BOUNDING_BOX in dataset_configs.annotation_types:
                    ax_s = visualise_polygons(polygons=[p.bbox_polygon for p in image.labels],
                                              labels=[p.class_name for p in image.labels], ax=ax_s,
                                              show=False, linewidth=2,
                                              filename=filename,
                                              title=f"Cropped Objects  {image.image_name} polygons")

                if AnnotationType.KEYPOINT in dataset_configs.annotation_types:
                    visualise_points_only(points=[p.incenter_centroid for p in image.labels],
                                          labels=[p.class_name for p in image.labels], ax=ax_s,
                                          text_buffer=True, font_size=15,
                                          show=False, markersize=10,
                                          filename=filename,
                                          title=f"Cropped Objects {image.image_name} Points")
                plt.close()

        aI.set_hasty_annotations(hA=hA_crops)
        coco_path = aI.coco(output_path_dset / f"coco_format_{dataset_configs.crop_size}_{overlap}.json")
        images_list = dp.get_images()

        logger.info(f"Finished {dset} at {output_path_dset}")
        hA_crops.save(output_path_dset / f"hasty_format_crops_{dataset_configs.crop_size}_{overlap}.json")

        HastyConverter.convert_to_herdnet_format(hA_crops,
                                                 output_file=output_path_dset / f"herdnet_format_{dataset_configs.crop_size}_{overlap}_crops.csv",
                                                 label_mapping=dataset_configs.label_mapping)

        stats = dp.get_stats()
        logger.info(f"Stats {dset}: {stats}")
        # destination_path = output_path_dset / f"crops_{dataset_configs.crop_size}_num{num}_overlap{overlap}"
        destination_path = output_path_dset / f"crops_{dataset_configs.crop_size}_num{num}_overlap{overlap}"

        try:
            shutil.rmtree(destination_path)
            logger.warning(f"Removed {destination_path} prior to writing the new dataset. This is to avoid stale data to remain there.")
        except FileNotFoundError:
            pass
        shutil.move(output_path_dset / f"crops_{dataset_configs.crop_size}", destination_path)

        logger.info(f"Moved to {destination_path}")

        report.destination_path = destination_path

        # Save the report
        report_dict = json.loads(report.model_dump_json())
        with open(dataset_configs.labels_path / dataset.dataset_name / f"datapreparation_report_{dset}.yaml", 'w',
                  encoding='utf-8') as f:
            yaml.dump(report_dict, f,
                      default_flow_style=False, indent=2)

        logger.info(
            f"Saved report to {dataset_configs.labels_path / dataset.dataset_name / f'datapreparation_report_{dset}.yaml'}")

        if dataset.remove_default_folder:
            shutil.rmtree(output_path_dset.joinpath(HastyConverter.DEFAULT_DATASET_NAME), ignore_errors=True)
        if dataset.remove_padding_folder:
            shutil.rmtree(output_path_dset.joinpath("padded_images"), ignore_errors=True)

    gc.collect()

    # logger.info(f"Finished all datasets, reports saved to {labels_path}")
