import json
import shutil
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sahi import AutoDetectionModel

from com.biospheredata.helper.candidate_proposal import filter_for_non_empty_images, \
    sliced_prediction
from com.biospheredata.helper.image.identifier import get_image_id, get_mosaic_slice_name
from com.biospheredata.helper.image.manipulation.convert import convert_geotiff_to_jpg
from com.biospheredata.helper.image.manipulation.slice import slice_very_big_raster
from com.biospheredata.helper.filenames import get_dataset_image_merged_filesname
from com.biospheredata.types.HastyAnnotation import HastyAnnotation
from com.biospheredata.types.Mission import Mission


class CandidateProposal(object):
    """
    A Candidate proposal is a prediction on an image with a high propability of that object being there
    """


    @staticmethod
    def candidate_proposal_prediction_from_mission(
            mission: Mission,
            model_path,
            image_size=None,
            logger=None,
            mosaic=False,
            slice_size=1280,
            export_visuals_path: bool | Path = False
    ) -> (HastyAnnotation, list, Path):
        """

        generate annotations from the model. It can take an orthophoto or the single images

        @param slice_size: hovering window
        @param export_visuals_path: bool | Path
        @param mission:
        @param model_path:
        @param image_size: dimension the original image is going to be cut into
        @param logger:
        @param mosaic: if the candidate proposal should be done using the mosaic or the single images
        @return: (HastyAnnotation, list, Path)
        """
        # detection_model = Yolov5DetectionModel( model_path=str(model_path), confidence_threshold=0.4, device="cpu")   # or 'cuda:0'

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov5',
            model_path=str(model_path),
            confidence_threshold=0.5,
            device=str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        )

        base_path = mission.base_path

        """ take the orthophoto of the mission and do the sliced prediction on top of that"""
        if mosaic:
            image_name = mission.get_geotiff_path()

            logger.info(
                f"Start slicing the image {image_name} into tiles of dimensions {image_size} by {image_size} pieces")

            ## we switch the base path because the slices are located somewhere else
            base_path, image_names, slice_dict = slice_very_big_raster(base_path,
                                                                       image_name,
                                                                       x_size=image_size,
                                                                       y_size=image_size,
                                                                       FORCE_UPDATE=True,
                                                                       FILENAME_PREFIX=get_mosaic_slice_name(mission.mission_name, mission.creation_date))

            for sliced_jpg_image_path, sliced_tif_image_path in slice_dict.items():
                convert_geotiff_to_jpg(FORCE_UPDATE=True,
                                       sliced_jpg_image_path=sliced_jpg_image_path,
                                       sliced_tif_image_path=sliced_tif_image_path
                                       )

            image_names = [x.parts[-1] for x in image_names]
            logger.info(f"Sliced the image {image_name} into {len(image_names)} pieces")

            ## remove non usable images
            logger.info(f"image list length before filtering for nearly empty images: {len(image_names)}")
            image_names = filter_for_non_empty_images(base_path=base_path, image_names=image_names)
            logger.info(f"image list length after filtering for nearly empty images: {len(image_names)}")

        else:
            """ take the single images of the mission and do the sliced prediction on top of that """
            logger.info(f"continue working with the individual images from the mission")
            image_names = mission.get_images(absolute=False)

        hA = HastyAnnotation(project_name="prediction_debug",
                             create_date="2022-03-20T18:05:30Z",
                             export_date="2022-04-20T13:52:33Z",
                             detection_model=detection_model)  ## TODO don't theses hardcoded values

        hA.set_labelclasses_from_model(detection_model=detection_model)
        hA.set_base_path(base_path)

        if export_visuals_path:
            export_visuals_path = export_visuals_path.joinpath("exported_visuals")
            export_visuals_path.mkdir(exist_ok=True)

        for image_name in image_names:
            logger.info(f"predicting on: {image_name}")

            result = sliced_prediction(
                base_path=base_path,
                image_name=image_name,
                detection_model=detection_model,
                slice_width=slice_size,
                slice_height=slice_size,
            )

            image_name_zip = f"{image_name}"


            if len(result.object_prediction_list) > 0:

                # export the visuals here:
                if export_visuals_path:
                    # https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80

                    result.export_visuals(export_dir=str(export_visuals_path),
                                          file_name=image_name,
                                          text_size=1.5)

                    logger.info(f"exported visuals to {export_visuals_path.joinpath(image_name)}")

                image_id = get_image_id(base_path.joinpath(image_name))

                hA.add_image(image_id=image_id,
                             width=result.image.width,
                             height=result.image.height,
                             image_name=image_name_zip,
                             dataset_name=mission.mission_name,
                             )

                for r in result.object_prediction_list:
                    bbox = r.x1y1x2y2.to_voc_bbox()
                    label_id = "06c7492c-051c-4a45-8c4b-68ffb06a1935"  ## TODO fix this
                    # label_id = str(uuid.uuid1())
                    hA.add_label(image_id=image_id, id=str(label_id),
                                 class_name=r.category.name, bbox=bbox)


        return hA, image_names, base_path

    @staticmethod
    def generate_candidate_proposals(missions: [Mission],
                                     yolov5_model_path: Path,
                                     searched_label: str,
                                     output: Path,
                                     image_size=None,
                                     slice_size=640,
                                     mosaic=False
                                     ):
        """
        execute everything which is needed to start correcting the labels

        @param mosaic:
        @param slice_size:
        @param image_size:
        @param missions:
        @param yolov5_model_path:
        @param searched_label:
        @param output:
        @return:
        """
        annotation_collection = []
        if mosaic:
            tmp_output = output.joinpath("candidate_proposals_mosaic")
        else:
            tmp_output = output.joinpath("candidate_proposals")

        tmp_output.mkdir(exist_ok=True, parents=True)

        for mission in missions:
            hA_1, image_names_1, sliced_image_path_1 = CandidateProposal.candidate_proposal_prediction_from_mission(
                mission=mission,
                model_path=yolov5_model_path,
                slice_size=slice_size,
                image_size=image_size,
                logger=logger,
                mosaic=mosaic,
                export_visuals_path=output,
            )
            annotation_collection.append(hA_1)
            annotation_file_filtered = hA_1.persist(base_path=tmp_output, filename="hasty_annotation_all.json")

            #mission.set_hasty_annotation_file(str(annotation_file))
            #mission.set_hasty_annotation_file_mosaic(str(annotation_file))
            #mission.persist()

            hA_images_with_more_than_zero_turtles_1 = \
                HastyAnnotation.static_get_images_with_more_than(hA=hA_1,
                                                                searched_label=searched_label,
                                                                threshold=0)

            for image_dict in hA_images_with_more_than_zero_turtles_1.images_dict.values():
                dsn = image_dict["dataset_name"]
                imn = image_dict["image_name"]
                iid = image_dict["image_id"]
                new_image_name = get_dataset_image_merged_filesname(dsn=dsn, imn=imn, iid=iid)

                shutil.copyfile(
                    sliced_image_path_1.joinpath(imn),
                    tmp_output.joinpath(f"{new_image_name}")
                )

            # hA_images_with_more_than_zero_turtles_1.update_image_name_with_ds()

        hA_images_with_more_than_zero_turtles_1 = annotation_collection[0]
        del annotation_collection[0]
        hA_filtered = HastyAnnotation.merge_hasty_annotations(hA_1=hA_images_with_more_than_zero_turtles_1,
                                                     hA_list=annotation_collection)

        logger.info(f"wrote candidate proposal to {tmp_output}")
        return hA_filtered.persist(base_path=tmp_output, filename="hasty_annotation_filtered.json")
