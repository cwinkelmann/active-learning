# config_path = Path("/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/San_STJB01_10012023/batch_workflow_report_config_San_STJB01_10012023.yaml")
import typing

import copy

import shapely
from loguru import logger
from pathlib import Path
import fiftyone as fo
import pandas as pd

from com.biospheredata.types.HastyAnnotationV2 import ImageLabel, Keypoint, AnnotatedImage, hA_from_file, \
    HastyAnnotationV2, ImageLabelCollection
from util.util import visualise_polygons, visualise_image

def cvat2hasty(hA_tiled_prediction: HastyAnnotationV2,
               dataset_name, point_field="detection", class_name = "iguana") -> typing.Tuple[pd.DataFrame, HastyAnnotationV2, typing.List[ImageLabel]]:
    """
    TODO refactor this as soon as possible
    :param hA_before_path:
    :param dataset_name:
    :return:
    """

    hA_corrected = copy.deepcopy(hA_tiled_prediction)
    hA_corrected.images = []
    iCLdl: typing.List[ImageLabel] = []

    anno_key = dataset_name

    dataset = fo.load_dataset(dataset_name)
    dataset.load_annotations(anno_key)

    # Load the view that was annotated in the App
    view = dataset.load_annotation_view(anno_key)

    stats = []

    try:
        keypoint_class_id = (
            hA_tiled_prediction.keypoint_schemas[0].keypoint_classes[0].keypoint_class_id
        )  # a bit of a hack because there is only one keypoint schema here, but there could be more
    except:
        keypoint_class_id = "ed18e0f9-095f-46ff-bc95-febf4a53f0ff"
        logger.warning("No keypoint class id found, using a default value")

    # reconstruct an annotation file
    for sample in view:
        filepath = sample.filepath
        # This is the image hash
        hasty_image_id = sample.hasty_image_id
        hasty_filename = sample.filename
        logger.info(f"Processing {hasty_filename}")
        annotated_image = [i for i in hA_tiled_prediction.images if i.image_id == sample.hasty_image_id]
        if len(annotated_image) != 1:
            logger.error(f"Image {hasty_filename} not found in hA_tiled_prediction")
            # TODO this should be an error
            continue
        image = copy.deepcopy(annotated_image[0])
        assert isinstance(image, AnnotatedImage)

        stats_row = {}
        updated_labels = []
        new_labels = []
        unchanged_labels = []
        deleted_labels = []

        if hasattr(sample, "ground_truth_boxes"):
            for kp in sample.ground_truth_boxes.detections:
                # print(kp)
                x1, y1, w, h = kp.bounding_box
                x2 = x1 + w
                y2 = y1 + h
                x1 *= image.width
                x2 *= image.width
                y1 *= image.height
                y2 *= image.height

                bbox = [int(x1), int(y1), int(x2), int(y2)]
                pt = shapely.box(*bbox).centroid

                if hasattr(kp, "hasty_id"):
                    # The object is a known object from before
                    # logger.info(f"Object {kp.hasty_id} was known before")

                    image_label = [l for l in image.labels if l.id == kp.hasty_id][0]

                    dist = pt.distance(image_label.centroid)
                    if dist > 2:
                        # The object was moved
                        # logger.info(f"Object {kp.hasty_id} was moved")

                        image_label.bbox = bbox
                        updated_labels.append(image_label)

                    else:
                        # The object was not moved
                        # logger.info(f"Object {kp.hasty_id} was not moved")

                        new_labels.append(image_label)

                else:

                    # The object is new
                    logger.info("New object")
                    il = ImageLabel(
                        class_name="iguana",
                        bbox=bbox,
                        polygon=None,
                        mask=None,
                        z_index=0,
                        keypoints=[],
                    )
                    new_labels.append(il)

        else:
            logger.info(f"Sample {sample.id} has no ground_truth_boxes")


        if hasattr(sample, point_field):
            """ are there points in the sample """
            keypoints_field = getattr(sample, point_field)


            for kp in keypoints_field.keypoints:
                # iterate over the keypoints
                pt = shapely.Point(
                    kp.points[0][0] * image.width, kp.points[0][1] * image.height
                )

                hkp = Keypoint(
                    x=int(pt.x),
                    y=int(pt.y),
                    norder=0,
                    keypoint_class_id=keypoint_class_id,
                )

                il = ImageLabel(
                    class_name=kp.label,
                    keypoints=[hkp],
                )

                if hasattr(kp, "hasty_id"):
                    # The object is a known object from before
                    logger.info(f"Object {kp.hasty_id} was known before")

                    image_label = [l for l in image.labels if l.id == kp.hasty_id][0]
                    assert isinstance(image_label, ImageLabel)
                    dist = pt.distance(image_label.incenter_centroid)

                    il.id = kp.hasty_id

                    if dist > 10:
                        # The object was moved
                        # logger.info(f"Object {kp.hasty_id} was moved")
                        il.attributes = (
                            il.attributes | image_label.attributes | {"cvat": "moved"}
                        )
                        updated_labels.append(il)
                    else:
                        # The object was not moved
                        # logger.info(f"Object {kp.hasty_id} was not moved")
                        il.attributes = (
                            il.attributes
                            | image_label.attributes
                            | {"cvat": "unchanged"}
                        )
                        unchanged_labels.append(il)

                    # every label which has a hasty_id and is here is not deleted

                # The id is not in the old labels
                else:
                    # The object is new
                    logger.info("New object")
                    il.attributes = {"cvat": "new"}
                    new_labels.append(il)

                del(kp)


        else:
            logger.info(f"Sample {sample.id} has no ground_truth_points")
            # checking if they were deleted

        stats_row["filename"] = hasty_filename

        # TODO generalise this

        updated_labels_ig = [il for il in updated_labels if il.class_name == class_name]
        new_labels_ig = [il for il in new_labels if il.class_name == class_name]
        unchanged_labels_ig = [il for il in unchanged_labels if il.class_name == class_name]

        stats_row["updated_labels"] = len(updated_labels_ig)
        stats_row["new_labels"] = len(new_labels_ig)
        stats_row["unchanged_labels"] = len(unchanged_labels_ig)
        stats_row["after_correction"] = (
            len(updated_labels_ig) + len(new_labels_ig) + len(unchanged_labels_ig)
        )
        stats_row["before_correction"] = len([il for il in image.labels if il.class_name == class_name])

        stats.append(stats_row)
        corrected_labels = updated_labels + new_labels + unchanged_labels

        old_labels = image.labels
        for ol in old_labels:
            if ol.id not in [il.id for il in corrected_labels]:
                deleted_labels.append(ol)

        iCLdl.extend( deleted_labels )

        image.labels = corrected_labels

        hA_corrected.images.append(image)


    stats_df = pd.DataFrame(stats)

    return stats_df, hA_corrected, iCLdl


