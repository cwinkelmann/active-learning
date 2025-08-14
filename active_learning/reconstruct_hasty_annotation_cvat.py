# config_path = Path("/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/San_STJB01_10012023/batch_workflow_report_config_San_STJB01_10012023.yaml")
import os

import uuid

import typing

import copy

import shapely
from loguru import logger
from pathlib import Path
import fiftyone as fo
import pandas as pd

from com.biospheredata.types.HastyAnnotationV2 import ImageLabel, Keypoint, AnnotatedImage, hA_from_file, \
    HastyAnnotationV2, ImageLabelCollection
from PIL import Image


def cvat2hasty(hA_tiled_prediction: HastyAnnotationV2,
               dataset_name, point_field="detection", class_name = "iguana") -> typing.Tuple[pd.DataFrame, HastyAnnotationV2, typing.List[ImageLabel]]:
    """
    TODO refactor this as soon as possible
    :param hA_before_path:
    :param dataset_name:
    :return:
    """
    logger.warning(f"This is a terrible function. Use cvat2hasty_v2")
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


def download_cvat_annotations(dataset_name, anno_key=None):
    """
    Downloads annotations from CVAT and prepares them for processing.

    Args:
        dataset_name: Name of the FiftyOne dataset to load
        anno_key: The annotation key to use (defaults to dataset_name if None)

    Returns:
        tuple: (dataset view with annotations, FiftyOne dataset)
    """
    if anno_key is None:
        anno_key = dataset_name

    # Load the dataset
    dataset = fo.load_dataset(dataset_name)
    dataset.load_annotations(anno_key)

    # Load the view that was annotated in the App
    view = dataset.load_annotation_view(anno_key)

    logger.info(f"Loaded annotations for dataset {dataset_name} with {len(view)} samples")

    return view, dataset


def foDataset2Hasty(hA_template: HastyAnnotationV2,
                    dataset,
                    anno_field=None,
                    point_field="detection",
                    keypoint_class_id=None):
    """
    Converts a FiftyOne dataset with annotations into Hasty annotation format,
    using an existing HastyAnnotationV2 template.

    Args:
        hA_template: Template HastyAnnotationV2 object with schemas and classes defined
        dataset: FiftyOne dataset or view containing annotations
        anno_field: Field name for bounding box annotations (if None, will look for 'ground_truth_boxes')
        point_field: Field name for point annotations (default: "detection")
        class_name: Target class name (default: "iguana")
        keypoint_class_id: ID for keypoint class (if None, will use from template)

    Returns:
        HastyAnnotationV2: A Hasty annotation object populated with data from the FiftyOne dataset
    """
    # Create a deep copy of the template to avoid modifying it
    hasty_annotation = copy.deepcopy(hA_template)

    # Clear out any existing images but keep schemas, classes, etc.
    hasty_annotation.images = []

    # Find the keypoint class ID if not provided
    if keypoint_class_id is None:
        for schema in hasty_annotation.keypoint_schemas:
            for kp_class in schema.keypoint_classes:
                if "body" in kp_class.keypoint_class_name.lower():
                    keypoint_class_id = kp_class.keypoint_class_id
                    break
            if keypoint_class_id:
                break

        # If still not found, use default
        if not keypoint_class_id:
            keypoint_class_id = "ed18e0f9-095f-46ff-bc95-febf4a53f0ff"

    # Process each sample in the dataset (rest of function remains similar)

    for sample in dataset:

        # Create image record
        if hasattr(sample, "hasty_image_id"):
            image_id = sample.hasty_image_id

            orgininal_image = [i for i in hA_template.images if i.image_id == image_id][0]
        else:
            # Generate a consistent image ID if none exists
            image_id = str(uuid.uuid4())

        image_filename = sample.filename if hasattr(sample, "filename") else os.path.basename(sample.filepath)

        # Get image dimensions
        if hasattr(sample, "metadata"):
            width = sample.metadata.width
            height = sample.metadata.height
        else:
            # You might need to read the image to get dimensions if not in metadata
            try:
                img = Image.open(sample.filepath)
                width, height = img.size
                img.close()
            except:
                logger.warning(f"Could not determine dimensions for {image_filename}, using defaults")
                width, height = 1000, 1000  # Default values

        # Create image labels collection
        image_labels = []

        # Process bounding box annotations if present
        bbox_field = anno_field if anno_field else "ground_truth_boxes"
        if hasattr(sample, bbox_field):
            bbox_detections = getattr(sample, bbox_field)
            for det in bbox_detections.detections if bbox_detections else []:
                if det.label != bbox_field and bbox_field != "any":
                    continue

                # Convert normalized coordinates to pixel values
                x1, y1, w, h = det.bounding_box
                x2 = x1 + w
                y2 = y1 + h
                x1 *= width
                x2 *= width
                y1 *= height
                y2 *= height

                bbox = [int(x1), int(y1), int(x2), int(y2)]

                # Create a label with bbox
                label_id = det.hasty_id if hasattr(det, "hasty_id") else str(uuid.uuid4())

                label = ImageLabel(
                    id=label_id,
                    class_name=det.label,
                    bbox=bbox,
                    polygon=None,
                    mask=None,
                    z_index=0,
                    keypoints=[],
                    attributes={"cvat": "created"}
                )

                image_labels.append(label)

        # Process keypoint annotations if present
        if hasattr(sample, point_field):
            keypoints_field = getattr(sample, point_field)

            for kp in keypoints_field.keypoints:

                # Convert normalized coordinates to pixel values
                x = kp.points[0][0] * width
                y = kp.points[0][1] * height

                hasty_keypoint = Keypoint(
                    x=int(x),
                    y=int(y),
                    norder=0,
                    keypoint_class_id=keypoint_class_id
                )

                # Create a label with keypoint
                label_id = kp.hasty_id if hasattr(kp, "hasty_id") else str(uuid.uuid4())

                try:
                    original_label = [l for l in orgininal_image.labels if l.id == label_id][0]
                    # TODO update the class
                    original_label.class_name = kp.label
                    original_label.keypoints = [hasty_keypoint]
                    original_label.attributes["cvat"] = "created"
                    image_labels.append(original_label)

                except IndexError:
                    label = ImageLabel(
                        id=label_id,
                        class_name=kp.label,
                        keypoints=[hasty_keypoint],
                        attributes={"cvat": "created"}
                    )
                    image_labels.append(label)

        # Create AnnotatedImage object
        dataset_name = sample.dataset_name if hasattr(sample, "dataset_name") else None

        annotated_image = AnnotatedImage(
            image_id=image_id,
            image_name=image_filename,
            dataset_name=dataset_name,
            ds_image_name=None,
            width=width,
            height=height,
            image_status="Done",
            tags=[],
            image_mode=None,
            labels=image_labels
        )

        # Add the annotated image to our collection
        hasty_annotation.images.append(annotated_image)

    logger.info(f"Created Hasty annotation with {len(hasty_annotation.images)} images " +
                f"and {hasty_annotation.label_count()} labels")

    return hasty_annotation


def determine_changes(
    hA_reference: HastyAnnotationV2,
    hA_reference_updated: HastyAnnotationV2,
    class_name="iguana"):
    """
    Compare the original and updated Hasty annotations to determine changes.
    :param hA_reference:
    :param hA_updated:
    :param class_name:
    :return:
    """

    df_old = hA_reference.get_flat_df()
    df_new = hA_reference_updated.get_flat_df()

    rows_not_in_df2 = df_new[~df_new['label_id'].isin(df_old['label_id'])]

    logger.info(f"There were {len(df_old)} labels in the original annotation and {len(df_new)} in the updated annotation")

    for i in hA_reference_updated.images:
        updated_image = hA_reference_updated.get_image_by_id(i.image_id)
        reference_image = hA_reference.get_image_by_id(i.image_id)
        diff = set(updated_image.labels) - set(reference_image.labels)
        if len(diff) > 0:

            updated_image.labels

        logger.warning(f"Continue the implementation of this")



def cvat2hasty_v2(hA_tiled_prediction: HastyAnnotationV2,
               dataset_name,
                  point_field="detection",
                  class_name = "iguana") -> HastyAnnotationV2:
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