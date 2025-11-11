# config_path = Path("/Users/christian/data/2TB/ai-core/data/google_drive_mirror/Orthomosaics_for_quality_analysis/San_STJB01_10012023/batch_workflow_report_config_San_STJB01_10012023.yaml")
import os

import uuid

import typing

import copy

import shapely
from fiftyone.utils.cvat import CVATBackend
from loguru import logger
from pathlib import Path
import fiftyone as fo
import pandas as pd

from active_learning.types.Exceptions import NoChangesDetected
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


import fiftyone as fo
from datetime import datetime
import logging

import fiftyone as fo
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


def get_cvat_annotation_status(dataset: fo.Dataset, anno_key: str) -> dict:
    """
    Fetch CVAT annotation status including job assignee, stage, and state.

    Args:
        dataset: FiftyOne dataset
        anno_key: Annotation run key

    Returns:
        dict: Comprehensive annotation metadata including CVAT job statuses
    """

    if anno_key not in dataset.list_annotation_runs():
        logger.error(f"Annotation run '{anno_key}' not found in {dataset.list_annotation_runs()}")
        return None

    try:
        anno_info = dataset.get_annotation_info(anno_key)
        view = dataset.load_annotation_view(anno_key)
        results = dataset.load_annotation_results(anno_key)

        # Build basic status
        status = {
            'anno_key': anno_key,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(dataset),
            'annotated_samples': len(view),
            'backend': type(results.backend).__name__ if hasattr(results, 'backend') else 'unknown',
        }

        # Get CVAT-specific information
        if hasattr(anno_info, 'config'):
            config = anno_info.config
            status['cvat_info'] = {
                'organization': config.get('organization'),
                'project_name': config.get('project_name'),
                'label_field': config.get('label_field'),
            }

        # Get CVAT task and job information
        if hasattr(results, 'backend') and isinstance(results.backend, CVATBackend):
            backend = results.backend

            # Try different ways to access the API client
            api_client = None

            # Method 1: Direct attribute
            if hasattr(backend, 'api'):
                api_client = backend.api
            # Method 2: Through _api attribute
            elif hasattr(backend, '_api'):
                api_client = backend._api
            # Method 3: Through config
            elif hasattr(backend, 'config') and hasattr(backend.config, 'api'):
                api_client = backend.config.api
            # Method 4: Connect using config
            elif hasattr(backend, 'config'):
                try:
                    api_client = backend.connect_to_api()
                except:
                    pass

            if api_client is None:
                logger.error("Could not access CVAT API client")
                logger.error(f"Backend attributes: {dir(backend)}")
                status['is_complete'] = None
                return status

            cvat_tasks = []
            all_jobs_complete = True

            # Get task IDs
            task_ids = results.task_ids if hasattr(results, 'task_ids') else []

            if not task_ids:
                logger.warning(f"No task IDs found for annotation run '{anno_key}'")
                status['is_complete'] = None
                return status

            for task_id in task_ids:
                try:
                    # Fetch task from CVAT API
                    task = api_client.tasks.retrieve(task_id)

                    task_info = {
                        'task_id': task_id,
                        'name': task.name,
                        'status': task.status,
                        'owner': task.owner.username if hasattr(task, 'owner') and hasattr(task.owner,
                                                                                           'username') else None,
                        'jobs': []
                    }

                    # Fetch jobs for this task
                    jobs_response = api_client.jobs.list(task_id=task_id)

                    # Handle different response types
                    jobs = jobs_response.results if hasattr(jobs_response, 'results') else jobs_response

                    for job in jobs:
                        job_info = {
                            'job_id': job.id,
                            'assignee': job.assignee.username if hasattr(job, 'assignee') and job.assignee else None,
                            'stage': job.stage if hasattr(job, 'stage') else None,
                            'state': job.state if hasattr(job, 'state') else None,
                            'start_frame': job.start_frame if hasattr(job, 'start_frame') else None,
                            'stop_frame': job.stop_frame if hasattr(job, 'stop_frame') else None,
                        }

                        # Check if job is complete
                        if job_info['state'] != 'completed':
                            all_jobs_complete = False

                        task_info['jobs'].append(job_info)

                    cvat_tasks.append(task_info)

                except Exception as e:
                    logger.error(f"Failed to fetch CVAT task {task_id}: {type(e).__name__}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

            status['cvat_tasks'] = cvat_tasks
            status['is_complete'] = all_jobs_complete
            status['total_jobs'] = sum(len(task['jobs']) for task in cvat_tasks)
            status['completed_jobs'] = sum(
                1 for task in cvat_tasks
                for job in task['jobs']
                if job['state'] == 'completed'
            )

            # Log status
            completion_status = "COMPLETE" if status['is_complete'] else "IN PROGRESS"
            logger.info(
                f"Status for '{anno_key}': {completion_status} - {status['completed_jobs']}/{status['total_jobs']} jobs complete")
        else:
            status['is_complete'] = None
            logger.warning(
                f"Backend is not CVATBackend (is {type(results.backend).__name__}), cannot determine job completion status")
            logger.info(f"Status for '{anno_key}': {status['annotated_samples']}/{status['total_samples']} samples")

        return status

    except Exception as e:
        logger.error(f"Failed to fetch annotation status for '{anno_key}': {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def is_annotation_complete(dataset: fo.Dataset, anno_key: str) -> bool:
    """
    Check if an annotation run is complete based on CVAT job states.

    Args:
        dataset: FiftyOne dataset
        anno_key: Annotation run key

    Returns:
        bool: True if all jobs are completed, False otherwise, None if cannot determine
    """
    status = get_cvat_annotation_status(dataset, anno_key)

    if not status:
        return None

    is_complete = status.get('is_complete')

    if is_complete is None:
        logger.warning(f"Cannot determine completion status for '{anno_key}' (non-CVAT backend?)")
        return None

    if is_complete:
        logger.info(f"✓ Annotation '{anno_key}' is COMPLETE ({status['completed_jobs']}/{status['total_jobs']} jobs)")
    else:
        logger.info(
            f"⏳ Annotation '{anno_key}' is IN PROGRESS ({status['completed_jobs']}/{status['total_jobs']} jobs)")

        # Log job details
        if 'cvat_tasks' in status:
            for task in status['cvat_tasks']:
                for job in task['jobs']:
                    if job['state'] != 'completed':
                        logger.info(
                            f"  Job {job['job_id']}: {job['state']} ({job['stage']}) - Assignee: {job['assignee']}")

    return is_complete


def get_job_details(dataset: fo.Dataset, anno_key: str) -> list:
    """
    Get detailed job information for an annotation run.

    Args:
        dataset: FiftyOne dataset
        anno_key: Annotation run key

    Returns:
        list: List of job details with assignee, stage, and state
    """
    status = get_cvat_annotation_status(dataset, anno_key)

    if not status or 'cvat_tasks' not in status:
        logger.error(f"Cannot get job details for '{anno_key}'")
        return []

    all_jobs = []
    for task in status['cvat_tasks']:
        for job in task['jobs']:
            all_jobs.append({
                'task_id': task['task_id'],
                'task_name': task['name'],
                'job_id': job['job_id'],
                'assignee': job['assignee'],
                'stage': job['stage'],
                'state': job['state'],
                'frame_range': f"{job['start_frame']}-{job['stop_frame']}"
            })

    return all_jobs








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

            if image_id == "ec1eb174-6979-485b-b0c9-aecef26c3a47":
                pass

            try:
                orgininal_image = [i for i in hA_template.images if i.image_id == image_id][0]
            except IndexError:
                orgininal_image = [i for i in hA_template.images if i.image_name == sample.hasty_image_name][0]

                pass
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
            image_status="DONE",
            tags=[],
            image_mode=None,
            labels=image_labels
        )

        # Add the annotated image to our collection
        hasty_annotation.images.append(annotated_image)

    logger.info(f"Created Hasty annotation with {len(hasty_annotation.images)} images " +
                f"and {hasty_annotation.label_count()} labels")

    return hasty_annotation



import pandas as pd
from shapely.geometry import Point
from shapely import wkt


def check_moved_points(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 0.0) -> pd.DataFrame:
    """
    Join two annotation dataframes by label_id and check if points have been moved.

    Args:
        df1: First dataframe (e.g., original annotations)
        df2: Second dataframe (e.g., updated annotations)
        tolerance: Distance tolerance in pixels (default: 0.0 for exact match)

    Returns:
        DataFrame with moved points and their original/new locations
    """

    # Join on label_id
    merged = df1.merge(
        df2,
        on='label_id',
        how='inner',
        suffixes=('_original', '_updated')
    )

    if len(merged) == 0:
        print("No matching label_ids found between dataframes")
        return pd.DataFrame()

    print(f"Found {len(merged)} matching label_ids")


    # Calculate if points moved
    def points_differ(row):
        orig = row['centroid_original']
        updated = row['centroid_updated']

        if orig is None or updated is None:
            return pd.DataFrame()

        # Calculate Euclidean distance
        distance = orig.distance(updated)

        return distance > tolerance

    def calculate_distance(row):
        orig = row['centroid_original']
        updated = row['centroid_updated']

        if orig is None or updated is None:
            return None

        distance = orig.distance(updated)
        return distance

    merged['moved'] = merged.apply(points_differ, axis=1)
    merged['distance_pixels'] = merged.apply(calculate_distance, axis=1)

    # Filter to only moved points
    moved_points = merged[merged['moved'] == True].copy()

    if len(moved_points) == 0:
        logger.info("✓ No points were moved")
        return pd.DataFrame()

    logger.info(f"⚠ Found {len(moved_points)} moved points")

    # Select relevant columns for output
    result = moved_points[[
        'label_id',
        'unique_ID_original',
        'image_name_original',
        'image_name_updated',
        'centroid_original',
        'centroid_updated',
        'distance_pixels',
        'attribute_original',
        'attribute_updated'
    ]].copy()

    return result





def find_added_and_removed_annotations(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Find annotations that were added or removed between two dataframes.

    Args:
        df1: Original dataframe
        df2: Updated dataframe

    Returns:
        tuple: (added_df, removed_df)
    """

    label_ids_1 = set(df1['label_id'].dropna())
    label_ids_2 = set(df2['label_id'].dropna())

    # Removed: in df1 but not in df2
    removed_ids = label_ids_1 - label_ids_2
    removed_df = df1[df1['label_id'].isin(removed_ids)].copy()

    # Added: in df2 but not in df1
    added_ids = label_ids_2 - label_ids_1
    added_df = df2[df2['label_id'].isin(added_ids)].copy()

    print(f"Removed annotations: {len(removed_df)}")
    print(f"Added annotations: {len(added_df)}")

    return added_df, removed_df





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
    changes = {}
    df_old = hA_reference.get_flat_df()
    df_new = hA_reference_updated.get_flat_df()

    old_label_ids = set(df_old.label_id)

    try:
        new_label_ids = set(df_new.label_id)
        moved_points = check_moved_points(df_old, df_new, tolerance=0)
    except:
        new_label_ids = set()
        moved_points = set()


    if len(moved_points) == 0:
        raise NoChangesDetected(f"No changes detected, probably the task is not edited yet.")

    # new labels
    rows_not_in_df2 = df_new[~df_new['label_id'].isin(df_old['label_id'])]
    # deleted labels, where label is does not exist in df_new
    rows_not_in_df_new = df_old[~df_old['label_id'].isin(df_new['label_id'])]

    changes["deleted_annotatations"] = len(rows_not_in_df_new)
    changes["added_annotations"] = len(rows_not_in_df2)
    changes["total_changes"] = len(rows_not_in_df_new) + len(rows_not_in_df2)
    changes["old_objects"] = len(old_label_ids)
    changes["new_objects"] = len(new_label_ids)



    total_diff = 0
    logger.info(f"There were {len(df_old)} labels in the original annotation and {len(df_new)} in the updated annotation")
# delplanque_train___15ef3a616405520ff8d4515f1b9f8a52f44d402e.JPG_1fd041e7-2214-4720-9039-4c4c6715b7fc.jpg
    for i in hA_reference_updated.images:
        updated_image = hA_reference_updated.get_image_by_id(i.image_id)
        reference_image = hA_reference.get_image_by_id(i.image_id)
        diff = set(updated_image.labels) - set(reference_image.labels)
        if len(diff) > 0:
            total_diff += len(diff)
            # updated_image.labels
            # TODO implement me
        # logger.warning(f"Continue the implementation of this")

    return changes

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