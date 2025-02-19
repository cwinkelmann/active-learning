import json
import shutil
import typing
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import shapely
from loguru import logger
from shapely.geometry.point import Point
from ultralytics.data.converter import convert_coco

from active_learning.config.mapping import keypoint_id_mapping
from com.biospheredata.converter.HastyConverter import get_image_dimensions
from com.biospheredata.types.COCOAnnotation import COCOAnnotations, Image, Category, Annotation
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, LabelClass, ImageLabel, AnnotatedImage, \
    PredictedImageLabel, Keypoint, ImageLabelCollection
import geopandas as gpd
import fiftyone as fo

def hasty2coco(hA: HastyAnnotationV2) -> COCOAnnotations:
    # 1. Create a map for images (Hasty image_id -> COCO image_id)
    #    We'll just enumerate the images
    image_id_map = {}
    coco_images = []
    for i, img in enumerate(hA.images, start=1):
        image_id_map[str(img.image_id)] = i
        coco_images.append(Image(
            id=i,
            width=img.width,
            height=img.height,
            file_name=img.image_name,
            license=None,
            coco_url=None,
            date_captured=None
        ))

    # 2. Create categories
    #    We'll map Hasty's label_classes to COCO categories
    coco_categories = []
    category_name_to_id = {}
    for i, cat in enumerate(hA.label_classes, start=1):
        category_name_to_id[cat.class_name] = i
        coco_categories.append(Category(
            id=i,
            name=cat.class_name,
            supercategory=cat.class_type if cat.class_type else None
        ))

    # 3. Create annotations
    #    We'll assign a new annotation id for each label
    coco_annotations = []
    ann_id = 1
    for img in hA.images:
        image_id_int = image_id_map[str(img.image_id)]
        for label in img.labels:
            # Determine the category_id from the label's class_name
            if label.class_name not in category_name_to_id:
                # Skip if we don't have a matching category
                continue
            cat_id = category_name_to_id[label.class_name]

            # Handle bbox
            # Hasty bbox is [x1, y1, x2, y2], COCO bbox should be [x, y, width, height]
            if label.bbox is not None:
                x1, y1, x2, y2 = label.bbox
                coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            elif label.polygon is not None:
                # Compute bbox from polygon
                poly = shapely.Polygon(label.polygon)
                x1, y1, x2, y2 = poly.bounds
                coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            else:
                coco_bbox = None

            # Handle polygon (if any)
            # COCO segmentation is a list of polygons. Each polygon is a list of [x, y, x, y, ...]
            # If polygon is present, flatten it into a single list
            coco_segmentation = None
            if label.polygon is not None:
                # polygon should be a list of [x, y]
                # Flatten it: [x1,y1,x2,y2,...]
                seg_coords = []
                for (px, py) in label.polygon:
                    seg_coords.extend([float(px), float(py)])
                coco_segmentation = [seg_coords]

            # Compute area
            area = 0.0
            if label.polygon is not None:
                poly = shapely.Polygon(label.polygon)
                area = float(poly.area)
            elif label.bbox is not None:
                width = x2 - x1
                height = y2 - y1
                area = float(width * height)

            # iscrowd - default to 0 unless specified
            iscrowd = 0

            coco_annotations.append(Annotation(
                id=ann_id,
                image_id=image_id_int,
                category_id=cat_id,
                segmentation=coco_segmentation,
                area=area,
                bbox=coco_bbox,
                iscrowd=iscrowd
            ))
            ann_id += 1

    coco_output = COCOAnnotations(
        images=coco_images,
        annotations=coco_annotations,
        categories=coco_categories,
        licenses=None,
        info=None
    )

    return coco_output


def coco2yolo(
                yolo_path: Path,
              source_dir: Path,
              coco_annotations: Path,
              images: typing.List[Path]) -> Path:
    """ Convert COCO annotations to YOLO format and copy images to output directory"""
    # TODO remove dependency to ultralytics
    temp_dir = source_dir / "yolo_tmp"

    assert coco_annotations.exists(), f"{coco_annotations} does not exist"
    # TODO this ultralytics COCO converter is crap
    convert_coco(
        labels_dir=coco_annotations.parent,
        save_dir=temp_dir,
    )

    for i, l in enumerate(temp_dir.joinpath("labels").joinpath("coco_format").glob("*.txt")):
        shutil.move(l, yolo_path / l.name)
    if i == 0:
        raise ValueError(f"No labels found in {temp_dir}")

    for img in images:
        shutil.copy(img, yolo_path / img.name)

    shutil.rmtree(temp_dir)

    return source_dir


def AED2COCO(aed_path: Path,
              source_dir: Path,
              images: typing.List[Path]) -> Path:
    """convert the African Elephants dataset to COCO format
    TODO, this would only work if annotations are converted to boxes by assuming a box size
    """

    raise NotImplementedError("This function is not implemented yet")


def iSAID2Hasty(isaid_path: Path) -> Path:
    """convert the iSAID dataset to COCO format
    TODO, this would only work if annotations are converted to boxes by assuming a box size
    """
    with open(isaid_path, "r") as f:
        isaid = json.load(f)
    pass


def coco2hasty(coco_data: typing.Dict, images_path: Path, project_name="coco_conversion", dataset_name = None) -> HastyAnnotationV2:
    """
        Convert COCO dataset to HastyAnnotationV2 format.

        Args:
            coco_path (Path): Path to the COCO JSON file.
            output_path (Path): Path to save the converted HastyAnnotationV2 JSON file.
            project_name (str): Name of the Hasty project.
        """

    # Extract COCO data
    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = {cat["id"]: cat for cat in coco_data["categories"]}

    # Create LabelClass list for HastyAnnotation
    label_classes = [
        LabelClass(
            class_id=str(uuid.uuid4()),
            parent_class_id=None,
            class_name=category["name"],
            class_type="object",  # or "segmentation" if needed
            color="#FFFFFF",  # Assign default or meaningful colors
            norder=index,
            attributes=[],
            description=category.get("description", None),
        )
        for index, category in enumerate(categories.values())
    ]

    # Map annotations to images
    annotations_by_image = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Create AnnotatedImage list for HastyAnnotation
    hasty_images = []
    for image in images:
        image_annotations = annotations_by_image.get(image["id"], [])
        labels = []

        for ann in image_annotations:
            label_id = ann.get("id", str(uuid.uuid4()))
            if len(ann.get("segmentation", None)) > 1:
                logger.warning(f"Warning: More than one segmentation found for image {image['file_name']} and annotation {label_id}, for each segmenation mask a separate label will be created and the box recalculated.")
            for segmentation_mask in ann.get("segmentation", None):
                coordinate_pairs = [(segmentation_mask[i], segmentation_mask[i + 1]) for i in range(0, len(segmentation_mask), 2)]

                shapely_polygon = shapely.geometry.Polygon(coordinate_pairs)
                new_box = [int(x) for x in shapely_polygon.bounds]
                labels.append(
                    ImageLabel(
                        id=label_id,
                        class_name=categories[ann["category_id"]]["name"],
                        bbox=new_box,
                        polygon=coordinate_pairs,
                        attributes={},  # Add attributes as needed
                    ))



        image_width, image_height = get_image_dimensions(images_path / image["file_name"])

        hasty_image = AnnotatedImage(
            image_id=image["id"],
            image_name=image["file_name"],
            dataset_name=dataset_name,
            width=image_width,
            height=image_height,
            labels=labels,
        )
        hasty_images.append(hasty_image)

    # Create HastyAnnotationV2 object
    hasty_annotation = HastyAnnotationV2(
        project_name=project_name,
        create_date=datetime.now(),
        export_format_version="1.1",
        export_date=datetime.now(),
        label_classes=label_classes,
        images=hasty_images,
    )

    return hasty_annotation


def herdnet_prediction_to_hasty(df_prediction: pd.DataFrame, images_path: Path) -> typing.List[ImageLabelCollection]:
    assert "images" in df_prediction.columns, "images column not found in the DataFrame"
    # assert labels
    assert "labels" in df_prediction.columns, "labels column not found in the DataFrame"
    assert "scores" in df_prediction.columns, "scores column not found in the DataFrame"
    assert "x" in df_prediction.columns, "x column not found in the DataFrame"
    assert "y" in df_prediction.columns, "y column not found in the DataFrame"
    assert "species" in df_prediction.columns, "species column not found in the DataFrame"

    ILC_list: typing.List[ImageLabelCollection] = []

    for image_name, df_group in df_prediction.groupby("images"):
        image_name = str(image_name)
        w, h = get_image_dimensions(image_path= images_path / image_name)

        annotations: typing.List[PredictedImageLabel] = []
        # Iterate over DataFrame rows
        for _, row in df_group.iterrows():

            if row["scores"] is None or pd.isna(row["scores"]):
                continue

            annotation = PredictedImageLabel(
                score=float(row["scores"]),
                class_name=row["species"],  # use spiecies as the label/class_name
                bbox=None,   # if not available, keep as None
                polygon=None,
                mask=[],     # or adjust if you have mask data
                keypoints = [Keypoint(
                    x=int(row.x),
                    y=int(row.y),
                    keypoint_class_id=keypoint_id_mapping.get(row.species, None),
                )],  # you can store extra information if needed
                kind=row.get("kind", None)
            )
            annotations.append(annotation)

        ILC_list.append( ImageLabelCollection(
            image_name=str(image_name),
            labels=annotations,
            width=w,
            height=h
        ))

    return ILC_list


def prediction_list_to_gdf(predictions):
    data = []
    for prediction in predictions:
        # Append to the data list
        data.append({
            "label": prediction.class_name,  # Assuming there is a label attribute
            "confidence": prediction.score,  # Assuming there is a confidence score
            "geometry": Point(prediction.incenter_centroid.x, prediction.incenter_centroid.y)  # Create Point geometry
        })

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:32715")  # Assuming EPSG:32715 projection
    return gdf

def _create_keypoints_s(hA_image: typing.Union[AnnotatedImage, ImageLabelCollection]) -> typing.List[fo.Keypoint]:
    """

    :param hA_image:
    :return:
    """
    keypoints = []
    w, h = hA_image.width, hA_image.height

    for r in hA_image.labels:

        pt = (int(r.incenter_centroid.x) / w, int(r.incenter_centroid.y) / h)
        lab = r.class_name

        # TODO we need to add the attributes to the label, i.e. if the point is a head or a tail
        if isinstance(r, PredictedImageLabel):
            kp = fo.Keypoint(
            # kind="str",
            hasty_id=r.id,
            label=str(lab),
            points=[pt],
            confidence=[r.score]
        )
        else:
            kp = fo.Keypoint(
            # kind="str",
            hasty_id=r.id,
            label=str(lab),
            points=[pt]

        )

        keypoints.append(kp)
    return keypoints


def _create_fake_boxes(hA_image: typing.Union[AnnotatedImage, ImageLabelCollection]) -> typing.List[fo.Detection]:
    """

    :param hA_image:
    :return:
    """
    offset = 80
    keypoints = []
    w, h = hA_image.width, hA_image.height

    for r in hA_image.labels:
        x1 = int(r.incenter_centroid.x) - offset
        x2 = int(r.incenter_centroid.x) + offset
        y1 = int(r.incenter_centroid.y) - offset
        y2 = int(r.incenter_centroid.y) + offset
        box_w, box_h = (x2 - x1) / w, (y2 - y1) / h
        x1 = x1 / w
        y1 = y1 / h
        pt = (int(r.incenter_centroid.x) / w, int(r.incenter_centroid.y) / h)
        lab = r.class_name

        # TODO we need to add the attributes to the label, i.e. if the point is a head or a tail
        kp = fo.Detection(label=str(lab), bounding_box=[x1, y1, box_w, box_h],
            # kind="str",
            hasty_id=r.id,
            attributes=r.attributes,
            # attributes={"custom_attribute": [{"bla": "keks"}]},
            tags=[],
            # confidence = r.score
        )

        keypoints.append(kp)
    return keypoints


def _create_boxes_s(hA_image: typing.Union[AnnotatedImage, ImageLabelCollection]) -> typing.List[fo.Detection]:
    """

    :param hA_image:
    :return:
    """
    boxes = []
    w, h = hA_image.width, hA_image.height

    for r in hA_image.labels:

        x1, y1, x2, y2 = r.x1y1x2y2[0], r.x1y1x2y2[1], r.x1y1x2y2[2], r.x1y1x2y2[3]
        box_w, box_h = x2 - x1, y2 - y1
        box_w /= w
        box_h /= h

        x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
        lab = r.class_name


        kp = fo.Detection(label=str(lab), bounding_box=[x1, y1, box_w, box_h],
            # kind="str",
            hasty_id=r.id,
            attributes=r.attributes,
            # attributes={"custom_attribute": [{"bla": "keks"}]},
            tags=["bla", "keks"],
            confidence=0.95
        )

        boxes.append(kp)
    return boxes
