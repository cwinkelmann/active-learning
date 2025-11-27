import json
import numpy as np
import shutil
import typing
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import shapely
from loguru import logger
from shapely.geometry.point import Point
try:
    from ultralytics.data.converter import convert_coco
except ImportError:
    logger.warning("ultralytics not installed, coco2yolo will not work")

from active_learning.config.mapping import keypoint_id_mapping
from active_learning.util.image import get_image_id
from active_learning.util.projection import project_gdfcrs, convert_gdf_to_jpeg_coords, get_geotransform, \
    get_orthomosaic_crs, pixel_to_world_point
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




def dota2coco(input_data):
    # Parse input data
    lines = input_data.strip().split('\n')

    # Extract metadata
    metadata = {}
    for i, line in enumerate(lines):
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key] = value
            continue
        else:
            annotation_start = i
            break

    # Initialize COCO format dictionary
    coco_format = {
        "info": {
            "description": "Converted from custom annotation format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [
            {
                "id": 1,
                "width": 10000,  # Placeholder, adjust based on your image
                "height": 10000,  # Placeholder, adjust based on your image
                "file_name": "image.jpg",  # Placeholder
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().strftime("%Y-%m-%d")
            }
        ],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "small-vehicle",
                "supercategory": "vehicle"
            }
        ]
    }

    # Add source information if available
    if "imagesource" in metadata:
        coco_format["images"][0]["source"] = metadata["imagesource"]

    # Add GSD information if available
    if "gsd" in metadata:
        coco_format["images"][0]["gsd"] = float(metadata["gsd"])

    # Process annotations
    annotation_id = 1
    category_ids = {"small-vehicle": 1}  # Map category names to IDs

    for i in range(annotation_start, len(lines)):
        parts = lines[i].strip().split()

        if len(parts) < 11:  # Need at least 8 coordinates + category + score
            continue

        # Extract polygon coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
        polygon = [
            float(parts[0]), float(parts[1]),  # x1, y1
            float(parts[2]), float(parts[3]),  # x2, y2
            float(parts[4]), float(parts[5]),  # x3, y3
            float(parts[6]), float(parts[7])  # x4, y4
        ]

        # Calculate segmentation (format required by COCO)
        segmentation = [polygon]

        # Extract category name and score
        category_name = parts[8]
        score = float(parts[9])

        # Add category to categories if not exists
        if category_name not in category_ids:
            category_id = len(category_ids) + 1
            category_ids[category_name] = category_id
            coco_format["categories"].append({
                "id": category_id,
                "name": category_name,
                "supercategory": "object"
            })

        # Calculate bounding box [x, y, width, height]
        x_coords = [polygon[0], polygon[2], polygon[4], polygon[6]]
        y_coords = [polygon[1], polygon[3], polygon[5], polygon[7]]
        x_min, y_min = min(x_coords), min(y_coords)
        width, height = max(x_coords) - x_min, max(y_coords) - y_min

        # Calculate polygon area
        # For a simple approximation, we can use the shoelace formula
        area = 0
        n = 4  # Number of vertices
        for i in range(n):
            j = (i + 1) % n
            area += x_coords[i] * y_coords[j]
            area -= y_coords[i] * x_coords[j]
        area = abs(area) / 2

        # Create annotation
        annotation = {
            "id": annotation_id,
            "image_id": 1,
            "category_id": category_ids[category_name],
            "segmentation": segmentation,
            "area": area,
            "bbox": [x_min, y_min, width, height],
            "iscrowd": 0,
            "score": score
        }

        coco_format["annotations"].append(annotation)
        annotation_id += 1

    return coco_format


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


def herdnet_prediction_to_hasty(df_prediction: pd.DataFrame,
                                images_path: Path,
                                dataset_name: str | None = None,
                                hA_reference: typing.Optional[HastyAnnotationV2] = None,
                                ) -> typing.List[AnnotatedImage]:
    assert "images" in df_prediction.columns, "images column not found in the DataFrame"
    # assert labels
    assert "labels" in df_prediction.columns, "labels, integer 1,2... column not found in the DataFrame"
    assert "scores" in df_prediction.columns, "scores column not found in the DataFrame"
    assert "x" in df_prediction.columns, "x column not found in the DataFrame"
    assert "y" in df_prediction.columns, "y column not found in the DataFrame"
    assert "species" in df_prediction.columns, "species str, i.e. iguana column not found in the DataFrame"

    ILC_list: typing.List[ImageLabelCollection] = []

    for image_name, df_group in df_prediction.groupby("images"):
        image_name = str(image_name)
        w, h = get_image_dimensions(image_path= images_path / image_name)

        annotations: typing.List[PredictedImageLabel, ImageLabel] = []
        # Iterate over DataFrame rows
        for _, row in df_group.iterrows():
            if row.x is None or row.y is None or np.isnan(row.x) or np.isnan(row.y) :
                logger.warning(f"Skipping row with missing x or y coordinates for image {image_name}")
                continue

            try:
                annotation = PredictedImageLabel(
                    score=float(row["scores"]),
                    class_name=row["species"],  # use species as the label/class_name
                    bbox=None,   # if not available, keep as None
                    polygon=None,
                    mask=[],     # or adjust if you have mask data
                    keypoints = [Keypoint(
                        x=int(row.x),
                        y=int(row.y),
                        keypoint_class_id=keypoint_id_mapping.get(row.species.lower(), keypoint_id_mapping.get("body")),
                    )],  # you can store extra information if needed
                    kind=row.get("kind", None)
                )
            except Exception as e:
                # logger.info(f"Error creating PredictedImageLabel image {image_name}, {e}")
                annotation = ImageLabel(
                    class_name=row["species"],  # use species as the label/class_name
                    bbox=None,  # if not available, keep as None
                    polygon=None,
                    mask=[],  # or adjust if you have mask data
                    keypoints=[Keypoint(
                        x=int(row.x),
                        y=int(row.y),
                        keypoint_class_id=keypoint_id_mapping.get(row.species.lower(), keypoint_id_mapping.get("body")),
                    )],  # you can store extra information if needed

                )
            annotations.append(annotation)

        if hA_reference is not None:
            ilC = hA_reference.get_image_by_name(image_name, dataset_name=dataset_name)
            # ilC.labels.extend(annotations)
            # get the image from the reference

            ilC = AnnotatedImage(
                image_id=ilC.image_id,
                image_name=str(image_name),
                labels=annotations,
                width=w,
                height=h,
                dataset_name=dataset_name,
            )
        else:
            ilC = AnnotatedImage(
            image_name=str(image_name),
            labels=annotations,
            width=w,
            height=h,
                dataset_name=dataset_name,
        )
        ILC_list.append( ilC )

    return ILC_list

def _is_with_edge(x: int, y: int, width: int, height: int, edge_threshold: int = 40):
    return x < edge_threshold or x > (width - edge_threshold) or y < edge_threshold or y > (height - edge_threshold)

def hasty_to_shp(tif_path: Path,
                 hA_reference: HastyAnnotationV2,
                 edge_threshold: int = 50,
                 suffix=".tif"):
    """
    Convert Hasty Annotation to a GeoDataFrame with geometries in geospatial coordinates.
    :param tif_path: 
    :param hA_reference: 
    :param suffix: 
    :return: 
    """
    # TODO look at this: convert_jpeg_to_geotiff_coords from playground/052_shp2other.py
    # convert_jpeg_to_geotiff_coords()

    assert tif_path is not None, "tif_path is None"

    data = []
    if len(hA_reference.images) == 0:
        raise ValueError("No images in Hasty Annotation")

    # Mark objects on the edge
    for img in hA_reference.images:
        img_name = Path(img.image_name).with_suffix(suffix=suffix)
        geo_transform = get_geotransform(tif_path / img_name)
        crs = get_orthomosaic_crs(tif_path / img_name)

        for label in img.labels:

            # get the pixel coordinates
            x, y = label.incenter_centroid.x, label.incenter_centroid.y
            # get the world coordinates
            p = pixel_to_world_point(geo_transform, x, y)
            # set the new coordinates
            # TODO add some more metadata
            if isinstance(label, PredictedImageLabel):
                score = label.score
            else:
                score = None
            data.append({
                "img_name": img_name,
                "img_id": img.image_id,
                "label": label.class_name,
                "label_id": label.id,
                "score": score,
                "geometry": p,
                "on_edge": _is_with_edge(x, y, img.width, img.height, edge_threshold=edge_threshold),
                "local_x": int(x),
                "local_y": int(y),
            })

    return gpd.GeoDataFrame(data, crs=crs)

def ifa_point_shapefile_to_hasty(gdf: gpd.GeoDataFrame, images_path: Path,
                       labels=1, species="iguana") -> ImageLabelCollection:
    """
    create a Hasty ImageLabelCollection from a shapefile, which is very specific based on the iguanasFromAbove workflow which creates dots without id/name which depict iguanas
    :param gdf:
    :param images_path:
    :return:
    """
    logger.info(f"convert shapefile to Hasty ImageLabelCollection, {images_path.name}")
    gdf["images"] = images_path.name
    gdf = project_gdfcrs(gdf, images_path)

    # project the global coordinates to the local coordinates of the orthomosaic
    gdf_local = convert_gdf_to_jpeg_coords(gdf, images_path)

    # rename local_x to x, local_y to y
    gdf_local.rename(columns={"local_x": "x", "local_y": "y"}, inplace=True)
    gdf_local["scores"] = 0
    gdf_local["labels"] = labels
    gdf_local["species"] = species

    # create an ImageCollection of the annotations
    ilc = herdnet_prediction_to_hasty(gdf_local, images_path=images_path.parent)

    return ilc

def prediction_list_to_gdf(predictions: typing.List[PredictedImageLabel], crs="EPSG:32715"):
    """
    Convert a list of predictions to a GeoDataFrame.
    :param predictions:
    :return:
    """
    data = []
    for prediction in predictions:
        # Append to the data list
        data.append({
            "label": prediction.class_name,  # Assuming there is a label attribute
            "confidence": prediction.score,  # Assuming there is a confidence score
            "geometry": Point(prediction.incenter_centroid.x, prediction.incenter_centroid.y)  # Create Point geometry
        })

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry="geometry", crs=crs )
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
