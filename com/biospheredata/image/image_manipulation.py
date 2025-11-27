# """ from flight simulation sim"""
# import copy
# import typing
# import uuid
#
# import shutil
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import PIL.Image
# import shapely
# from loguru import logger
# from typing import List, Union, Tuple
#
# from shapely import Polygon, affinity, box
#
# from com.biospheredata.converter.Annotation import add_offset_to_box
# from com.biospheredata.types.HastyAnnotationV2 import ImageLabel
# from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage
# from com.biospheredata.types.annotationbox import BboxXYXY, BboxXYWH, xywh2xyxy
# from com.biospheredata.visualization.visualize_result import blackout_bbox
# from PIL import Image
#
# DATA_SET_NAME = "train_augmented"
#
#
# def crop_out_individual_object(i: AnnotatedImage,
#                                full_images_path,
#                                output_path,
#                                offset: int = None,
#                                width: int = None,
#                                height: int = None) -> List[Path]:
#     """
#     create crops from individual iguanas from each image
#     :param df_flat:
#     :param full_images_path:
#     :param output_path:
#     :return:
#     """
#     assert isinstance(i, AnnotatedImage)
#     im = PIL.Image.open(full_images_path / i.dataset_name / i.image_name)
#
#     images = []
#     crops = []
#     for label in i.labels:
#
#         ## TODO this is just polygon creation
#         # add a bit of an offset
#         if offset:
#             box = add_offset_to_box(label.bbox, i.height, i.width, offset)
#
#         # constant boundary around the box
#         elif width and height:
#             box = resize_box(label.bbox_polygon, width, height)
#
#             if box.bounds[0] < 0:
#                 box = affinity.translate(box, -box.bounds[0], 0)
#             if box.bounds[1] < 0:
#                 box = affinity.translate(box, 0, -box.bounds[1])
#
#             if box.bounds[2] > i.width:
#                 box = affinity.translate(box, i.width - box.bounds[2], 0)
#             if box.bounds[3] > i.height:
#                 box = affinity.translate(box, 0, i.height - box.bounds[3])
#
#             box = [int(x) for x in box.bounds]
#
#         else:
#             raise ValueError("offset or width and height must be provided")
#
#         ## # TODO the actual crop should move somewhere else
#         im1 = im.crop(box)
#         # im1.show()
#         im1 = im1.convert("RGB")
#         box_name = f"{Path(i.image_name).stem}__ID_{label.id}.jpg"
#         im1.save(output_path.joinpath(box_name))
#         images.append(im1)
#
#         crop_example_path = output_path.joinpath(box_name)
#         crops.append(crop_example_path)
#
#     return crops
#
#
# def resize_box(rectangle: Polygon, width, height) -> Polygon:
#     minx, miny, maxx, maxy = rectangle.bounds
#
#     resize_x = width - (maxx - minx)
#     resize_y = height - (maxy - miny)
#     new_rectangle = Polygon([
#         (int(round(minx - resize_x / 2)), int(round(miny - resize_y / 2))),
#         (int(round(maxx + resize_x / 2)), int(round(miny - resize_y / 2))),
#         (int(round(maxx + resize_x / 2)), int(round(maxy + resize_y / 2))),
#         (int(round(minx - resize_x / 2)), int(round(maxy + resize_y / 2)))
#     ])
#
#     return new_rectangle
#
#
# def crop_annotations_from_image_crops(crop_boxes: Union[BboxXYXY, BboxXYWH],
#                                       annotations: List[ImageLabel]):
#     """
#     crop out the annotations from the image
#     :param crop_boxes:
#     :param annotations:
#     :return:
#     """
#     if isinstance(crop_boxes, BboxXYWH):
#         crop_boxes = xywh2xyxy(crop_boxes)
#
#
#     # TODO finalise this code
#
#
# def crop_out_images(hi: AnnotatedImage,
#                     slice_height: int,
#                     slice_width: int,
#                     full_images_path: Path,
#                     output_path: Path,
#                     ) -> (List[Path], List[Path], List[AnnotatedImage]):
#     """ iterate through the image and crop out the tiles
#
#     :param output_path:
#     :param full_images_path:
#     :param slice_width:
#     :param slice_height:
#     :param hi:
#     """
#     counter = 0  ## TODO is this the right position?
#     images_with_objects = []
#     images_without_objects = []
#     labels_paths = []
#
#     output_path.mkdir(parents=True, exist_ok=True)
#     output_path.joinpath("object").mkdir(parents=True, exist_ok=True)
#     output_path.joinpath("empty").mkdir(parents=True, exist_ok=True)
#
#     image = PIL.Image.open(full_images_path / hi.dataset_name / hi.image_name)
#     # imr = np.array(image)
#
#     # Convert to string if you need a string representation
#     annotations = []
#
#     # slice the image in tiles
#     for height_i in range((hi.height // slice_height)):
#         for width_j in range((hi.width // slice_width)):
#             x1 = width_j * slice_width
#             y1 = hi.height - (height_i * slice_height)
#             x2 = ((width_j + 1) * slice_width)
#             y2 = (hi.height - (height_i + 1) * slice_height)
#
#             # cut the tile slice from the image
#
#             sliced = image.crop((x1, y2, x2, y1))  # TODO use a type for these?
#
#             # TODO save the coordinate of the slice to reconstruct it later
#
#             # sliding window of the image
#             pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
#             # TODO create boxes too
#             imsaved = False
#             slice_labels = []
#
#             ## TODO extract this to another function
#             empty = True
#             reject = False
#             max_bbox_area = 0
#             one_good_box = False
#
#             for box in hi.labels:
#                 # iterate through labels and check if any of the boxes intersects with the sliding window
#
#                 if pol.intersects(box.bbox_polygon):
#                     # any of the annotations is in the sliding window
#                     intersection_polygon = shapely.intersection(pol, box.bbox_polygon)
#                     max_bbox_area = max(max_bbox_area, intersection_polygon.area)
#                     empty = False
#
#                     if box.bbox_polygon.within(pol):
#                         one_good_box = True
#
#                         minx, miny, maxx, maxy = pol.bounds
#
#                         # Translate the coordinates of the inner polygon
#                         translated_coords = [(x - minx, y - miny) for x, y in box.bbox_polygon.exterior.coords]
#                         # Create a new polygon with the translated coordinates
#                         translated_inner_polygon = Polygon(translated_coords)
#                         il = ImageLabel(id=str(uuid.uuid4()),
#                                         image_id=str(uuid.uuid4()),
#                                         # category_id="888",  # TODO get this right
#                                         class_name="iguana",
#                                         bbox=[int(x) for x in translated_inner_polygon.bounds],
#                                         iscrowd=0, segmentation=[])
#                         slice_labels.append(il)
#
#                     else:
#                         # a part of the box is in the sliding window
#                         minx, miny, maxx, maxy = pol.bounds
#                         # Translate the coordinates of the inner polygon
#                         translated_coords = [(x - minx, y - miny) for x, y in box.bbox_polygon.exterior.coords]
#                         translated_inner_polygon = Polygon(translated_coords)
#
#                         sliced = blackout_bbox(sliced, bbox_x1y1x2y2=[int(x) for x in translated_inner_polygon.bounds])
#
#             if not empty:
#                 # print("not empty")
#                 pass
#
#             if empty == False and max_bbox_area < 5000:
#                 reject = True
#
#             # sliced_im = PIL.Image.fromarray(sliced)
#
#             filename = str(Path(hi.image_name).stem)
#
#             if not reject or one_good_box:
#                 if empty:
#                     slice_path_jpg = output_path / "empty" / f"{filename}_{height_i}_{width_j}.jpg"
#                     slice_path_tiff = output_path / "empty" / f"{filename}_{height_i}_{width_j}.tiff"
#                     slice_path_png = output_path / "empty" / f"{filename}_{height_i}_{width_j}.png"
#                     images_without_objects.append(slice_path_jpg)
#
#                 else:
#                     slice_path_jpg = output_path / "object" / f"{filename}_{height_i}_{width_j}.jpg"
#                     slice_path_tiff = output_path / "object" / f"{filename}_{height_i}_{width_j}.tiff"
#                     slice_path_png = output_path / "object" / f"{filename}_{height_i}_{width_j}.png"
#
#                     images_with_objects.append(slice_path_jpg)
#                     im = AnnotatedImage(
#                         image_id=str(uuid.uuid4()),
#                         dataset_name="train_augmented",
#                         image_name=f"{filename}_{height_i}_{width_j}.jpg",
#                         labels=slice_labels,
#                         width=slice_width,
#                         height=slice_height)
#                     annotations.append(im)
#
#                 # logger.info(f"slice_path: {slice_path_jpg}")
#                 sliced_im = sliced.convert("RGB")  # TODO I need this?
#                 sliced_im.save(slice_path_jpg, 'JPEG', quality=90, optimize=True)
#                 #sliced_im.save(slice_path_tiff, 'TIFF', compression=None)
#                 #sliced_im.save(slice_path_png )
#
#     return images_without_objects, images_with_objects, annotations
#
# def include_empty_frac(frac) -> bool:
#     if frac == False:
#         return frac
#     return np.random.rand() < frac
#
# def crop_out_images_v2(hi: AnnotatedImage,
#                        rasters: List[shapely.Polygon],
#                        full_image_path: Path,
#                        output_path: Path,
#                        dataset_name: str = DATA_SET_NAME,
#                        include_empty = False,
#                        edge_blackout = True) -> typing.Tuple[List[AnnotatedImage], List[Path]]:
#     """ iterate through rasters and crop out the tiles from the image return the new images and an annotations file
#
#     :param include_empty:
#     :param dataset_name:
#     :param full_image_path:
#     :param rasters:
#     :param output_path:
#     :param full_images_path:
#
#     :param hi:
#     """
#
#     images_with_objects = []
#
#     output_path.mkdir(parents=True, exist_ok=True)
#     output_path_images = output_path
#     output_path_images.mkdir(parents=True, exist_ok=True)
#
#     image = PIL.Image.open(full_image_path)
#     # imr = np.array(image)
#
#     # Convert to string if you need a string representation
#     images = []
#     image_paths: typing.List[Path] = []
#
#     # slice the image in tiles
#
#     # sliding window of the image
#     for pol in rasters:
#         assert isinstance(pol, shapely.Polygon)
#         sliced = image.crop(pol.bounds)
#         minx, miny, maxx, maxy = pol.bounds
#         slice_width = maxx - minx
#         slice_height = maxy - miny
#         slice_labels = []
#
#
#         ## TODO extract this to another function
#         empty = True
#         reject = False
#         max_bbox_area = 0
#         one_good_box = False
#         raster_id = 0 # TODO encapsulate the Rasters into a class which contains a trackable id
#
#         masks = [] # for bordering bounding boxes, those are blacked out, the masks are saved here to remove keypoints later
#
#
#         for annotation in hi.labels:
#             is_polygon = isinstance(annotation.polygon_s, shapely.Polygon)
#             is_keypoint = isinstance(annotation.keypoints, typing.List) and len(annotation.keypoints) > 0
#             is_bbox = isinstance(annotation.bbox_polygon, shapely.Polygon) and not is_polygon
#
#             # iterate through labels and check if any of the boxes intersects with the sliding window
#             if ((is_polygon and pol.intersects(annotation.polygon_s))
#                     or (is_bbox and pol.intersects(annotation.bbox_polygon))
#                     or (is_keypoint and pol.contains(annotation.keypoints[0].coordinate))):
#
#                 # any of the annotations is in the sliding window
#                 if is_bbox:
#                     intersection_polygon = shapely.intersection(pol, annotation.bbox_polygon)
#                     max_bbox_area = max(max_bbox_area, intersection_polygon.area)
#                 elif is_keypoint:
#                     is_point_inside = shapely.Point(annotation.keypoints[0].coordinate).within(pol)
#                 else:
#                     try:
#                         intersection_polygon = shapely.intersection(pol, annotation.polygon_s)
#                         max_bbox_area = max(max_bbox_area, intersection_polygon.area)
#                     except shapely.errors.GEOSException as e:
#                         logger.error(f"Error in intersection: {e}")
#                         logger.error(f"Annotation is not valid: {annotation.id} ")
#                         continue
#
#                 if is_bbox and annotation.bbox_polygon.within(pol):
#                     # the box is completely within the sliding window
#                     one_good_box = True
#                     empty = False
#                     # Translate the coordinates of the inner polygon
#                     translated_coords = [(x - minx, y - miny) for x, y in annotation.bbox_polygon.exterior.coords]
#                     # Create a new polygon with the translated coordinates
#                     translated_inner_polygon = Polygon(translated_coords)
#                     il = ImageLabel(
#                         id=str(uuid.uuid4()),
#                         class_name=annotation.class_name,
#                         bbox=[int(x) for x in translated_inner_polygon.bounds],
#                     )
#                     slice_labels.append(il)
#
#
#                 elif is_polygon and annotation.polygon_s.within(pol):
#                     # the Polygon is completely within the sliding window
#                     one_good_box = True
#                     empty = False
#                     # Translate the coordinates of the inner polygon
#                     translated_coords = [(x - minx, y - miny) for x, y in annotation.polygon]
#                     # Create a new polygon with the translated coordinates
#
#                     il = ImageLabel(
#                         id=annotation.id,
#                         class_name=annotation.class_name,
#                         polygon=translated_coords,
#                     )
#                     slice_labels.append(il)
#                     # a part of the box is outside of the sliding window, we want to black it out
#
#
#                 # Process the keypoints
#                 elif is_keypoint and annotation.keypoints[0].coordinate.within(pol):
#                     # translated_coords = [(k.x - minx, k.y - miny) for k in box.keypoints]
#                     # Create a new polygon with the translated coordinates
#                     empty = False
#                     box_keypoints = []
#                     for k in annotation.keypoints:
#                         kc = copy.deepcopy(k)
#                         kc.x = int(k.x - minx)
#                         kc.y = int(k.y - miny)
#                         box_keypoints.append(kc)
#
#                     #translated_keypoints = [Keypoint(x=int(k.x - minx), y=int(k.y - miny)) for k in box.keypoints]
#
#                     il = ImageLabel(
#                         id=annotation.id,
#                         class_name=annotation.class_name,
#                         keypoints=box_keypoints,
#                     )
#                     slice_labels.append(il)
#
#                 elif is_keypoint and not annotation.keypoints[0].coordinate.within(pol):
#                     # The keypoint is outside of the sliding window
#                     # So far we do not do anything here
#                     pass
#
#
#                 # elif box.attributes.get("partial", False) or box.attributes.get("visibility", -1) < 2:
#                 #     logger.info(f"Bor or Polygon is a partial={box.attributes.get('partial')} or is badly visible=box.attributes.get('visibility', -1)")
#                 #
#                 #     translated_coords = [(x - minx, y - miny) for x, y in box.bbox_polygon.exterior.coords]
#                 #     translated_inner_polygon = Polygon(translated_coords)
#                 #     sliced = blackout_bbox(sliced, bbox_x1y1x2y2=[int(x) for x in translated_inner_polygon.bounds])
#
#                 else:
#                     # logger.info(f"Box or polygon is not completly within the sliding window {annotation.id}")
#                     # Translate the coordinates of the inner polygon
#                     if edge_blackout:
#                         translated_coords = [(x - minx, y - miny) for x, y in annotation.bbox_polygon.exterior.coords]
#                         translated_inner_polygon = Polygon(translated_coords)
#                         masks.append(translated_inner_polygon)
#
#                         sliced = blackout_bbox(sliced, bbox_x1y1x2y2=[int(x) for x in translated_inner_polygon.bounds])
#
#                     else:
#                         raise NotImplementedError("Redraw the box or polygon to fit the sliding window")
#             else:
#                 # logger.info(f"Box or polygon is not within the sliding window {annotation.id}")
#                 pass
#         for m in masks:
#             # remove points which are in the mask
#             slice_labels = [sl for sl in slice_labels if isinstance(sl.incenter_centroid, shapely.Point) and sl.incenter_centroid.within(m) == False]
#
#
#         if empty == False and max_bbox_area < 5000:
#             # reject = True
#             # logger.warning(f"Should Rejecting image, label very small {hi.image_name} because of max_bbox_area {max_bbox_area}")
#             # slice_labels = []
#             pass
#         filename = str(Path(hi.image_name).stem)
#
#         # if (not reject and not empty) or one_good_box:
#
#         xx, yy = pol.exterior.coords.xy
#         slice_path_jpg = output_path / f"{filename}_x{int(xx[0])}_y{int(yy[0])}.jpg"
#
#         if (include_empty_frac(include_empty) or not empty) and not reject:
#
#             im = AnnotatedImage(
#                 image_id=str(uuid.uuid4()),
#                 dataset_name=dataset_name if dataset_name else DATA_SET_NAME,
#                 image_name=slice_path_jpg.name,
#                 labels=slice_labels,
#                 width=int(slice_width),
#                 height=int(slice_height))
#
#             images.append(im)
#
#             sliced_im = sliced.convert("RGB")
#             sliced_im.save(slice_path_jpg)
#             image_paths.append(slice_path_jpg)
#
#         raster_id += 1
#
#     return images, image_paths
#
#
# def crop_out_images_v3(image: PIL.Image,
#                        rasters: List[shapely.Polygon],
#                        ) -> List[PIL.Image]:
#     """
#     iterate through rasters and crop out the tiles from the image return the new images
#
#     :param rasters:
#     :param output_path:
#     :param full_images_path:
#
#     :param hi:
#     """
#
#     # imr = np.array(image)
#
#     # Convert to string if you need a string representation
#     images = []
#
#     # slice the image in tiles
#
#     # sliding window of the image
#     for pol in rasters:
#         assert isinstance(pol, shapely.Polygon)
#         sliced = image.crop(pol.bounds)
#         images.append(sliced)
#
#     return images
#
#
# def pad_to_multiple(original_image_path: Path, padded_image_path: Path,
#                     slice_width: int, slice_height: int, overlap: int):
#     """
#     Pads an image so that its width and height are rounded up to the nearest step defined by:
#     step_x = slice_width - overlap
#     step_y = slice_height - overlap
#
#     Padding is applied only to the bottom (y_max) and right (x_max) edges.
#
#     Parameters:
#         original_image_path (Path): Path to the original image file.
#         padded_image_path (Path): Path to save the padded image.
#         slice_width (int): Width of each tile.
#         slice_height (int): Height of each tile.
#         overlap (int): Overlap between tiles in pixels.
#
#     Returns:
#         (int, int): The new padded width and height of the image.
#     """
#     # Open the original image
#     image = Image.open(original_image_path)
#     width, height = image.size
#
#     # Compute the step sizes
#     step_x = slice_width - overlap
#     step_y = slice_height - overlap
#
#     # Calculate padding so that width and height become multiples of step_x and step_y respectively
#     padding_x = (step_x - (width % step_x)) % step_x
#     padding_y = (step_y - (height % step_y)) % step_y
#
#     new_width = width + padding_x
#     new_height = height + padding_y
#
#     # Create a new image with the padded dimensions
#     padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))  # black background
#     padded_image.paste(image, (0, 0))
#     padded_image.save(padded_image_path)
#
#     logger.info(f"Padded image saved to {padded_image_path} with size: {new_width}x{new_height}")
#     return new_width, new_height