import random
from typing import Optional

import PIL
import pandas as pd
import shapely
from matplotlib import pyplot as plt, patches
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger
from shapely import Point, affinity
from shapely.geometry import Polygon
import math
from PIL import Image, ImageDraw

from com.biospheredata.types.HastyAnnotation import HastyAnnotation
from com.biospheredata.types.annotationbox import Bbox, Annotation, BboxXYXY
from com.biospheredata.types.annotationbox import LargeAnnotation

def frame_from_big_image(image_name,
                         base_path,
                         full_images_path,
                         full_hasty_annotation_file_path,
                         flight_path):
    """
    # Fixme this is spaghetti code. This function should not call the clip functions
    :param image_name:
    :param base_path:
    :param full_images_path:
    :param full_hasty_annotation_file_path:
    :param flight_path:
    :return:
    """

    output_path = base_path.joinpath("output")
    output_path.mkdir(exist_ok=True)
    # shutil.rmtree(output_path)
    # output_path.mkdir(exist_ok=False)

    boxes = build_generic_annotation_list(full_hasty_annotation_file_path, image_name)

    image_tiles, df_boxes, _ = clip_image_to_videoframes_v2(
        slice_size=1280,
        image_path=full_images_path.joinpath(image_name),
        flight_path=flight_path,
        output_path=output_path,
        labels=boxes,
        visualize=False
    )

    return df_boxes, image_tiles





def plot_perspective(ax, p: Point, angle: float, length):
    """
    visualise where the drone looks at
    :param ax:
    :param p:
    :param angle:
    :param length:
    :return:
    """
    x, y = p.x, p.y

    endx = x + length * math.cos(math.radians(angle))
    endy = y + length * math.sin(math.radians(angle))
    ax.plot([x, endx], [y, endy])

def plot_footprint(ax, p: Point, o_x, o_y, angle: float):
    """
    visualise where the drone looks at
    :param ax:
    :param p:
    :param angle:
    :param length:
    :return:
    """

    footprint = Polygon([
        (p.x-o_x, p.y-o_y),
        (p.x-o_x, p.y+o_y),
        (p.x+o_x, p.y + o_y),
        (p.x+o_x, p.y - o_y),
        (p.x-o_x, p.y-o_y),
    ])
    affine = affinity.rotate(footprint, angle, (p.x, p.y))
    affine = affinity.rotate(footprint, angle)

    ## get all x and y coordindates of polygon
    x, y = affine.exterior.coords.xy

    ax.plot(x, y)


def visualize_bounding_box_v2(imname,
                              base_path: Path,
                              labels: list[Annotation],
                              output_file_name=None,
                              show=False,
                              output_path=None,
                              inframe_text=None,
                              size=(35, 35)):
    """
    # TODO this is probably deprecated
    Visualize Boxes in
    labels is a list of Annotation. Each list depics the bounding box and the ID
    """
    if output_file_name is None:
        output_file_name = imname

    im = Image.open(base_path.joinpath(imname))
    imr = np.array(im, dtype=np.uint8)

    pilimage = Image.open(base_path.joinpath(imname))


    fig, ax = plt.subplots(1,
                           figsize=size,
                           dpi=150)
    plt.tight_layout()
    # ax.imshow(imr)
    plt.imshow(pilimage)

    for label in labels:
        # Create a Rectangle patch
        box = label.bbox
        box = Polygon([(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)])
        rect = patches.Polygon(box.exterior.coords,
                               linewidth=2, edgecolor='r',
                               facecolor='none')

        # Add the patch to the axes
        ax.add_patch(rect)
        # obviously use a different formula for different shapes
        # plt.text(list(box.exterior.coords)[4][0],
        #          list(box.exterior.coords)[4][1] - 5,
        #          str(f"{label.class_id}"), color="red", fontsize=20
        #          )
    plt.text(30, 150, inframe_text, color="white", fontsize=15)

    plt.tight_layout()
    if output_path is not None:
        Path.mkdir(output_path, exist_ok=True, parents=True)
        plt.savefig(output_path.joinpath(f'{output_file_name}'))
    if show:
        plt.show()
        # st.pyplot(fig, use_container_width=2200, clear_figure=True)
    plt.close()

    return output_file_name


def get_box_from_polygon(new_iguanas_boxes, columns=["minx", "miny", "maxx", "maxy", "individual_id"]):
    """
    https://shapely.readthedocs.io/en/stable/manual.html
    :param new_iguanas_boxes:
    :return:
    """
    df_boxes = pd.DataFrame(new_iguanas_boxes, columns=columns)

    return df_boxes


def clip_image_to_videoframes(slice_size,
                              image_path,
                              stride_x: int,
                              stride_y: int,
                              output_path: Path,
                              label_path=None,
                              labels=[],
                              ext=".JPG",
                              visualize=False):
    """
    simply cut an image into equally sliced images along the stride steps in x and y direction

    @param slice_size:
    @param falsepath:
    @param imname:
    @param imr:
    @param output_path:
    @param boxes:
    @param ext:
    @return:
    """

    with Image.open(image_path) as im:
        # im = Image.open(image_path)
        imr = np.array(im, dtype=np.uint8)
    height = imr.shape[0]
    width = imr.shape[1]

    image_tiles = []
    list_boxes_of_dfboxes = []
    labels_paths = []
    cutout_polygons = []

    output_path.mkdir(parents=True, exist_ok=True)
    y = 0
    slice_boxes_x = [x for x in range(0, width - slice_size, stride_x)]
    slice_boxes_y = [y for y in range(0, height - slice_size, stride_y)]
    slice_boxes_xy = list(zip(slice_boxes_x, slice_boxes_y))

    # slice_boxes_xy = [(x, y) for x in range(0, width-slice_size, stride_x)]

    if stride_y == 0:
        slice_boxes_y = [0 for x in slice_boxes_x]
        slice_boxes_y = [0]
    else:
        slice_boxes_y = [y for y in range(0, height - slice_size, stride_y)]
        slice_boxes_y = [0]

    # for i in slice_boxes_y:
    #     for j in slice_boxes_x:
    frame = 0
    offset_x = 0
    offset_y = 0
    for j, i in slice_boxes_xy:
        x1 = j
        y1 = i

        x2 = j + slice_size
        y2 = i + slice_size

        slice_pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        ## Work through the labels
        new_iguanas_boxes = []
        for box in labels:
            if slice_pol.intersects(box["bbox"]):
                # the annotation is in the sliding window
                inter = slice_pol.intersection(box["bbox"])

                # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                new_box = inter.envelope

                # get central point for the new bounding box
                centre = new_box.centroid

                # get coordinates of polygon vertices
                x, y = new_box.exterior.coords.xy
                x = np.array(x) - offset_x
                y = np.array(y) - offset_y

                new_box = Polygon(list(zip(x, y)))
                # get bounding box width and height normalized to slice size
                new_width = (max(x) - min(x)) / slice_size
                new_height = (max(y) - min(y)) / slice_size

                # we have to normalize central x and invert y for yolo format
                new_x = (centre.coords.xy[0][0] - x1) / slice_size
                new_y = (y1 - centre.coords.xy[1][0]) / slice_size

                # TODO don't forget about the other detail like individual id, class id etc,

                iguana_labels = list(new_box.bounds)
                iguana_labels.append(box["individual_id"])
                new_iguanas_boxes.append(iguana_labels)

        offset_x += stride_x
        offset_y += stride_y

        pol = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        cutout_polygons.append((x1, y1, x2, y2))
        # TODO draw the polygon on the original image
        slice_labels = []

        # https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/

        sliced = imr[y1:y2, x1:x2]

        # sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
        logger.info(f"trying to cut this x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
        sliced_im = Image.fromarray(sliced)
        filename = image_path.name

        slice_path_jpg = output_path.joinpath(filename.replace(ext, f'_f{frame:04d}{ext}'))
        slice_path_jpg = slice_path_jpg.parent.joinpath(f"{frame:04d}{ext}")
        slice_path_csv = slice_path_jpg.parent.joinpath(f"{frame:04d}.csv")
        frame += 1
        sliced_im = sliced_im.convert("RGB")

        sliced_im.save(f"{slice_path_jpg}", "JPEG", quality=95, optimize=True, progressive=True)
        logger.info(f"slices saved to {slice_path_jpg}")

        # yield slice_path_jpg
        image_tiles.append(slice_path_jpg)

        if visualize:
            ## add the annotations to the slice for visualisation
            visualize_bounding_box_v2(imname=slice_path_jpg, labels=new_iguanas_boxes)

        # persist the new iguana boxes
        df_boxes = get_box_from_polygon(new_iguanas_boxes)
        df_boxes.to_csv(slice_path_csv)
        logger.info(f"sliced boxes saved to {slice_path_csv}")

        list_boxes_of_dfboxes.append(df_boxes)

    """
    ffmpeg -framerate 1 -pattern_type glob -i 'DJI*.JPG' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    ffmpeg -framerate 1 -i DJI%04d.JPG -c:v libx264 -r 2 -pix_fmt yuv420p out.gif
    ffmpeg -framerate 1 -i DJI_0001_f%04d*.JPG -c:v libx264 -r 2 -pix_fmt yuv420p out.mp4
    """

    # draw the virtual frames and the iguana boxes on the original shot
    with Image.open(image_path) as imdraw:
        draw = ImageDraw.Draw(imdraw)
        for x1, y1, x2, y2 in cutout_polygons:
            # draw.rectangle((200, 100, 300, 200), fill="red")
            draw.rectangle((x1, y1, x2, y2), outline="yellow",
                           width=5)

            # add the iguana labels
            # draw.rectangle((200, 100, 300, 200), fill="red")
            for pol in labels:
                draw.polygon(pol["bbox"].exterior.coords, outline="red", width=5)
            # for l in labels:
            # draw.rectangle((l["x1"], l["y1"], l["x2"], l["y2"]), outline="red", width=5)

        imdraw.save(output_path.joinpath("flight_overview.jpg"))

    return image_tiles, list_boxes_of_dfboxes


def clip_image_to_videoframes_v2(slice_size,
                                 image_path,
                                 flight_path: list[shapely.Point],
                                 output_path: Path,
                                 label_path=None,
                                 labels= [],
                                 ext=".JPG",
                                 visualize=False, show=False):
    """
    simply cut an image into equally sliced images along a flight path

    @return:
    """
    # in case we are cutting an orthomosaic
    PIL.Image.MAX_IMAGE_PIXELS = 509083780
    with Image.open(image_path) as im:
        # im = Image.open(image_path)
        imr = np.array(im, dtype=np.uint8)
    height = imr.shape[0]
    width = imr.shape[1]

    image_tiles = []
    list_boxes_of_dfboxes = []
    labels_paths = []
    cutout_polygons = []

    output_path.mkdir(parents=True, exist_ok=True)

    frame = 0
    offset_x = 0
    offset_y = 0

    last_x = None
    last_y = None

    # for j, i in flight_path: # these where tuples before
    for p in flight_path:
        ## TODO
        i = int(p.y) # TODO, it would be more intuitive if the points would be the center.
        j = int(p.x)
        x1 = j
        y1 = i
        offset_x = x1  # TODO refactor this
        offset_y = y1
        x2 = j + slice_size
        y2 = i + slice_size
        # the cutout quare which is defined by the flight path
        cutout_image = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        ## Work through the labels
        new_iguanas_boxes = []
        yolo_annotation_iguanas_boxes = []
        oo_new_iguanas_boxes = []

        for box_obj in labels:
            if cutout_image.intersects(box_obj["bbox"]):
                """
                the annotation is part of the sliding window
                """
                inter = cutout_image.intersection(box_obj["bbox"])

                # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection ( to describe four lines you need 5 points )
                new_box = inter.envelope

                # get central point for the new bounding box
                centre = new_box.centroid
                ## FIXME that the offset is zero here is wrong. Because we are in the middle of the picture

                # get coordinates of polygon vertices
                x, y = new_box.exterior.coords.xy
                x = np.array(x) - offset_x
                y = np.array(y) - offset_y

                new_box = Polygon(list(zip(x, y)))
                # get normalized bounding box width and height normalized to slice size
                new_width = (max(x) - min(x)) / slice_size
                new_height = (max(y) - min(y)) / slice_size

                # we have to normalize central x and invert y for yolo format
                new_x = (centre.coords.xy[0][0] - x1) / slice_size
                new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                yolo_box = (new_x, new_y, new_height, new_width)
                # TODO don't forget about the other detail like individual id, class id etc,

                iguana_labels = list(new_box.bounds)
                oo_new_iguanas_boxes.append(Annotation(class_id=box_obj["individual_id"],
                                                       bbox=BboxXYXY(*iguana_labels),
                                                       image_name="TODO"))
                iguana_labels.append(box_obj["individual_id"])
                yolo_annotation_iguanas_boxes.append(iguana_labels)

                new_iguanas_boxes.append(new_box)

        if last_x is None:
            offset_x = 0
        else:
            offset_x = offset_x + last_x - i
        last_x = i

        if last_y is None:
            offset_y = 0
        else:
            offset_y = offset_y + last_y - j
        last_y = j

        # offset_x += stride_x
        # offset_y += stride_y

        pol = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        cutout_polygons.append((x1, y1, x2, y2))

        slice_labels = []

        # https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
        # slice an image out of the original
        sliced = imr[y1:y2, x1:x2]

        # sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
        logger.info(f"trying to cut this x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
        sliced_im = Image.fromarray(sliced)
        filename = image_path.name

        slice_path_jpg = output_path.joinpath(filename.replace(ext, f'_f{frame:04d}{ext}'))
        slice_path_jpg = slice_path_jpg.parent.joinpath(f"{frame:04d}{ext}")
        slice_path_csv = slice_path_jpg.parent.joinpath(f"{frame:04d}.csv")
        frame += 1
        sliced_im = sliced_im.convert("RGB")

        sliced_im.save(f"{slice_path_jpg}", "JPEG", quality=95, optimize=True, progressive=True)
        logger.info(f"slices saved to {slice_path_jpg}")

        # yield slice_path_jpg
        image_tiles.append(slice_path_jpg)

        ## FIXME externalise this
        if visualize:
            ## add the annotations to the slice for visualisation
            visualize_bounding_box_v2(imname=slice_path_jpg.name,
                                      labels=oo_new_iguanas_boxes,
                                      base_path=output_path,
                                      output_path=output_path.joinpath("vis"),
                                      show=show
                                      )

        # persist the new iguana boxes
        df_boxes = get_box_from_polygon(yolo_annotation_iguanas_boxes)
        df_boxes.to_csv(slice_path_csv)
        logger.info(f"sliced boxes saved to {slice_path_csv}")

        list_boxes_of_dfboxes.append(df_boxes)

    """
    ffmpeg -framerate 1 -pattern_type glob -i 'DJI*.JPG' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    ffmpeg -framerate 1 -i DJI%04d.JPG -c:v libx264 -r 2 -pix_fmt yuv420p out.mp4
    ffmpeg -framerate 1 -i DJI_0001_f%04d*.JPG -c:v libx264 -r 2 -pix_fmt yuv420p out.mp4
    """
    logger.info(f"Drawing the simulated drone shots on the map.")
    # draw the virtual frames and the iguana boxes on the original shot
    with Image.open(image_path) as imdraw:
        draw = ImageDraw.Draw(imdraw)
        for x1, y1, x2, y2 in cutout_polygons:
            logger.info(f"drawing: {(x1, y1, x2, y2)}")

            # draw.rectangle((200, 100, 300, 200), fill="red")
            draw.rectangle((x1, y1, x2, y2), outline="yellow",
                           width=5)

            # add the iguana labels
            # draw.rectangle((200, 100, 300, 200), fill="red")
            for pol in labels:
                draw.polygon(pol["bbox"].exterior.coords, outline="red", width=5)
            # for l in labels:
            # draw.rectangle((l["x1"], l["y1"], l["x2"], l["y2"]), outline="red", width=5)
        logger.info(f"Polygon: {cutout_polygons}")
        imdraw.save(output_path.joinpath("flight_overview.jpg"))

    return image_tiles, list_boxes_of_dfboxes, output_path.joinpath("flight_overview.jpg")


def clip_image_to_videoframes_v3(slice_size,
                                 image_path,
                                 flight_path: list[shapely.Point],
                                 output_path: Path,
                                 label_path=None,
                                 labels=[LargeAnnotation],
                                 ext=".JPG",
                                 visualize=False,
                                 show=False):
    """
    simply cut an image into equally sliced images along a flight path
    given annotations will be split to the corresponding slices

    @return:
    """
    # in case we are cutting an orthomosaic
    PIL.Image.MAX_IMAGE_PIXELS = 509083780
    with Image.open(image_path) as im:
        imr = np.array(im, dtype=np.uint8)

    image_tiles = []
    list_boxes_of_dfboxes = []
    cutout_polygons = []
    output_path.mkdir(parents=True, exist_ok=True)

    frame = 0
    last_x = None
    last_y = None

    # for j, i in flight_path: # these where tuples before
    for p in flight_path:
        ## TODO
        i = int(p.y) # TODO, it would be more intuitive if the points would be the center.
        j = int(p.x)
        x1 = j
        y1 = i
        offset_x = x1  # TODO refactor this
        offset_y = y1
        x2 = j + slice_size
        y2 = i + slice_size
        # the cutout quare which is defined by the flight path
        cutout_image = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        ## Work through the labels
        new_iguanas_boxes = []
        yolo_annotation_iguanas_boxes = []
        oo_new_iguanas_boxes = []

        for box_obj in labels:
            if cutout_image.intersects(box_obj.bbox_polygon):
                """
                the annotation is part of the sliding window
                """
                inter = cutout_image.intersection(box_obj.bbox_polygon)

                # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection ( to describe four lines you need 5 points )
                new_box = inter.envelope

                # get central point for the new bounding box
                centre = new_box.centroid
                ## FIXME that the offset is zero here is wrong. Because we are in the middle of the picture

                # get coordinates of polygon vertices
                x, y = new_box.exterior.coords.xy
                x = np.array(x) - offset_x
                y = np.array(y) - offset_y

                new_box = Polygon(list(zip(x, y)))
                # get normalized bounding box width and height normalized to slice size
                new_width = (max(x) - min(x)) / slice_size
                new_height = (max(y) - min(y)) / slice_size

                # we have to normalize central x and invert y for yolo format
                new_x = (centre.coords.xy[0][0] - x1) / slice_size
                new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                yolo_box = (new_x, new_y, new_height, new_width)
                # TODO don't forget about the other detail like individual id, class id etc,

                iguana_labels = [int(x) for x in list(new_box.bounds)]
                oo_new_iguanas_boxes.append(Annotation(class_id=box_obj.ID,
                                                       bbox=BboxXYXY(*iguana_labels),
                                                       image_name="TODO"))

                iguana_labels.append(box_obj.ID)
                yolo_annotation_iguanas_boxes.append(iguana_labels)

                new_iguanas_boxes.append(new_box)
            ## end of         for box_obj in labels:


        pol = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        cutout_polygons.append((x1, y1, x2, y2))

        slice_labels = []

        # https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
        # slice an image out of the original
        sliced = imr[y1:y2, x1:x2]

        # sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
        logger.info(f"trying to cut this x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
        sliced_im = Image.fromarray(sliced)
        filename = image_path.name

        slice_path_jpg = output_path.joinpath(filename.replace(ext, f'_f{frame:04d}{ext}'))
        slice_path_jpg = slice_path_jpg.parent.joinpath(f"{frame:04d}{ext}")
        slice_path_csv = slice_path_jpg.parent.joinpath(f"{frame:04d}.csv")
        frame += 1
        sliced_im = sliced_im.convert("RGB")

        sliced_im.save(f"{slice_path_jpg}", "JPEG", quality=95, optimize=True, progressive=True)
        logger.info(f"slices saved to {slice_path_jpg}")

        # yield slice_path_jpg
        image_tiles.append(slice_path_jpg)

        ## FIXME externalise this
        if visualize:
            ## add the annotations to the slice for visualisation
            visualize_bounding_box_v2(imname=slice_path_jpg.name,
                                      labels=oo_new_iguanas_boxes,
                                      base_path=output_path,
                                      output_path=output_path.joinpath("vis"),
                                      show=show
                                      )

        # persist the new iguana boxes
        df_boxes = get_box_from_polygon(yolo_annotation_iguanas_boxes)
        df_boxes.to_csv(slice_path_csv, index=False)
        logger.info(f"sliced boxes saved to {slice_path_csv}")

        list_boxes_of_dfboxes.append(df_boxes)
        ## up till here we are slicing the image

    """
    TODO is a code snipped to create a video from the images
    ffmpeg -framerate 1 -pattern_type glob -i 'DJI*.JPG' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    ffmpeg -framerate 1 -i DJI%04d.JPG -c:v libx264 -r 2 -pix_fmt yuv420p out.mp4
    ffmpeg -framerate 1 -i DJI_0001_f%04d*.JPG -c:v libx264 -r 2 -pix_fmt yuv420p out.mp4
    """

    logger.info(f"Drawing the simulated drone shots on the map.")
    # draw the virtual frames and the iguana boxes on the original shot
    with Image.open(image_path) as imdraw:
        draw = ImageDraw.Draw(imdraw)
        for x1, y1, x2, y2 in cutout_polygons:
            logger.info(f"drawing: {(x1, y1, x2, y2)}")

            # draw.rectangle((200, 100, 300, 200), fill="red")
            draw.rectangle((x1, y1, x2, y2), outline="yellow",
                           width=5)

            # add the iguana labels
            # draw.rectangle((200, 100, 300, 200), fill="red")
            for pol in labels:
                draw.polygon(pol.bbox_polygon.exterior.coords, outline="red", width=5)
            # for l in labels:
            # draw.rectangle((l["x1"], l["y1"], l["x2"], l["y2"]), outline="red", width=5)
        logger.info(f"Polygon: {cutout_polygons}")
        imdraw.save(output_path.joinpath("flight_overview.jpg"))

    return image_tiles, list_boxes_of_dfboxes, output_path.joinpath("flight_overview.jpg")


def clip_image_to_videoframes_v4(slice_size,
                              image_path,
                              stride_x: int,
                              stride_y: int,
                              output_path: Path,
                              label_path=None,
                              labels=[],
                              ext=".JPG",
                              visualize=False):
    """
    simply cut an image into equally sliced images along the stride steps in x and y direction

    @param slice_size:
    @param falsepath:
    @param imname:
    @param imr:
    @param output_path:
    @param boxes:
    @param ext:
    @return:
    """

    with Image.open(image_path) as im:
        # im = Image.open(image_path)
        imr = np.array(im, dtype=np.uint8)
    height = imr.shape[0]
    width = imr.shape[1]

    image_tiles = []
    list_boxes_of_dfboxes = []
    labels_paths = []
    cutout_polygons = []

    output_path.mkdir(parents=True, exist_ok=True)
    y = 0
    slice_boxes_x = [x for x in range(0, width - slice_size, stride_x)]
    slice_boxes_y = [y for y in range(0, height - slice_size, stride_y)]
    slice_boxes_xy = list(zip(slice_boxes_x, slice_boxes_y))

    # slice_boxes_xy = [(x, y) for x in range(0, width-slice_size, stride_x)]

    if stride_y == 0:
        slice_boxes_y = [0 for x in slice_boxes_x]
        slice_boxes_y = [0]
    else:
        slice_boxes_y = [y for y in range(0, height - slice_size, stride_y)]
        slice_boxes_y = [0]

    # for i in slice_boxes_y:
    #     for j in slice_boxes_x:
    frame = 0
    offset_x = 0
    offset_y = 0
    for j, i in slice_boxes_xy:
        x1 = j
        y1 = i

        x2 = j + slice_size
        y2 = i + slice_size

        slice_pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        ## Work through the labels
        new_iguanas_boxes = []
        for box in labels:
            if slice_pol.intersects(box["bbox"]):
                # the annotation is in the sliding window
                inter = slice_pol.intersection(box["bbox"])

                # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                new_box = inter.envelope

                # get central point for the new bounding box
                centre = new_box.centroid

                # get coordinates of polygon vertices
                x, y = new_box.exterior.coords.xy
                x = np.array(x) - offset_x
                y = np.array(y) - offset_y

                new_box = Polygon(list(zip(x, y)))
                # get bounding box width and height normalized to slice size
                new_width = (max(x) - min(x)) / slice_size
                new_height = (max(y) - min(y)) / slice_size

                # we have to normalize central x and invert y for yolo format
                new_x = (centre.coords.xy[0][0] - x1) / slice_size
                new_y = (y1 - centre.coords.xy[1][0]) / slice_size

                # TODO don't forget about the other detail like individual id, class id etc,

                iguana_labels = list(new_box.bounds)
                iguana_labels.append(box["individual_id"])
                new_iguanas_boxes.append(iguana_labels)

        offset_x += stride_x
        offset_y += stride_y

        pol = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        cutout_polygons.append((x1, y1, x2, y2))
        # TODO draw the polygon on the original image
        slice_labels = []

        # https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/

        sliced = imr[y1:y2, x1:x2]

        # sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
        logger.info(f"trying to cut this x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
        sliced_im = Image.fromarray(sliced)
        filename = image_path.name

        slice_path_jpg = output_path.joinpath(filename.replace(ext, f'_f{frame:04d}{ext}'))
        slice_path_jpg = slice_path_jpg.parent.joinpath(f"{frame:04d}{ext}")
        slice_path_csv = slice_path_jpg.parent.joinpath(f"{frame:04d}.csv")
        frame += 1
        sliced_im = sliced_im.convert("RGB")

        sliced_im.save(f"{slice_path_jpg}", "JPEG", quality=95, optimize=True, progressive=True)
        logger.info(f"slices saved to {slice_path_jpg}")

        # yield slice_path_jpg
        image_tiles.append(slice_path_jpg)


    return image_tiles, list_boxes_of_dfboxes