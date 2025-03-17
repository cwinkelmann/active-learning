import tempfile

import pytest
from pathlib import Path
import PIL
import PIL.Image
from typing import List

from active_learning.types.ImageCropMetadata import ImageCropMetadata
from active_learning.util.image_manipulation import crop_out_individual_object, crop_out_images, crop_out_images_v2, \
    create_regular_raster_grid, sliced_predict_geotiff
from com.biospheredata.types.HastyAnnotationV2 import ImageLabel, AnnotatedImage, HastyAnnotationV2, hA_from_file, \
    Keypoint

from tests.helper import get_testdata_base_path

@pytest.fixture
def real_annotations():
    base_path = get_testdata_base_path()

    hA = hA_from_file(base_path / "FMO02_03_05_labels.json")

    return hA

@pytest.fixture
def image_label():
    image_name = "DJI_0331.JPG"
    il1 = ImageLabel(bbox=[0, 0, 100, 100], class_name="iguana")
    il2 = ImageLabel(bbox=[100, 100, 200, 200], class_name="iguana")
    il3 = ImageLabel(bbox=[3000, 3400, 3100, 3500], class_name="iguana")

    labels = [il1, il2, il3]
    img = AnnotatedImage(image_name=image_name, dataset_name="FMO05", width=5472, height=3648, labels=labels)

    return img

@pytest.fixture
def image_label_points():
    image_name = "DJI_0331.JPG"
    il1 = ImageLabel(bbox=None,   # if not available, keep as None
                polygon=None,
                mask=[],     # or adjust if you have mask data
                keypoints = [Keypoint(
                    x=123,
                    y=123,
                    keypoint_class_id="123456"
                )],
                     class_name="iguana"    )

    il2 = ImageLabel(bbox=None,  # if not available, keep as None
                     polygon=None,
                     mask=[],  # or adjust if you have mask data
                     keypoints=[Keypoint(
                         x=567,
                         y=987,
                         keypoint_class_id="123456"
                     )],
                     class_name="iguana")

    il3 = ImageLabel(bbox=None,  # if not available, keep as None
                     polygon=None,
                     mask=[],  # or adjust if you have mask data
                     keypoints=[Keypoint(
                         x=1500,
                         y=123,
                         keypoint_class_id="123456"
                     )],
                     class_name="iguana")

    labels = [il1, il2, il3]
    img = AnnotatedImage(image_name=image_name, dataset_name="FMO05", width=5472, height=3648, labels=labels)

    return img

@pytest.fixture
def image_label_polygon():
    image_name = "DJI_0331.JPG"
    # Rectangle shape
    lizard = ImageLabel(
        polygon=[(150, 200), (150, 300), (250, 300), (250, 200), (150, 200)],
        class_name="lizard"
    )

    # Triangle shape
    snake = ImageLabel(
        polygon=[(500, 600), (550, 700), (450, 700), (500, 600)],
        class_name="snake"
    )

    # Pentagon shape
    turtle = ImageLabel(
        polygon=[(1000, 1000), (1050, 950), (1100, 1000), (1075, 1075), (1025, 1075), (1000, 1000)],
        class_name="turtle"
    )

    # Hexagon shape
    crocodile = ImageLabel(
        polygon=[(2000, 2000), (2100, 2000), (2150, 2075), (2100, 2150), (2000, 2150), (1950, 2075), (2000, 2000)],
        class_name="crocodile"
    )

    labels = [lizard, snake, turtle, crocodile]
    img = AnnotatedImage(image_name=image_name, dataset_name="FMO05", width=5472, height=3648, labels=labels)

    return img


@pytest.fixture
def output_path():
    base_path = get_testdata_base_path()
    output_path = base_path / "./output"
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


@pytest.fixture
def full_images_path():
    base_path = get_testdata_base_path()
    full_images_path = base_path / Path("images")
    return full_images_path

def test_crop_out_individual_object_offset(image_label, output_path, full_images_path):
    """
    Test cropping out individual objects by offset
    :param image_label:
    :param output_path:
    :param full_images_path:
    :return:
    """
    im = PIL.Image.open(full_images_path / image_label.dataset_name / image_label.image_name)

    boxes, image_mappings, cropped_annotated_images = crop_out_individual_object(image_label,
                                       im=im,
                                       output_path=output_path,
                                       offset=112)

    assert len(boxes) == 2

def test_crop_out_individual_object_constant(image_label, output_path, full_images_path):
    """
    Test cropping out individual objects
    :param image_label:
    :param output_path:
    :param full_images_path:
    :return:
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = Path(tmpdirname)

        im = PIL.Image.open(full_images_path / image_label.dataset_name / image_label.image_name)

        image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(image_label,
                                           im=im,
                                           output_path=output_path,
                                           width=512, height=512)

        assert len(image_mappings) == 3
        assert len(image_mappings) == len(cropped_annotated_images)
        assert isinstance(image_mappings[0], ImageCropMetadata)

        output_images = list(output_path.glob("*.jpg"))
        assert len(output_images) == 3
        assert cropped_annotated_images

        ## TODO project back


def test_crop_out_individual_object_constant_polygon(image_label_polygon, output_path, full_images_path):
    """
    Test cropping out individual objects
    :param image_label:
    :param output_path:
    :param full_images_path:
    :return:
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = Path(tmpdirname)

        im = PIL.Image.open(full_images_path / image_label_polygon.dataset_name / image_label_polygon.image_name)

        image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(
            image_label_polygon,
            im=im,
            output_path=output_path,
            width=512,
            height=512)

        assert len(image_mappings) == 3
        assert len(image_mappings) == len(cropped_annotated_images)
        assert isinstance(image_mappings[0], ImageCropMetadata)

        output_images = list(output_path.glob("*.jpg"))
        assert len(output_images) == 3
        assert cropped_annotated_images


def test_project_annotation_back():
    """
    Based on a small crop, project the annotation back to the original image
    :return:
    """


def test_crop_out_individual_object_real(real_annotations: HastyAnnotationV2, output_path, full_images_path):
    """
    Test cropping out individual objects from a real annotation file
    :param real_annotations:
    :param output_path:
    :param full_images_path:
    :return:
    """
    img_0331 = [i for i in real_annotations.images if i.image_name == "DJI_0331.JPG"][0]


    crops = crop_out_individual_object(img_0331,
                                       full_images_path=full_images_path,
                                       output_path=output_path, offset=50)

    assert len(crops) == 2



def test_crop_out_individual_object_constant_2(image_label, output_path, full_images_path):
    """
    Create Thumbnails ob objects
    :param image_label:
    :param output_path:
    :param full_images_path:
    :return:
    """
    im = PIL.Image.open(full_images_path / image_label.dataset_name / image_label.image_name)

    thumbnails = crop_out_individual_object(image_label,
                                       im=im,
                                       output_path=output_path,
                                       width=224, height=224, offset=None)

    assert len(thumbnails) == 2


def test_crop_out_images(image_label, output_path, full_images_path):
    crop_size = 224
    images_without_objects, images_with_objects, images = crop_out_images(image_label,
                                                                          slice_height=crop_size,
                                                                          slice_width=crop_size,
                                                                          full_images_path=full_images_path,
                                                                          output_path=output_path)

    assert len(images_without_objects) == 383
    assert len(images_with_objects) == 1, "both boxes are in the same frame"
    assert len(images) == 1


def test_crop_out_images_v2(image_label, output_path, full_images_path, cutout_boxes):
    """
    Crop out images from a grid starting with (0,0)

    :param image_label:
    :param output_path:
    :param full_images_path:
    :param cutout_boxes:
    :return:
    """

    grid, _ = create_regular_raster_grid(max_x=image_label.width,
                                         max_y=image_label.height,
                                         slice_height=1280,
                                         slice_width=1280,
                                         overlap=0)

    images = crop_out_images_v2(image_label,
                               rasters=grid,
                               full_images_path=full_images_path,
                               output_path=output_path)

    assert len(images) == len(grid), "8 images should be created from the grid"


