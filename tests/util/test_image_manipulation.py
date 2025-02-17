import pytest
from pathlib import Path
import PIL
import PIL.Image
from typing import List

from active_learning.util.image_manipulation import crop_out_individual_object, crop_out_images, crop_out_images_v2, \
    create_regular_raster_grid, sliced_predict_geotiff
from com.biospheredata.helper.image.manipulation.slice import GeoSlicer
from com.biospheredata.types.HastyAnnotationV2 import ImageLabel, AnnotatedImage, HastyAnnotationV2, hA_from_file

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

    labels = [il1, il2]
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


def test_crop_out_individual_object(image_label, output_path, full_images_path):
    crops = crop_out_individual_object(image_label,
                                       full_images_path=full_images_path,
                                       output_path=output_path, offset=50)

    assert len(crops) == 2


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



def test_crop_out_individual_object_constant(image_label, output_path, full_images_path):
    """
    Create Thumbnails ob objects
    :param image_label:
    :param output_path:
    :param full_images_path:
    :return:
    """

    thumbnails = crop_out_individual_object(image_label,
                                       full_images_path=full_images_path,
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


