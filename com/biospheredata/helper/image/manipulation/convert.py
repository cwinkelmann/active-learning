import os

import rasterio.shutil
from loguru import logger

## there is a bug: 01/21/2023 18:52:01 - WARNING - rasterio._env -   CPLE_AppDefined in 4-band JPEGs will be interpreted on reading as in CMYK colorspace
## 2023-03-27 this is partly solved. First, hasty can deal with georeferenced jpgs now and I convert the images now to RGB now
def convert_geotiff_to_jpg(FORCE_UPDATE, sliced_jpg_image_path, sliced_tif_image_path):
    """
    transform the geotiff file to jpgs. One reason is to yolo needs jpgs.
    :param FORCE_UPDATE:
    :param sliced_jpg_image_path:
    :param sliced_tif_image_path:
    :return:
    """
    # convert the geotiff slice to a jpg
    if not os.path.exists(str(sliced_jpg_image_path)) or FORCE_UPDATE:
        logger.info(f"write {str(sliced_jpg_image_path)} to disk")
        rasterio.shutil.copy(
            str(sliced_tif_image_path),
            str(sliced_jpg_image_path),
            driver='JPEG')

        ## This is a bit of a hack because hasty cannot deal with georeferenced jpgs
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = int(os.getenv("OPENCV_IO_MAX_IMAGE_PIXELS", 1100000000000))
        im1 = Image.open(sliced_jpg_image_path)
        ## save a image using extension
        print(im1.format, im1.size, im1.mode)
        im1 = im1.convert('RGBA')
        r, g, b, alpha = im1.split()
        im1 = Image.merge("RGB", (r, g, b))
        im1 = im1.save(sliced_jpg_image_path)

        return sliced_jpg_image_path
