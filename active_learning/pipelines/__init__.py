from pathlib import Path

from com.biospheredata.converter.HastyConverter import HastyConverter, hasty_filter_pipeline
from com.biospheredata.helper.image_annotation.annotation import create_regular_raster_grid
from com.biospheredata.image.image_manipulation import crop_out_images_v2, pad_to_multiple
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2