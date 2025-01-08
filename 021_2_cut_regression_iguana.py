"""
TODO this is not finished yet
This results in a dataset ready for a regression in which for each image we save the amount of iguanas in it

cutout tiles in a grid, keep images which contain iguanas and then
the same amoun of random empty images
Then we get a list of images and the associated number of object in it.

"""
from datetime import datetime

import shutil
from pathlib import Path
import random
from PIL import Image
import numpy as np


from com.biospheredata.helper.image_annotation.annotation import create_regular_raster_grid
from com.biospheredata.image.image_manipulation import crop_out_images_v2, pad_to_multiple
from com.biospheredata.types.HastyAnnotationV2 import AnnotatedImage, hA_from_file, HastyAnnotationV2
from com.biospheredata.visualization.visualize_result import visualise_polygons, visualise_image





if __name__ == "__main__":
    base_path = Path("/Users/christian/data/training_data/2024_12_09/")
    # annotation_file_name = "hasty_format_iguana.json"
    annotation_file_name = "hasty_format_iguana_point.json"
    crop_size = 512
    overlap = 250

    for dset in ["train", "val", "test"]:

        labels_path = base_path / dset
        full_images_path = labels_path
        full_images_path_padded = labels_path / "padded"
        full_images_path_padded.mkdir(exist_ok=True, parents=True)
        output_path = labels_path / f"crops_{crop_size}"
        output_path.mkdir(exist_ok=True, parents=True)

        hA = hA_from_file(labels_path / annotation_file_name)
        # hA.images = [i for i in hA.images if i.image_status == "COMPLETED"]

        all_images = []
        for i in hA.images:
            original_image_path = full_images_path / i.dataset_name / i.image_name
            padded_image_path = full_images_path_padded /  i.dataset_name / i.image_name
            padded_image_path.parent.mkdir(exist_ok=True, parents=True)
            new_width, new_height = pad_to_multiple(original_image_path, padded_image_path, crop_size, crop_size, overlap )
            grid, _ = create_regular_raster_grid(max_x=new_width,
                                                 max_y=new_height,
                                                 slice_height=crop_size,
                                                 slice_width=crop_size,
                                                 overlap=overlap)

            # axi = visualise_image(image_path=padded_image_path, show=False, title=f"grid_{i.image_name}", )
            # visualise_polygons(grid, show=True, title=f"grid_{i.image_name}", max_x=new_width, max_y=new_height, ax=axi)

            images = crop_out_images_v2(i, rasters=grid,
                                        full_images_path=full_images_path_padded,
                                        output_path=output_path, include_empty=False)



            all_images.extend(images)

            # break # TODO remove this

        hA.images = all_images
        hA_regression = HastyAnnotationV2(
            project_name="iguanas_regression",
            images=all_images,
            export_format_version="1.1",
            label_classes=hA.label_classes
        )

        with open(output_path / "hasty_format.json", mode="w") as f:
            f.write(hA_regression.model_dump_json())
