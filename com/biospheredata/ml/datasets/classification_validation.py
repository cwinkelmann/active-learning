"""

"""

import typing

import numpy as np
import random

import pandas as pd
import torch
from pathlib import Path
from matplotlib import pyplot as plt

import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from loguru import logger

from com.biospheredata.image.augmentations import crop_augment_crop, get_image_by_id, \
    get_by_id, LabelMissingException, crop_augment_empty
from com.biospheredata.ml.datasets.helper import imshow
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, \
    convert_HastyAnnotationV2_to_HastyAnnotationV2flat, HastyAnnotationV2_flat, hA_from_file, AnnotatedImage as HastyImage
from com.biospheredata.visualization.visualize_result import visualise_polygons

from com.biospheredata.helper.image_annotation.annotation import create_regular_raster_grid




class ImageClassificationGriddedDataset(Dataset):
    """Dataset for Validation Purposes. It creates a regular grid on an image and creates crops.
    # TODO do not forget Zoom Augmentations, perspective, shear, mosaic, mixup, seamline, cutmix, smear
    """
    def __init__(self,
                 ann: HastyAnnotationV2,
                 images_path: Path,
                 transform=None,
                 preprocess=None,
                 grid_size=224,
                 overlap=0,
                 balance_classes=True,
                 classes=["iguana"],
                 visualise = False):
        """

        :param images:
        :param images_path:
        :param transform: final transformation to be applied to the image before it is fed to the model
        :param preprocess: a full image isually is not suitable for training, so we need to preprocess it, ie. crop it to a smaller frame, ratate, shear, modify it
        """
        # self.df = HastyConverter.convert_to_regression_df(images)


        # self.images = images
        self.images_path = images_path
        self.crops_path = images_path / f"{grid_size}"
        self.grid_size = grid_size
        self.overlap = overlap

        self.transform = transform
        self.preprocess = preprocess
        self.balance_classes = balance_classes

        self.image_crop_labels: typing.List[Image] = []
        self.classes = classes

        self.visualise = visualise

        for i in ann.images:
            grid, _ = create_regular_raster_grid(max_x=i.width, max_y=i.height,
                                                 slice_height=grid_size, slice_width=grid_size,
                                                 overlap=overlap)

            i_crops = crop_out_images_v2(i,
                                         rasters=grid,
                                         full_images_path=self.images_path,
                                         output_path=self.crops_path,
                                         dataset_name=f"{grid_size}")

            self.image_crop_labels.extend(i_crops)

            if self.visualise:
                visualise_polygons(grid, show=True, title="grid", max_x=i.width, max_y=i.height)


        if self.balance_classes:
            # make sure there are the same number of images with and without objects
            empties = []
            with_objects = []

            for i in self.image_crop_labels:
                if len(i.labels) > 0 and len([l for l in i.labels if l.class_name in self.classes]) > 0:
                    with_objects.append(i)
                else:
                    empties.append(i)

            empties_sampled = random.sample(empties, len(with_objects))

            self.image_crop_labels = with_objects + empties_sampled

        ann.images = self.image_crop_labels

        self.ann = ann
        self.flat_annotations = convert_HastyAnnotationV2_to_HastyAnnotationV2flat(project=ann)
        self.df = pd.DataFrame([a.dict() for a in self.flat_annotations])


    def __len__(self):
        return len(self.image_crop_labels) * 2

    def __getitem__(self, idx):
        """ Select an image crop it around the bounding box, apply augmentations and return it """

        h_image = self.ann.images[idx]

        assert isinstance(h_image, HastyImage), f"Expected HastyAnnotationV2_flat, got {type(h_image)}"

        # h_image = get_image_by_id(image_id=annotation.image_id, images=self.ann.images)
        image = Image.open(self.images_path / h_image.dataset_name / h_image.image_name)

        logger.info(f"image size: {image.size}, imageLabels: {h_image.labels}")
        assert isinstance(image, Image.Image), f"Expected Image.Image, got {type(image)}"

        # Convert the PIL image to a NumPy array and then to normalized Tensor. For Debugging purposes this
        # is not done in the transform function
        # img_tensor = tv_tensors.Image(image)
        image = np.asarray(image)
        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image).float()
        # Rearrange dimensions to (C, H, W) from (H, W, C)
        image_tensor = image_tensor.permute(2, 0, 1)
        # Normalize the tensor to have values in [0, 1]
        image_tensor /= 255.0
        # Define normalization transform (mean and std for each channel, commonly used values for ImageNet)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Apply the normalization
        normalized_tensor = normalize(image_tensor)

        target = {
            "image": normalized_tensor,
            "object_num": len([i.bbox for i in h_image.labels]),
            # "bboxes": imageLabels, # TODO I need to modify the collation to get this right
            "class_name": "iguana_yes" if len([i.bbox for i in h_image.labels]) > 0 else "iguana_no"
        }
        return target



if __name__ == '__main__':
    """
    Load the annotations, create a grid, crop the images, apply augmentations and return them
    """
    labels_path = Path("/Users/christian/data/training_data/iguanas_2024_06_17_FMO02")
    input_path_train = labels_path / "train"
    input_path_train.mkdir(exist_ok=True)

    output_path_augmented = labels_path / "train"

    hA = hA_from_file(input_path_train / "hasty_format.json")

    # slightly different from the 'ImageClassificationDataset' as this one creates only a grid
    iguana_ds = ImageClassificationGriddedDataset(ann=hA,
                                                  images_path=input_path_train,
                                                  visualise=True
                                                  )


    # TODO fix the shuffle and test if an empty image is really always empty because it seemed it is not
    dataloader = DataLoader(iguana_ds,
                            batch_size=6,
                            shuffle=True,
                            num_workers=0,
                            # collate_fn=collate_fn # TODO use this collate function to work properly with bounding boxes.
                            )

    for dd in dataloader:
        print("Batch of images has shape: ", dd["image"].shape)
        print("Batch of labels has shape: ", dd["object_num"].shape)
        print("Batch of class_name has shape: ", dd["class_name"])

        grid = torchvision.utils.make_grid(dd["image"])

        imshow(grid, title=[int(x) for x in dd["object_num"]])
        # imshow(grid, title=[x for x in dd["class_name"]])

