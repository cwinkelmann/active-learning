"""
Dataset for Image Classifiction Based on the number of objects in the image, when zero objects exist the image is classified as empty
if there 1 or more it is classified as not empty
"""
import copy
import sys

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

import albumentations as A
from PIL import Image
from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="WARNING")


from com.biospheredata.image.augmentations import crop_augment_crop, get_image_by_id, \
    get_by_id, LabelMissingException, crop_augment_empty, TranslationWithinBoundsNotPossible
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, \
    convert_HastyAnnotationV2_to_HastyAnnotationV2flat, HastyAnnotationV2_flat, hA_from_file
from com.biospheredata.ml.datasets.helper import imshow



def collate_fn(batch):
    """ collation function to be used with the pytorch DataLoader """
    images = []
    boxes = []

    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['bboxes'])

    images = torch.stack(images, dim=0)

    return images, boxes

# def collate_fn(batch):
#     return tuple(zip(*batch))

class ImageClassificationDataset(Dataset):
    """maps to each image the number of objects in it
    # TODO do not forget Zoom Augmentations, perspective, shear, mosaic, mixup, seamline, cutmix, smear
    """
    def __init__(self,
                 ann: HastyAnnotationV2,
                 images_path: Path,
                 transform=None,
                 preprocess=None,
                 grid_size=224,
                 initial_offset = 300,
                 shuffle=True):
        """

        :param images:
        :param images_path:
        :param transform: final transformation to be applied to the image before it is fed to the model
        :param preprocess: a full image isually is not suitable for training, so we need to preprocess it, ie. crop it to a smaller frame, ratate, shear, modify it
        """
        # self.df = HastyConverter.convert_to_regression_df(images)
        self.ann = ann
        self.flat_annotations = convert_HastyAnnotationV2_to_HastyAnnotationV2flat(project=ann)
        self.df = pd.DataFrame([a.dict() for a in self.flat_annotations])

        # self.images = images
        self.images_path = images_path
        self.class_names = ["No", "Yes"]
        self.transform = transform
        self.preprocess = preprocess
        self.shuffle = shuffle
        self.grid_size = grid_size
        self.initial_offset = initial_offset

    def __len__(self):

        label_count = [len(l.labels) for l in self.ann.images]
        return int(sum(label_count) * 2) # In the balanced case we want to have the same number of empty and non-empty images. If there 100 labels we have a dataset size of 200
        # return 100 # TODO find a good number
        # return sum(label_count) // 2
        # return len(self.ann.images * 10) # TODO find a good number

    def __getitem__(self, idx):
        """ Select an image crop it around the bounding box, apply augmentations and return it """

        n = 0
        final_box_size = self.grid_size # TODO pass this
        while True:

            annotation = self.flat_annotations[idx % len(self.flat_annotations)]

            assert isinstance(annotation,
                              HastyAnnotationV2_flat), f"Expected HastyAnnotationV2_flat, got {type(annotation)}"

            h_image = get_image_by_id(image_id=annotation.image_id, images=self.ann.images)
            h_image = copy.deepcopy(h_image)
            raw_image = Image.open(self.images_path / h_image.dataset_name / h_image.image_name)
            select_image_label = get_by_id(annotation.label_id, h_image.labels)
            logger.info(f"image_name: {annotation.image_name}")
            if idx < len(self.flat_annotations):
                try:

                    image, imageLabels = crop_augment_crop(selected_image_label=select_image_label,
                                                           image=raw_image,
                                                           labels=h_image.labels,
                                                           offset=self.initial_offset,
                                                           final_box_size=final_box_size,
                                                           albumentations_transforms=self.transform,
                                                           visualise=False
                                                           )
                    logger.info(f"getting occupied image with {len(imageLabels)} labels")
                except (LabelMissingException, TranslationWithinBoundsNotPossible) as e:
                    n += 1
                    logger.warning(f"LabelMissingException: {e} for annotation {annotation}")
                    if n > 100:
                        raise ValueError("Too many attempts to get a label")

                    # TODO add the augmentations to the empty images
                    # take an empty image
                    image = crop_augment_empty(image=raw_image,
                                                            final_box_size=final_box_size,
                                                            labels=h_image.labels,
                                                            transforms=self.transform)
                    imageLabels = []
            else:
                logger.info(f"getting empty image")
                # take an empty image
                image = crop_augment_empty(image=raw_image,
                                                        final_box_size=final_box_size,
                                                        labels=h_image.labels,
                                                        transforms=self.transform)
                imageLabels = []


            logger.info(f"image size: {image.size}, imageLabels: {imageLabels}")

            assert isinstance(image, Image.Image), f"Expected Image.Image, got {type(image)}"

            # Convert the PIL image to a NumPy array and then to normalized Tensor. For Debugging purposes this
            # is not done in the transform function
            # img_tensor = tv_tensors.Image(image)
            image = np.asarray(image)
            image = np.copy(image)
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
            image_tensor.size()
            # tv_bboxes = tv_tensors.BoundingBoxes([i.bbox for i in imageLabels], format="XYXY", canvas_size=(final_box_size, final_box_size))

            # TODO look at https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html There they return the bounding boxes and it works. I am getting unmatchings size exceptions.
            # tv_bboxes = tv_tensors.BoundingBoxes([i.bbox for i in imageLabels], format="XYXY",
            #                                      canvas_size=F.get_size(image_tensor))


            target = {
                "image": normalized_tensor,
                "object_num": len([i.bbox for i in imageLabels]),
                # "bboxes": imageLabels, # TODO I need to modify the collation to get this right
                "class_name": "iguana_yes" if len([i.bbox for i in imageLabels]) > 0 else "iguana_no"
            }
            return target





if __name__ == '__main__':
    for i in range(4):

        labels_path = Path("/Users/christian/data/training_data/iguanas_2024_06_17_FMO02")
        input_path_train = labels_path / "train"
        input_path_train.mkdir(exist_ok=True)

        output_path_augmented = labels_path / "test"

        hA = hA_from_file(input_path_train / "hasty_format.json")

        albumentation_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Blur(blur_limit=3, p=0.3),
            A.Affine(shear={"x": 20}, p=0.1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(p=0.2),
        ],
            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['category_ids'])
        )
        # TODO readd the albumentations
        # albumentation_transforms = None

        iguana_ds = ImageClassificationDataset(ann=hA,
                                               images_path=input_path_train,
                                               shuffle=False,
                                               transform=albumentation_transforms,
                                               grid_size=448,
                                               initial_offset=300
                                               )

        # TODO write a custom dataloader
        dataloader = DataLoader(iguana_ds,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0,

                                # collate_fn=collate_fn # TODO use this collate function to work properly with bounding boxes.
                                )
        nkl = 0
        for dd in dataloader:

            print("Batch of images has shape: ", dd["image"].shape)
            print("Batch of labels has shape: ", dd["object_num"].shape)
            print("Batch of class_name has shape: ", dd["class_name"])
        #     print(i_batch,
        #           len(sample_batched['class_name']))
            #img_tensor = torch.from_numpy(image)

            grid = torchvision.utils.make_grid(dd["image"])
            # if nkl == 3:
            #     imshow(grid, title=[int(x) for x in dd["object_num"]])
            #     # imshow(grid, title=[x for x in dd["class_name"]])
            #
            #     # break
            # nkl += 1

            imshow(grid, title=[int(x) for x in dd["object_num"]], filename=f"iguana_{i}.png")
            # imshow(grid, title=[x for x in dd["class_name"]])

        print("Done")