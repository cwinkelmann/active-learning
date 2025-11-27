"""
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Datatset describes the data itself

the dataloader is a helper to iterate, batch, shuffle, and load the data

install:
pip install pillow scikit-image matplotlib pandas torch torchvision albumentations shapely pydantic

"""
import typing

# Where do I put transformations?

import os

from pathlib import Path

import PIL
from PIL.Image import Image as PILImage
import torch
import pandas as pd
import torchvision
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Compose, ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Ignore warnings
import warnings

from com.biospheredata.image.augmentations import get_by_id, custom_transforms
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, ImageLabel


warnings.filterwarnings("ignore")


# TODO write a custom dataset
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

torchvision_transform = Compose([
    # ToPILImage(),  # Necessary if starting with a PIL Image
    transforms.ToTensor(),

    transforms.Normalize(mean=mean, std=std)# Can add more torchvision transforms here if needed
])

class IguanaHastyDataset(Dataset):
    """
    Iguanas from Hasty Annotation dataset.
    """

    def __init__(self, ann_file: Path,
                 root_dir: Path,
                 empty_prob: float = 0.8,
                 pre_transform=None,
                 transform=None):
        """
        Arguments:
            ann_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            imagez (int): size of the image which it outputed to the network
        """
        self.iguanas_annotation = hA_from_file(ann_file)
        self.root_dir = root_dir
        self.empty_prob = empty_prob
        self.pre_transform = pre_transform
        self.transform = transform

    def __len__(self):
        return len(self.iguanas_annotation.images)

    def __getitem__(self, idx) -> (np.ndarray, [ImageLabel]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get an image name from the annotation
        img_name = os.path.join(self.root_dir, self.iguanas_annotation.images[idx].dataset_name,
                                self.iguanas_annotation.images[idx].image_name)
        image = io.imread(img_name)
        bboxes = self.iguanas_annotation.images[idx].labels

        if self.pre_transform:
            image, bboxes = self.pre_transform(image, bboxes)

        # boxes = [l.bbox for l in self.iguanas_annotation.images[idx].labels]
        # get with and height
        # labels = [l.class_name for l in self.iguanas_annotation.images[idx].labels]
        # bboxes = [x + [y] for x, y in zip(boxes, labels)]

        #

        # apply the transform
        if self.transform:
            # TODO apply the transform but in not so crappy
            image, bboxes = self.transform(image, bboxes)

            if isinstance(image, PILImage):
                image = np.array(image)

        image = torchvision_transform(image)
            ## if the sample is not containing any bounding boxes redo the samping
        # TODO does it work like that??
        bboxes = [l.bbox for l in bboxes]
        #bboxes = torch.from_numpy(np.array(bboxes))
        #bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYWH", canvas_size=(640, 640))

        # TODO QuickFix
        if len(bboxes) == 0:
            class_name=0
        else:
            class_name = 1

        # img_tensor = torch.from_numpy(image)
        # img_tensor = img_tensor.permute(2, 0, 1)
        sample = {'image': image, 'class_name': class_name} # TODO prevent the error with the different length of the bboxes
        return sample


# if you are using Windows, uncomment the next line and indent the for loop.
# you might need to go back and change ``num_workers`` to 0.



def show_iguanas_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, class_name_batch = sample_batched['image'], sample_batched['class_name']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy())


    plt.title('Batch from dataloader')
    plt.show()

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    labels_path = Path("/Users/christian/data/training_data/iguanas_2024_03_23")
    input_path_train = labels_path / "train"
    input_path_train.mkdir(exist_ok=True)

    output_path_augmented = labels_path / "train"

    # hA = hA_from_file(input_path_train / "hasty_format.json")

    iguana_ds = IguanaHastyDataset(ann_file=input_path_train / "hasty_format.json",
                                   root_dir=input_path_train,
                                   # transform=transforms.Compose([
                                   #     RandomCrop(224),
                                   #     ToTensor()
                                   # ])
                                   pre_transform=None,
                                   transform=custom_transforms
                                   # transform=None
                                   )

    # TODO write a custom dataloader
    dataloader = DataLoader(iguana_ds,
                            batch_size=16,
                            shuffle=True,
                            num_workers=0)

    for dd in dataloader:
        print("Batch of images has shape: ", dd["image"].shape)
        print("Batch of labels has shape: ", dd["class_name"].shape)
    #     print(i_batch,
    #           len(sample_batched['class_name']))

        out = torchvision.utils.make_grid(dd["image"])
        imshow(out, title=[int(x) for x in dd["class_name"]])

        break
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_iguanas_batch(sample_batched)
            plt.axis('off')
            plt.show()
            break





    # Get a batch of training data
    #inputs, classes = next(iter(dataloader))

    # Make a grid from batch
    #out = torchvision.utils.make_grid(inputs["image"])

    #imshow(out, title=None)