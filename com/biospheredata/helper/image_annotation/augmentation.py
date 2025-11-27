"""
a script to generate augmentations from the images and bounding boxes
https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/


"""
import glob

import pandas as pd
from PIL import Image
import numpy as np
from loguru import logger

from com.biospheredata.visualization.visualize_result import visualize_bounding_boxes
from pathlib import Path

import albumentations as A


def augment_images_from_filelist(filelist: list,
                                 transformations,
                                 visualised_path,
                                 output_folder_path_new,
                                 n_augmentations):
    """
    augment images from a list of files and list transformations
    :param filelist:
    :return:
    """
    i = 0
    n = len(filelist)
    path_dataset = {}

    for imname in filelist:
        try:
            list_augmented_image_path = []
            list_path_to_label = []
            list_visualised_images_path = []

            print(f"file: {i}/{n} {imname} )")
            i += 1
            pillow_image = Image.open(imname)
            image = np.array(pillow_image)

            for i_aug in range(n_augmentations):
                label_path = list(Path(imname).with_suffix(".txt").parts)
                label_path = Path(*label_path)

                with open(label_path) as f:
                    bboxes = pd.read_csv(label_path, sep=" ", names=["class", "x", "y", "w", "h"])
                    iguana_count = bboxes.groupby('class').size()[0]

                    bboxes2 = bboxes[["x", "y", "w", "h"]]
                    bboxes2 = bboxes2.values.tolist()

                    category_ids = bboxes[["class"]].values.flatten().tolist()

                transformed = transformations(image=image, bboxes=bboxes2, category_ids=category_ids)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_category_ids = transformed['category_ids']

                im = Image.fromarray(transformed_image)

                augmented_image_path = f"{Path(imname).stem}_aug{i_aug}{Path(imname).suffix}"
                augmented_image_path = output_folder_path_new.joinpath(augmented_image_path)
                im.save(f"{augmented_image_path}", "JPEG")
                # logger.info(f"wrote augmented image to {augmented_image_path}")

                df_transformed_bboxes = pd.DataFrame(transformed_bboxes, columns=["x", "y", "w", "h"])
                df_transformed_bboxes["class"] = transformed_category_ids

                ## write augmented label to disk
                path_to_label = Path(augmented_image_path).with_suffix(".txt")
                # path_to_label = augmented_image_path.parent.joinpath(
                #     Path(f"{path_to_label.stem}_{i_aug}{path_to_label.suffix}"))

                df_transformed_bboxes[["class", "x", "y", "w", "h"]].to_csv(path_to_label,
                                                                            header=False, index=False, sep=' ')

                if visualised_path:
                    (Path(visualised_path)).mkdir(parents=True, exist_ok=True)
                    visualised_images_path = visualize_bounding_boxes(
                        imname=Path(augmented_image_path).name,
                        imarray=transformed_image,
                        label_name=path_to_label.name,
                        basepath=augmented_image_path.parent,
                        # suffix_images="images",
                        # suffix_labels="labels",
                        output_path=visualised_path)
                else:
                    visualised_images_path = visualised_path

                # logger.info(path_to_label)

                list_augmented_image_path.append(augmented_image_path)
                list_path_to_label.append(path_to_label)
                list_visualised_images_path.append(visualised_images_path)

            paths = {
                "augmented_image": list_augmented_image_path,
                "augmented_label": list_path_to_label,
                "augmented_visualised_image": list_visualised_images_path
            }

            ## TODO write this for each augmentation
            path_dataset[str(imname)] = paths

        except Exception as e:
            ## TODO very often this exception is causes the bounding box is so small
            logger.warning(e)

    return path_dataset


def augment_images(input_images_path,
                   input_labels_path,
                   output_folder_path_new,
                   visualised_path=None,
                   n_augmentations=5):
    """

    :param input_images_path:
    :param input_labels_path:
    :param output_folder_path_new:
    :return:
    """


    transform = A.Compose([
        # A.RandomCrop(width=500, height=500, p=0.9),
        A.HorizontalFlip(p=0.5),
        #A.Blur(p=0.5, blur_limit=(3, 18)),
        #A.ChannelShuffle(p=0.5),
        #A.CLAHE(p=0.5)
    ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.5))

    # category_ids = [0, 0, 0]
    (Path(output_folder_path_new)).mkdir(parents=True, exist_ok=True)

    filelist = glob.glob(str(input_images_path) + '/*.JPG', recursive=True)

    path_dataset = augment_images_from_filelist(
        filelist=filelist, transformations=transform,
        visualised_path=visualised_path, output_folder_path_new=output_folder_path_new,
        n_augmentations=n_augmentations
    )

    return path_dataset


if __name__ == "__main__":
    n_augmentations = 2
    # category_id_to_name = {0: 'iguana'}
    input_images_path = "/Users/christian/data/training_data/iguanas_2023_04_04_FMO01_16/tiled/images_with_objects_subset"
    input_labels_path = "/Users/christian/data/training_data/iguanas_2023_04_04_FMO01_16/tiled/images_with_objects_subset"
    output_folder_path_new = Path(
        "/Users/christian/data/training_data/iguanas_2023_04_04_FMO01_16/tiled/augmented_images_with_objects")
    visualised_path = "/Users/christian/data/training_data/iguanas_2023_04_04_FMO01_16/tiled/visualized_augmented_images_with_objects"

    n = 0
    while n < n_augmentations:
        augment_images(input_images_path, input_labels_path, output_folder_path_new, visualised_path=visualised_path,
                       n_augmentations=n_augmentations)
        n=n+1
