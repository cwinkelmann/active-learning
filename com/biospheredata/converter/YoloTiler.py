"""
https://towardsdatascience.com/tile-slice-yolo-dataset-for-small-objects-detection-a75bf26f7fa2
https://github.com/slanj/yolo-tiling

python3 tyle_yolo.py -source ./yolosample/ts/ -target ./yolosliced/ts/ -ext .JPG -size 512


"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import os
import random
from loguru import logger

from com.biospheredata.helper.image_annotation.augmentation import augment_images_from_filelist
from com.biospheredata.helper.random_split import generate_k_folds, get_test_set
from com.biospheredata.training.YoloDataYaml import YoloDataYaml
from com.biospheredata.types.GeoreferencedImage import GeoreferencedImage
from com.biospheredata.types.exceptions.NotEnoughImagesException import NotEnoughImagesException


class YoloTiler(object):
    """
    cut tiles from bigger images to prepare training data
    """

    IMAGES_WITH_OBJECTS = "images_with_objects"
    IMAGES_WITHOUT_OBJECTS = "images_without_objects"

    def __init(self, image: GeoreferencedImage):
        self.image = image

    @staticmethod
    def tile_folders(images_path: Path, labels_path: Path, tiled_path: Path, slice_size=1280, extension="JPG"):
        """
        create tiles of images and reproject the yolo labels

        @param images_path:
        @param labels_path: path to the yolo labels
        @param tiled_path:
        @param slice_size:
        @param extension:
        @return:
        """
        images = set(images_path.glob(f"*.{extension}")) - set(images_path.glob(f"._*"))
        if len(images) == 0:
            raise NotEnoughImagesException(f"not enough images for extension: {extension}")
        tiled_path.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            try:
                label_path = labels_path.joinpath(image_path.with_suffix('.txt').name)

                im = Image.open(image_path)
                imr = np.array(im, dtype=np.uint8)
                height = imr.shape[0]
                width = imr.shape[1]

                boxes = YoloTiler.tile(
                    height=height,
                    width=width,
                    label_path=label_path
                )

                images_without_objects, images_with_objects, new_labels_paths = YoloTiler.slice_boxes(
                    height=height,
                    width=width,
                    slice_size=slice_size,
                    newpath=tiled_path.joinpath(YoloTiler.IMAGES_WITH_OBJECTS),
                    falsepath=tiled_path.joinpath(YoloTiler.IMAGES_WITHOUT_OBJECTS),
                    imname=image_path,
                    imr=imr,
                    boxes=boxes,
                    ext=f".{extension}"
                )

                yield (images_without_objects, images_with_objects, new_labels_paths)
            except FileNotFoundError:

                logger.warning(f"File label_path {label_path} not found!")

                im = Image.open(image_path)
                imr = np.array(im, dtype=np.uint8)
                height = imr.shape[0]
                width = imr.shape[1]

                boxes = []

                images_without_objects, images_with_objects, new_labels_paths = YoloTiler.slice_boxes(
                    height=height,
                    width=width,
                    slice_size=slice_size,
                    newpath=tiled_path.joinpath(YoloTiler.IMAGES_WITH_OBJECTS),
                    falsepath=tiled_path.joinpath(YoloTiler.IMAGES_WITHOUT_OBJECTS),
                    imname=image_path,
                    imr=imr,
                    boxes=boxes,
                    ext=f".{extension}"
                )

                yield (images_without_objects, [], [])

    @staticmethod
    def tile(height: int, width: int, label_path: Path) -> list[tuple]:
        """
        read the yolo label txt file and create a list of boxes from it.
        convert x1 in %, y1 in %, h in % and w in % to total image coordinates.

        @param label_path: path to the yolo
        @param width: width of the image
        @type height: height of the image

        """

        #labname = self.image.image_path.replace(ext, '.txt')
        #labname = labname.replace("/images/", '/labels/')

        labels = pd.read_csv(label_path, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height

        boxes = []

        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w'] / 2
            y1 = (height - row[1]['y1']) - row[1]['h'] / 2
            x2 = row[1]['x1'] + row[1]['w'] / 2
            y2 = (height - row[1]['y1']) + row[1]['h'] / 2

            boxes.append(
                (
                    int(row[1]['class']),
                    Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                )
            )

        return boxes



    @staticmethod
    def simple_images_slice(slice_size, images_paths, newpath: Path, ext=".JPG"):
        """
        slice a set of images into smaller pieces

        @param slice_size:
        @param image_path:
        @param newpath:
        @param ext:
        @return:
        """
        image_slices = []
        for image_path in images_paths:

            sl = YoloTiler.simple_image_slice(
                slice_size=slice_size,
                newpath=newpath,
                image_path=image_path,
                ext=ext
            )
            for s in sl:
                image_slices.append(s)

        return image_slices

    @staticmethod
    def simple_image_slice(slice_size, image_path, newpath: Path, ext=".JPG"):
        """
        simply cut an image into equally sliced images.

        @param slice_size:
        @param falsepath:
        @param imname:
        @param imr:
        @param newpath:
        @param boxes:
        @param ext:
        @return:
        """

        with Image.open(image_path) as im:
            # im = Image.open(image_path)
            imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]

        counter = 0 ## TODO is this the right position
        images_with_objects = []
        images_without_objects = []
        labels_paths = []

        newpath.mkdir(parents=True, exist_ok=True)

        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                x1 = j * slice_size
                y1 = height - (i * slice_size)
                x2 = ((j + 1) * slice_size) - 1
                y2 = (height - (i + 1) * slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                if not imsaved and newpath:
                    # save images which don't contain any boxes
                    sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
                    sliced_im = Image.fromarray(sliced)
                    filename = image_path.parts[-1]
                    slice_path = newpath.joinpath(filename.replace(ext, f'_{i}_{j}{ext}'))

                    sliced_im = sliced_im.convert("RGB")
                    slice_path_jpg = str(slice_path).replace('.tif', f'.jpg') # TODO check this?

                    sliced_im.save(f"{slice_path_jpg}", "JPEG", quality=100, optimize=True, progressive=True)
                    logger.info(f"slice without boxes saved to {slice_path_jpg}")
                    imsaved = True

                    #yield slice_path_jpg
                    images_without_objects.append(slice_path_jpg)

        return images_without_objects


    @staticmethod
    def slice_boxes(height, width, slice_size,
                    falsepath: Path, imname, imr,
                    newpath: Path, boxes, ext):
        """

        With given coordinates of boxes in original images cutout slices will be created.
        the boxes will be sliced too

        @param height:
        @param width:
        @param slice_size:
        @param falsepath:
        @param imname:
        @param imr:
        @param newpath:
        @param boxes:
        @param ext:
        @return:
        """
        counter = 0 ## TODO is this the right position?
        images_with_objects = []
        images_without_objects = []
        labels_paths = []

        falsepath.mkdir(parents=True, exist_ok=True)
        newpath.mkdir(parents=True, exist_ok=True)

        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                x1 = j * slice_size
                y1 = height - (i * slice_size)
                x2 = ((j + 1) * slice_size) - 1
                y2 = (height - (i + 1) * slice_size) + 1

                # sliding window of the image
                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        # the annotation is in the sliding window
                        inter = pol.intersection(box[1])

                        if not imsaved:
                            sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
                            sliced_im = Image.fromarray(sliced)
                            filename = imname.name
                            slice_path = newpath.joinpath(filename.replace(ext, f'_{i}_{j}{ext}'))
                            slice_labels_path = newpath.joinpath(filename.replace(ext, f'_{i}_{j}.txt'))
                            logger.info(f"slice_path: {slice_path}")
                            # sliced_im = sliced_im.convert('CMYK') ### FIXME maybe I need this??
                            sliced_im = sliced_im.convert("RGB")
                            sliced_im.save(slice_path) ### FIXME For images with the "dem_translate ending this fails because they in some weird color space, either RGBA or CMYK"??
                            slice_path_jpg = str(slice_path).replace('.tif', f'.jpg')
                            sliced_im.save(f"{slice_path_jpg}", "JPEG", quality=95, optimize=True, progressive=True)
                            images_with_objects.append(slice_path_jpg)
                            imsaved = True

                        # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope

                        # get central point for the new bounding box
                        centre = new_box.centroid

                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy

                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size

                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / slice_size
                        new_y = (y1 - centre.coords.xy[1][0]) / slice_size

                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                # if there are labels in the slice then write the new txt to disk
                if len(slice_labels) > 0:
                    slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])

                    slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
                    labels_paths.append(slice_labels_path)

                    ## FIXME somethine is wrong here
                    # visualize_bounding_boxes(imname=slice_path_jpg,
                    #                          label_name=slice_labels_path,
                    #                          basepath=Path("/"),
                    #                          output_path=newpath)  ## TODO implement this properly
                else:
                    print(imname)
                # when the image is not already saved and empty images should be saved at all
                if not imsaved and falsepath:
                    # save images which don't contain any boxes
                    sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
                    sliced_im = Image.fromarray(sliced)
                    filename = imname.parts[-1]
                    slice_path = falsepath.joinpath(filename.replace(ext, f'_{i}_{j}{ext}'))

                    sliced_im = sliced_im.convert("RGB")
                    slice_path_jpg = str(slice_path).replace('.tif', f'.jpg')

                    sliced_im.save(f"{slice_path_jpg}", "JPEG", quality=100, optimize=True, progressive=True)
                    logger.info(f"slice without boxes saved to {slice_path_jpg}")
                    imsaved = True

                    images_without_objects.append(slice_path_jpg)

        return images_without_objects, images_with_objects, labels_paths

    @staticmethod
    def splitter(target: Path, target_upfolder: Path, ext, ratio = 0.7, max_train_images= sys.maxsize):
        """
        split images into train and validation dataset

        @deprecated
        :deprecated

        @param target:
        :param target_upfolder:
        :param ext:
        :param ratio:
        :return:
        """
        Path(target_upfolder).mkdir(exist_ok=True)
        imnames = glob.glob(f'{target}/*{ext}')
        if len(imnames) == 0:
            raise ValueError(f"No images found in {target} with file extension {ext}")
        names = [name.split('/')[-1] for name in imnames]
        random.shuffle(names)

        # split dataset for train and valid

        valid_images_path = target_upfolder.joinpath("valid/images")
        valid_labels_path = target_upfolder.joinpath("valid/labels")
        train_images_path = target_upfolder.joinpath("train/images")
        train_labels_path = target_upfolder.joinpath("train/labels")

        valid_images_path.mkdir(parents=True, exist_ok=True)
        valid_labels_path.mkdir(parents=True, exist_ok=True)
        train_images_path.mkdir(parents=True, exist_ok=True)
        train_labels_path.mkdir(parents=True, exist_ok=True)

        train = []
        valid = []
        num_train_images = 0
        for name in names:
            logger.info(f"copy images to train/valid folders")

            if random.random() < ratio and num_train_images < max_train_images:
                train.append(os.path.join(target, name))

                shutil.copyfile(target.joinpath(name), train_images_path.joinpath(name))
                name = name.replace(ext, '.txt')
                shutil.copyfile(target.joinpath(name), train_labels_path.joinpath(name))

                num_train_images += 1


            else:
                ## the rest gets into the validation set. This means if the train set is limited in size then the validation get bigger.
                valid.append(os.path.join(target, name))

                shutil.copyfile(target.joinpath(name), valid_images_path.joinpath(name))
                name = name.replace(ext, '.txt')
                shutil.copyfile(target.joinpath(name), valid_labels_path.joinpath(name))



        logger.info(f"train contains: {len(train)} images")
        logger.info(f"valid contains: {len(valid)} images")

    @staticmethod
    def splitter2(source: Path,
                  target: Path,
                  consistent_ids: list,
                  ext,
                  amount_train_images=sys.maxsize,
                  folds=5,
                  test_size=0.1,
                  training_data_location="/training_data", with_augmentations=0):
        """
        The elegant way split images into train and validation datasets using k-fold cross validation
        FIXME: this fails if I don't want to do k-fold

        :param consistent_ids: the class names which are relevant
        :param target:
        :param source:
        :param ext:
        :param ratio:
        :return:
        """
        stats = {}
        imnames = list(source.glob(f"*.{ext}"))
        imnames = [imname for imname in imnames if not str(imname.name).startswith(".")]
        if len(imnames) == 0:
            raise NotEnoughImagesException(f"No images found in {source} with file extension {ext}")
        else:
            stats["total_amount_images"] = len(imnames)
            logger.info(f"Found {len(imnames)} images in {source} ")
        # split dataset for train and valid
        df_with_object = pd.DataFrame({
            "filenames": imnames,
        })
        df_test_with, df_train = get_test_set(amount_train_images, df_with_object, test_size=test_size)
        # TODO take the old splitter here if folds == 1
        k_folds = generate_k_folds(df_train=df_train, folds=folds)

        if with_augmentations > 0:
            ## TODO generate this only if requested
            import albumentations as A
            transformations_functions =  [      # A.RandomCrop(width=400, height=400, p=0.9),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Blur(blur_limit=4, p=0.3),
                # A.ElasticTransform(),
                # A.MaskDropout((10,15), p=1),
                # A.Cutout(p=0.5, num_holes=50, max_w_size=15),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.4, alpha_coef=0.1, p=0.3),
                # A.RandomSizedCrop((300 - 200, 300 + 100), 600, 600),
                A.GaussNoise(var_limit=(100, 150), p=0.3),
                A.ShiftScaleRotate(p=0.1),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.1),
                # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5)
            ]

            transformations_functions = [
                A.HorizontalFlip(p=0.1),
                A.VerticalFlip(p=0.1),
                A.RandomBrightnessContrast(p=0.5),
                A.Blur(blur_limit=4, p=0.3),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.4, alpha_coef=0.1, p=0.3),
                A.GaussNoise(var_limit=(100, 350), p=0.3),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.2),
                A.ShiftScaleRotate(p=0.3),

            ]
            transform = A.Compose(transformations_functions,
                bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.5))

            ## Todo the whole training dataset apply augmentations

            augmented_fold_train = augment_images_from_filelist(filelist=df_train["filenames"].to_list(),
                                                                 transformations=transform,
                                                                 output_folder_path_new=source,
                                                                 n_augmentations=with_augmentations,
                                                                 visualised_path=None
                                                                 )

            k_folds = YoloTiler.append_augmented_images(k_folds, augmented_fold_train)

        n = 1
        for fold in k_folds:
            fold_path = target.joinpath(f"fold_{n}")
            # attach the test data
            fold["test"] = df_test_with

            YoloTiler.create_physical_fold(fold, fold_path)
            ydy = YoloDataYaml(class_names=consistent_ids,
                               training_data_path="train",
                               validation_data_path="valid",
                               test_data_path="test",
                               split=Path(f"{training_data_location}").joinpath(target.parts[-2]).joinpath(
                                   target.parts[-1]).joinpath(f"fold_{n}"))

            ydy.to_yaml(fold_path)

            n += 1

        # for fold in k_folds:
        #     fold_path = target.joinpath(f"fold_{n}")
        #
        #     fold["test"] = df_test_with
        #     # YoloTiler.create_physical_fold(fold, fold_path, augmented_fold_train)
        #     YoloTiler.create_nonphysical_fold(fold, fold_path, augmented_fold_train)
        #
        #     ydy = YoloDataYaml(class_names=consistent_ids,
        #                        training_data_path="train",
        #                        validation_data_path="valid",
        #                        test_data_path="test",
        #                        split=Path(f"{training_data_location}").joinpath(target.parts[-2]).joinpath(
        #                            target.parts[-1]).joinpath(f"fold_{n}"))
        #
        #     ydy.to_yaml(fold_path)
        #
        #     n += 1

        return k_folds, stats

    @staticmethod
    def create_physical_fold(fold, target):
        """
        move the images into specific folder prior to training

        @param df_test:
        @param fold:
        @param fold_number:
        @param move_file:
        @param target:
        @return:
        """
        def move_file(source, target: Path):
            """

            @param source:
            @param target:
            @return:
            """
            shutil.copyfile(source["filenames"], target.joinpath("images").joinpath(Path(source["filenames"]).parts[-1] ))
            label_name = Path(source["filenames"].with_suffix('.txt'))
            if label_name.is_file():
                shutil.copyfile(label_name,
                                target.joinpath("labels").joinpath(label_name.parts[-1]))


        ## TODO shorten this
        valid_images_path = target.joinpath(f"valid/images")
        valid_labels_path = target.joinpath(f"valid/labels")
        test_images_path = target.joinpath(f"test/images")
        test_labels_path = target.joinpath(f"test/labels")
        train_images_path = target.joinpath(f"train/images")
        train_labels_path = target.joinpath(f"train/labels")
        valid_images_path.mkdir(parents=True, exist_ok=True)
        valid_labels_path.mkdir(parents=True, exist_ok=True)
        test_images_path.mkdir(parents=True, exist_ok=True)
        test_labels_path.mkdir(parents=True, exist_ok=True)
        train_images_path.mkdir(parents=True, exist_ok=True)
        train_labels_path.mkdir(parents=True, exist_ok=True)

        ## fill the testset
        ## FIXME don't forget the testset
        fold["test"].apply(move_file, target=target.joinpath(f"test"), axis=1)
        fold["train"].apply(move_file, target=target.joinpath(f"train"), axis=1)
        fold["valid"].apply(move_file, target=target.joinpath(f"valid"), axis=1)

    @staticmethod
    def append_augmented_images(k_folds, augmented_fold_train):
        """
        instead of moving the images a txt file is created as an index

        @return:
        """
        updated_k_folds = []
        for fold in k_folds:

            ## the train fold is extended with the augmented data
            training_images = list(fold["train"]["filenames"].to_dict().values())

            ## make sure the augmented images are based on the images in the
            augmented_images_in_train = [k for k in augmented_fold_train.keys() if k in [str(x) for x in training_images]]
            images_names = []
            labels_names = []
            for i in [augmented_fold_train[k] for k in augmented_images_in_train]:
                for k in i["augmented_image"]:
                    images_names.append(k)
                    labels_names.append(k)

            pd.DataFrame(augmented_fold_train)

            fold["train"] = pd.DataFrame(pd.Series(training_images + images_names), columns=["filenames"])
            updated_k_folds.append(fold)

        return updated_k_folds



    # /home/christian/data/iguanas_2022_11_02
    # PYTHONPATH=$PYTHONPATH:../../../ python ./biospheredata.py hasty2yolo --base_path="/home/christian/data/iguanas_2022_11_02" --images=images.zip --labels=labels.zip
    @staticmethod
    def add_background_images(target_upfolder,
                              falsepath,
                              false_prob,
                              ext,
                              max_train_images=1):
        """
        @deprecated use the splitter2 to generate folds into the same folder

        Add images with no object to the dataset
        @deprecated background is just a special class and we want to make sure we limit the amount
        @param target_upfolder:
        @param falsepath:
        @param false_prob:
        @param ext:
        @return:
        """
        empty_images = []
        empty_labels = []

        images = glob.glob(str(falsepath.joinpath(f"*{ext}")))
        if len(images) == 0:
            raise ValueError(f"no background images found")
        else:
            logger.info(f"There are {len(images)} with no objects.")

        random.shuffle(images)

        for image in images:
            if false_prob:
                if random.random() <= false_prob:

                    if random.random() <= 0.7: # train test
                        train_test = "valid"
                    else:
                        train_test = "train"


                result = shutil.copyfile(image,
                                         target_upfolder.joinpath(f"{train_test}/images").joinpath(Path(image).parts[-1]))

                label_name = Path(image).parts[-1].split(".")[0]
                empty_labels.append(label_name)
                empty_label = target_upfolder.joinpath(f"{train_test}/labels").joinpath(f"{label_name}.txt")
                with open(empty_label, 'w') as f:
                    pass
                logger.info(f"moved empty image: {image} and label: {empty_label} as background to improve training: {result}.")
        logger.info(f"moved empty images to {target_upfolder}.")
        return empty_labels

