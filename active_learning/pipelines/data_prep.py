import multiprocessing

import copy
import multiprocessing
import typing
from loguru import logger
from pathlib import Path

from active_learning.config.dataset_filter import DatasetFilterConfig
from active_learning.filter import ImageFilter
from active_learning.types.Exceptions import LabelInconsistenyError
from active_learning.util.converter import coco2hasty, hasty2coco
from active_learning.util.image_manipulation import crop_by_regular_grid, crop_by_regular_grid_two_stage
from com.biospheredata.converter.HastyConverter import HastyConverter, hasty_filter_pipeline, unzip_files
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2, AnnotatedImage
from com.biospheredata.types.serialisation import save_model_to_file


def process_image(args):
    """
    Create a regular grid and crop the images
    """
    train_images_output_path, empty_fraction, crop_size, full_images_path_padded, i, images_path, overlap, visualise_path, edge_black_out, annotated_types = args
    try:
        images, cropped_images_path = crop_by_regular_grid(
            crop_size,
            full_images_path_padded,
            i,
            images_path,
            overlap=overlap,
            train_images_output_path=train_images_output_path,
            empty_fraction=empty_fraction,
            edge_black_out=edge_black_out,
            visualisation_path=visualise_path,
            annotated_types=annotated_types
        )
        return images, cropped_images_path
    except LabelInconsistenyError as e:
        logger.warning(f"Label inconsistency in image {i.image_name}, skipping this image.")
        logger.warning(e)
        return [], []
    except OSError as e:
        logger.error(f"OSError image {i.image_name}, skipping this image.")
        logger.error(e)
        return [], []



class UnpackAnnotations(object):
    """
    Unpack the data from various data sources, like hasty, iSAID, deeporest
    """

    def __init__(self, labels_path=None, images_path=None):
        # Initialize the backing attribute
        self._labels_path = labels_path
        self._images_path = images_path

    @property
    def labels_path(self):
        return self._labels_path

    @property
    def images_path(self):
        return self._images_path

    @labels_path.setter
    def labels_path(self, value):
        """Setter for the labels_path property."""
        if not isinstance(value, str):
            raise ValueError("labels_path must be a string")
        self._labels_path = value

    def get_hasty_annotations(self):
        return self.hA

    def get_coco_annotations(self):
        """ TODO impement """
        raise NotImplementedError("Method not implemented")
        return self.coco_data

    def unzip_hasty(self, hasty_annotations_labels_zipped: Path,
                    hasty_annotations_images_zipped: Path,
                    output_path: Path = None) -> typing.Tuple[HastyAnnotationV2, Path]:
        """
        Unzip the hasty annotations
        :param hasty_annotations_labels_zipped:
        :param hasty_annotations_images_zipped:
        :return:
        """
        # check if the both files exist
        assert hasty_annotations_labels_zipped.exists(), f"File not found: {hasty_annotations_labels_zipped}"
        assert hasty_annotations_images_zipped.exists(), f"File not found: {hasty_annotations_images_zipped}"

        if output_path is None:
            output_path = hasty_annotations_labels_zipped.parent

        hA, images_path = HastyConverter.from_zip(output_path=output_path,
                                                  hasty_annotations_labels_zipped=hasty_annotations_labels_zipped,
                                                  hasty_annotations_images_zipped=hasty_annotations_images_zipped
                                                  )
        return hA, images_path


from enum import Enum


class AnnotationFormat(Enum):
    HASTY = "hasty"
    COCO = "coco"
    YOLO = "yolo"
    DEEPFOREST = "deepforest"
    TRAPPER = "trapper"


class AnnotationsIntermediary(object):

    def __init__(self):
        self.dataset_name = None
        self.images_path = None
        self.hasty = None
        self.type = None

    def set_coco_annotations(self, coco_data, images_path, project_name, dataset_name):
        self.data = coco_data
        self.type = "coco"
        self.hA = coco2hasty(coco_data=coco_data, images_path=images_path,
                             project_name=project_name, dataset_name=dataset_name)

    def set_hasty_annotations(self, hA):
        self.hA = hA

    def get_hasty_annotations(self):
        return self.hA

    def add_images_path(self, images_path):
        logger.warning("not implemented yed")

    def get_deepforest_annotations(self, output_path):
        HastyConverter.convert_deep_forest(self.hA, output_file=output_path / "deep_forest_format.csv")
        logger.info(f"DeepForest annotations saved to {output_path / 'deep_forest_format.csv'}")

    def to_YOLO_annotations(self, output_path):

        # df_annotations = HastyConverter.convert_to_yolo(self.hA, type="box")

        yolo_boxes_path = output_path / "yolo_boxes"
        yolo_segments_path = output_path / "yolo_segments"
        yolo_boxes_path.mkdir(exist_ok=True, parents=True)
        yolo_segments_path.mkdir(exist_ok=True, parents=True)

        class_mapping, df_all_boxes = HastyConverter.convert_to_yolo_boxes(hA=self.hA, yolo_base_path=yolo_boxes_path)

        # TODO only convert to segments if there are segments in the dataset
        # class_mapping = HastyConverter.convert_to_yolo_segments(hA=self.hA, yolo_base_path=yolo_segments_path)

        class_names = [key for key, value in sorted(class_mapping.items(), key=lambda item: item[1])]
        return class_names

    def coco(self, output_path):
        coco_annotations = hasty2coco(self.hA)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        save_model_to_file(coco_annotations, output_path)
        return output_path

    @staticmethod
    def from_hasty_zip(base_path: Path,
                       hasty_annotations_labels_zipped: str,
                       hasty_annotations_images_zipped: str):
        """
        Unzip downloaded zipped hasty json 1.1. files to base_path
        :param base_path:
        :param hasty_annotations_labels_zipped:
        :param hasty_annotations_images_zipped:
        :return:
        """

        full_hasty_annotations_images_zipped_path = base_path / hasty_annotations_images_zipped
        full_hasty_annotations_labels_zipped_path = base_path / hasty_annotations_labels_zipped

        if not full_hasty_annotations_images_zipped_path.exists():
            raise FileNotFoundError(f"File not found: {full_hasty_annotations_images_zipped_path}")
        if not full_hasty_annotations_labels_zipped_path.exists():
            raise FileNotFoundError(f"File not found: {full_hasty_annotations_labels_zipped_path}")

        ## unzip the files
        images_path = base_path / HastyConverter.IMAGES_PATH
        unzip_files(base_path / hasty_annotations_images_zipped, images_path)
        unzip_files(base_path / hasty_annotations_labels_zipped, base_path / HastyConverter.ANNOTATION_PATH)

        for idx, annotation_file in enumerate(
                HastyConverter.get_unzipped_label_files(base_path / HastyConverter.ANNOTATION_PATH)):
            if idx > 0:
                raise ValueError("We only support one annotation file at the moment")

        return HastyConverter.from_folder(hA_path=annotation_file)

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name


class DataprepPipeline(object):
    """
    process annotations by filtering, tiling and some augmentations
    """
    dataset_filter = None
    tag_filter = None
    augmented_images = typing.Optional[typing.List[AnnotatedImage]]
    augmented_images_path = typing.Optional[typing.List[Path]]
    empty_fraction = False
    images_filter_func: typing.List[typing.Callable]  = []
    grid_manager: typing.Callable
    _image_type = "jpg"
    num_labels: int = 0

    def __init__(self,
                 annotations_labels: HastyAnnotationV2,
                 images_path: Path,
                 output_path: Path,
                 crop_size=512,
                 overlap=0,
                 images_fitler=None,
                 class_filter=None,
                 status_filter=None,
                 annotation_types=None,
                 empty_fraction= False,
                 num_labels=None,
                 config: typing.Optional[DatasetFilterConfig] = None):
        """

        :param annotations_labels:
        :param images_path:
        :param output_path:
        :param crop_size:
        :param overlap: between 0 and size of the crop, 0 means no overlap
        :param images_fitler:
        :param class_filter:
        :param status_filter:
        :param annotation_types:
        """

        self.num = None
        self.sample_strategy = None
        self.crop_size = crop_size
        self.overlap = overlap

        self.hA = annotations_labels
        self.hA_filtered = None
        self.hA_crops = None
        # self.images_path = None
        self.rename_dictionary = None
        self.output_path = output_path
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.train_images_output_path = output_path / f"crops_{self.crop_size}"
        self.train_images_output_path.mkdir(exist_ok=True, parents=True)
        self.visualise_path = None
        self.images_filter = images_fitler
        self.images_exclude = None
        self.class_filter = class_filter
        self.status_filter = status_filter
        self.annotation_types = annotation_types
        self.tag_filter = None
        self.num_labels = num_labels

        self.images_path = images_path
        self.empty_fraction = empty_fraction

        self.edge_black_out = False
        self.use_multiprocessing = True

        self.flat_images_path = None
        self.config = config

    def run(self, flatten=True):
        self.run_filter(flatten=flatten)

        if self.config.crop:
            hA_crop = self.run_crop()

            return hA_crop
        else:
            return self.hA_filtered

    def run_filter(self, flatten=True):
        """
        Run the data pipeline

        flatten - if True, flatten the images to a single folder

        :return:
        """
        if self.hA is None:
            raise ValueError("HastyAnnotationV2 object is not set. Please provide it before running the pipeline.")
        # order the images from the image with the least labels to the most labels
        self.hA.images = sorted(self.hA.images, key=lambda image: len(image.labels))

        hA = hasty_filter_pipeline(
            hA=self.hA,
            class_filter=self.class_filter,
            status_filter=self.status_filter,
            dataset_filter=self.dataset_filter,
            images_filter=self.images_filter,
            images_exclude=self.images_exclude,
            image_tags=self.tag_filter,
            annotation_types=self.annotation_types,
            num_images=self.num,
            sample_strategy=self.sample_strategy
        ) # TODO implement this with the images_filter_func


        if len(self.images_filter_func) != 0:
            for i in self.images_filter_func:
                hA = i(hA)

        if flatten:
            # flatten the dataset to avoid having subfolders and therefore quite some confusions later.
            ## FIXME: These are not parts of the annotation conversion pipeline but rather a part of the data preparation right before training.md & copying so many images here is a waste of time
            # TODO readd this
            default_dataset_name = "Default"
            flat_images_path = self.output_path.joinpath(default_dataset_name)
            hA_flat = HastyConverter.copy_images_to_flat_structure(hA=hA,
                                                                   base_bath=self.images_path,
                                                                   folder_path=flat_images_path)

            # Switch the image names to the ds_image_name
            for v in hA_flat.images:
                v.dataset_name = default_dataset_name
                v.image_name = f"{v.ds_image_name}"

            self.hA_filtered = copy.deepcopy(hA_flat)
        else:
            self.hA_filtered = copy.deepcopy(hA)
            flat_images_path = self.images_path
            self.flat_images_path = flat_images_path

        if self.rename_dictionary is not None:
            for k, v in self.rename_dictionary.items():
                self.hA_filtered.rename_label_class(k, v)



    def run_crop(self):
        assert self.flat_images_path is not None, "Please run the pipeline first to get the flat images path."
        hA_crop = self.data_crop_pipeline(crop_size=self.crop_size,
                                          overlap=self.overlap,
                                          hA=copy.deepcopy(self.hA_filtered),
                                          images_path=self.flat_images_path,
                                          edge_black_out=self.edge_black_out,
                                          use_multiprocessing=self.use_multiprocessing)

        self.hA_crops = hA_crop
        return hA_crop

    def data_crop_pipeline(self,
                           overlap: int,
                           crop_size: int,
                           hA: HastyAnnotationV2,
                           images_path: Path,
                           use_multiprocessing: bool = True,
                           edge_black_out=True,
                           ):
        """
        crop the images by a regular grid
        :param edge_black_out:
        :param use_multiprocessing:
        :param images_path:
        :param stage:
        :param overlap:
        :param crop_size:
        :param hA:
        :return:
        """

        # TODO use a temporary folder for this
        # TODO add the grid externally

        full_images_path_padded = self.output_path / "padded_images"
        full_images_path_padded.mkdir(exist_ok=True, parents=True)

        all_images: typing.List[AnnotatedImage] = []
        all_images_path: typing.List[Path] = []

        # TODO: make a cropping function out of this

        all_images = []
        all_images_path = []

        if use_multiprocessing:
            # train_images_output_path, empty_fraction, crop_size, full_images_path_padded, i, images_path, overlap, visualise_path, edge_black_out
            args_list = [(self.train_images_output_path, self.empty_fraction, crop_size,
                          full_images_path_padded, i, images_path, overlap, self.visualise_path, edge_black_out, self.annotation_types) for i in hA.images]

            # Use multiprocessing pool
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(process_image, args_list)

            # Collect results
            for annotated_images, cropped_images_path in results:
                all_images.extend(annotated_images)
                all_images_path.extend(cropped_images_path)

        else:
            for i in hA.images:

                # TODO split these steps: create a grid first, then crop
                # df_grid = self.grid_manager(i)
                # TODO implement this
                # annotated_images, cropped_images_path = self.cropper(df_grid, i, images_path, full_images_path_padded)
                try:
                    if len(self.annotation_types) == 2:
                        annotated_images, cropped_images_path = crop_by_regular_grid_two_stage(
                            crop_size=crop_size,
                            full_images_path_padded=full_images_path_padded,
                            i=i,
                            images_path=images_path,
                            overlap=overlap,
                            train_images_output_path=self.train_images_output_path,
                            empty_fraction=self.empty_fraction,
                            edge_black_out=edge_black_out,
                            visualisation_path=self.visualise_path,
                            annotated_types=self.annotation_types

                        )
                    else:
                        annotated_images, cropped_images_path = crop_by_regular_grid(
                            crop_size=crop_size,
                            full_images_path_padded=full_images_path_padded,
                            i=i,
                            images_path=images_path,
                            overlap=overlap,
                            train_images_output_path=self.train_images_output_path,
                            empty_fraction=self.empty_fraction,
                            edge_black_out=edge_black_out,
                            visualisation_path=self.visualise_path,
                            annotated_types=self.annotation_types

                            # TODO pass the grid manager here
                            # grid_manager=self.grid_manager
                        )
                except LabelInconsistenyError as e:
                    logger.warning(f"Label inconsistency in image {i.image_name}, skipping this image.")
                    logger.warning(e)
                    annotated_images = []
                    cropped_images_path = []

                all_images.extend(annotated_images)
                all_images_path.extend(cropped_images_path)

        hA.images = all_images
        hA_crops = HastyAnnotationV2(
            project_name="crops",
            images=all_images,
            export_format_version="1.1",
            label_classes=hA.label_classes,
        )

        self.augmented_images = all_images
        self.augmented_images_path = all_images_path

        return hA_crops

    def get_hA_filtered(self) -> HastyAnnotationV2:
        return self.hA_filtered

    def get_hA_crops(self) -> HastyAnnotationV2:
        return self.hA_crops

    def get_augmented_images(self):
        return self.augmented_images

    def get_images(self) -> typing.List[Path]:
        return self.augmented_images_path
        # return self.output_path.glob(f"*.{self._image_type}")

    def get_stats(self):
        return {
            "images": len(self.augmented_images),
            "labels": sum([len(i.labels) for i in self.augmented_images])
        }

    def set_images_num(self, num, sample_strategy):
        self.num = num
        self.sample_strategy = sample_strategy

    def add_images_filter_func(self, ifcn_att: ImageFilter):
        assert isinstance(ifcn_att, ImageFilter)
        self.images_filter_func.append( ifcn_att )
