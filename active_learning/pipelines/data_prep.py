import typing
from PIL import Image
from pathlib import Path

from com.biospheredata.converter.HastyConverter import HastyConverter, hasty_convert_pipeline
from com.biospheredata.helper.image_annotation.annotation import create_regular_raster_grid
from com.biospheredata.image.image_manipulation import crop_out_images_v2, pad_to_multiple
from com.biospheredata.types.HastyAnnotationV2 import hA_from_file, HastyAnnotationV2, AnnotatedImage


class DataprepPipeline():
    """
    process annotations by filtering, tiling and some augmentations
    """
    dataset_filter = None
    augmented_images = typing.Optional[typing.List[AnnotatedImage]]
    augmented_images_path = typing.Optional[typing.List[Path]]
    empty_fraction = False
    images_filter_func: typing.Callable = None

    _image_type = "jpg"

    def __init__(self,
                 hasty_annotations_labels_zipped,
                 hasty_annotations_images_zipped,
                 labels_path: Path,
                 output_path_train: Path,
                 stage: str,
                 crop_size=512,
                 overlap=0,
                 images_fitler=None,
                 class_filter=None,
                 status_filter=
                 ["COMPLETED"],
                 annotation_types=["points"]):

        self.crop_size = crop_size
        self.overlap = overlap
        # self.hasty_annotations_labels_zipped = hasty_annotations_labels_zipped
        # self.hasty_annotations_images_zipped = hasty_annotations_images_zipped
        self.output_path_train = output_path_train
        self.stage = stage
        self.labels_path = labels_path
        self.images_filter = images_fitler
        self.class_filter = class_filter
        self.status_filter = status_filter
        self.annotation_types = annotation_types

        self.labels_path_split = self.labels_path / stage
        self.output_path = self.labels_path_split / f"crops_{crop_size}"
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.empty_fraction = False

    def unzip(self, hasty_annotations_labels_zipped: Path, hasty_annotations_images_zipped: Path):
        if (not self.labels_path.joinpath(HastyConverter.IMAGES_PATH).exists() or
                not self.labels_path.joinpath(HastyConverter.ANNOTATION_PATH).exists()):

            hA = HastyConverter.from_zip(base_path=self.labels_path,
                                         hasty_annotations_labels_zipped=self.hasty_annotations_labels_zipped,
                                         hasty_annotations_images_zipped=self.hasty_annotations_images_zipped)

        else:
            labels = list(self.labels_path.joinpath(HastyConverter.ANNOTATION_PATH).glob("*.json"))
            assert len(labels) == 1, "There should not be another label"
            hA = hA_from_file(labels[0])

        return hA

    def run(self):
        self.unzip()

        hA = hasty_convert_pipeline(self.labels_path,
                                    self.output_path_train,
                                    class_filter=self.class_filter,
                                    status_filter=self.status_filter,
                                    dataset_filter=self.dataset_filter,
                                    images_filter=self.images_filter,
                                    # images_filter=["DJI_0258_FCD02.JPG"],
                                    # image_tags=["segment"],
                                    # annotation_types=["point"]
                                    annotation_types=self.annotation_types
                                    )
        if self.images_filter_func is not None:
            hA = self.images_filter_func(hA)

        with open(self.output_path_train / f"hasty_format_{'_'.join(self.class_filter)}.json", mode="w") as f:
            f.write(hA.model_dump_json())

        hA = self.data_prep_pipeline(crop_size=self.crop_size, overlap=self.overlap, stage=self.stage, hA=hA)

        self.hA = hA

    def data_prep_pipeline(self,
                           stage,
                           overlap,
                           crop_size,
                           hA: HastyAnnotationV2):
        """

        :param stage:
        :param overlap:
        :param crop_size:
        :param hA:
        :return:
        """
        full_images_path = self.labels_path_split

        # TODO use a temporary folder for this
        full_images_path_padded = self.labels_path_split / "padded"
        full_images_path_padded.mkdir(exist_ok=True, parents=True)

        all_images: typing.List[AnnotatedImage] = []
        all_images_path: typing.List[Path] = []

        for i in hA.images:
            original_image_path = full_images_path / i.dataset_name / i.image_name
            padded_image_path = full_images_path_padded / i.dataset_name / i.image_name
            padded_image_path.parent.mkdir(exist_ok=True, parents=True)
            new_width, new_height = pad_to_multiple(original_image_path,
                                                    padded_image_path,
                                                    crop_size,
                                                    crop_size,
                                                    overlap)

            grid, _ = create_regular_raster_grid(max_x=new_width,
                                                 max_y=new_height,
                                                 slice_height=crop_size,
                                                 slice_width=crop_size,
                                                 overlap=overlap)

            # axi = visualise_image(image_path=padded_image_path, show=False, title=f"grid_{i.image_name}", )
            # visualise_polygons(grid, show=True, title=f"grid_{i.image_name}", max_x=new_width, max_y=new_height, ax=axi)

            images, images_path = crop_out_images_v2(i, rasters=grid,
                                                     full_images_path=full_images_path_padded,
                                                     output_path=self.output_path,
                                                     include_empty=self.empty_fraction)
            image = Image.open(full_images_path / i.dataset_name / i.image_name)

            # TODO is it a good idea to seperate the two functions?
            # images_path = crop_out_images_v3(image=image, rasters=grid )

            all_images.extend(images)
            all_images_path.extend(images_path)

        hA.images = all_images
        hA_regression = HastyAnnotationV2(
            project_name="iguanas_regression",
            images=all_images,
            export_format_version="1.1",
            label_classes=hA.label_classes
        )

        with open(self.output_path / "hasty_format.json", mode="w") as f:
            f.write(hA_regression.model_dump_json())

        self.augmented_images = all_images
        self.augmented_images_path = all_images_path

        return hA_regression

    def get_hA(self) -> HastyAnnotationV2:
        return self.hA

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
