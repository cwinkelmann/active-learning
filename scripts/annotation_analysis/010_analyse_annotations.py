"""
A marked iguana is not consistent in size an sharpness etc


"""
import PIL
import pandas as pd
from matplotlib import pyplot as plt

from active_learning.util.image_manipulation import crop_out_individual_object
from active_learning.util.visualisation.annotation_vis import plot_frequency_distribution, plot_visibility_scatter, \
    plot_image_grid_by_visibility
from com.biospheredata.types.HastyAnnotationV2 import convert_HastyAnnotationV2_to_HastyAnnotationV2flat, Attribute


"""
Create patches from images and labels from hasty to be used in CVAT
"""
import shutil
from loguru import logger
from pathlib import Path

from active_learning.config.dataset_filter import DatasetFilterConfig
from active_learning.filter import ImageFilterConstantNum, ImageLabelFilterAttribute
from active_learning.pipelines.data_prep import DataprepPipeline, UnpackAnnotations, AnnotationsIntermediary
from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.converter.HastyConverter import HastyConverter

## TODO Download annotations from hasty



if __name__ == "__main__":
    """ This only works if the input is a hasty zip file which is very constraining. """

    # labels_path = Path("/Users/christian/data/training_data/2025_01_11")
    # hasty_annotations_labels_zipped = "labels_segments.zip"
    # hasty_annotations_images_zipped = "images_segments.zip"
    # annotation_types = [AnnotationType.POLYGON]
    #
    # labels_path = Path("/Users/christian/data/training_data/2024_12_09")
    # hasty_annotations_labels_zipped = "FMO02_03_05_labels.zip"
    # hasty_annotations_images_zipped = "FMO02_03_05_images.zip"
    # annotation_types = [AnnotationType.BOUNDING_BOX]

    # class_filter = ["iguana"]




    ## Meeting presentation
    # labels_path = Path("/Users/christian/data/training_data/2024_12_09_debug")
    # hasty_annotations_labels_zipped = "FMO02_03_05_labels.zip"
    # hasty_annotations_images_zipped = "FMO02_03_05_images.zip"
    # annotation_types = [AnnotationType.KEYPOINT]
    # class_filter = ["iguana_point", "iguana"]

    ## Segmentation masks
    labels_path = Path("/Users/christian/data/training_data/2025_04_18_all")
    hasty_annotations_labels_zipped = "labels.zip"
    hasty_annotations_images_zipped = "images.zip"
    annotation_types = [AnnotationType.BOUNDING_BOX]

    class_filter = ["iguana"]

    crop_size = 640
    overlap = 0
    # amount of empty images in the dataset


    ## Data preparation based on segmentation masks
    train_segments = DatasetFilterConfig(**{
        "dset": "train",
        # "images_filter": ["DJI_0483.JPG" ],
        # "images_filter": ["STJB06_12012023_Santiago_m_2_7_DJI_0128.JPG", "DJI_0079_FCD01.JPG", "DJI_0924.JPG", "DJI_0942.JPG",
        #                   "DJI_0097.JPG", "SRPB06 1053 - 1112 falcon_dem_translate_0_0.jpg", "DJI_0097.JPG", "DJI_0185.JPG",
        #                   "DJI_0195.JPG", "DJI_0237.JPG", "DJI_0285.JPG", "DJI_0220.JPG",
        #                   ],
        # "attribute_filter": [
        #     Attribute(name="visibility", type="int", values=[0,1,2,3,4,5,6,7,8,9,10])
        # ],
        # "dataset_filter": ["Fer_FPM01-02_20122023"],
        "dataset_filter": ["FMO03", "FMO05", "FMO02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
        "empty_fraction": 0.0,
        # "image_tags": ["segment"],
         # "num": 375

    })

    val_segments = DatasetFilterConfig(**{
        "dset": "val",
        "images_filter": ["DJI_0395.JPG", "DJI_0009.JPG", "DJI_0893.JPG", "DJI_0417.JPG"],
        "attribute_filter": [
            Attribute(name="visibility", type="int", values=[0, 1, 10])
        ],
        "empty_fraction": 0.0,
        "image_tags": ["segment"]
    })


    # datasets = [train_fmo05, val_fmo03, test_fmo02]
    # datasets = [train_floreana, val_fmo03, fernandina_ds]
    datasets = [train_segments]
    report = {}


    for dataset in datasets:  # , "val", "test"]:
        logger.info(f"Starting {dataset.dset}")
        dset = dataset.dset
        num = dataset.num
        ifcn = ImageFilterConstantNum(num=num, dataset_config = dataset, min_labels=0)

        if_att = ImageLabelFilterAttribute(dataset_config = dataset)

        uA = UnpackAnnotations()
        hA, images_path = uA.unzip_hasty(hasty_annotations_labels_zipped=labels_path / hasty_annotations_labels_zipped,
                       hasty_annotations_images_zipped = labels_path / hasty_annotations_images_zipped)
        logger.info(f"Unzipped {len(hA.images)} images.")
        output_path_dset = labels_path / dset
        output_path_dset_object = output_path_dset / "individual_object"
        output_path_dset_object_const = output_path_dset / "individual_object_constant"
        output_path_dset_object.mkdir(exist_ok=True, parents=True)
        output_path_dset_object_const.mkdir(exist_ok=True, parents=True)

        dp = DataprepPipeline(annotations_labels=hA,
                              images_path=images_path,
                              crop_size=crop_size,
                              overlap=overlap,
                              output_path=output_path_dset,
                              )
        dp.dataset_filter = dataset.dataset_filter

        dp.images_filter = dataset.images_filter

        # dp.add_images_filter_func(if_att)
        dp.add_images_filter_func(ifcn)


        dp.class_filter = class_filter
        # dp.status_filter = "COMPLETED"
        dp.annotation_types = annotation_types
        dp.empty_fraction = dataset.empty_fraction

        # TODO inject a function for cropping so not only the regular grid is possible but random rotated crops too
        dp.run(flatten=True)

        hA_filtered = dp.get_hA_filtered()

        individual_object_cropped_labels = []

        l = [{"label_id": l.id,
              "image_name": i.image_name,
              "width": l.width,
              "area": l.width * l.height,
              "visibility": l.attributes.get("visibility", -1),
              "height": l.height, "bbox": l.bbox} for i in hA_filtered.images for l in i.labels] # TODO add the bounding box in here too
        df_parameter = pd.DataFrame(l)
        df_parameter.to_csv(output_path_dset / "individual_object.csv")
        # visualise a histogram

        fig = plot_frequency_distribution(df_parameter, columns=['visibility', 'width', 'height'])
        fix2 = plot_visibility_scatter(df_parameter)
        plt.show()

        hA_crops = dp.get_hA_crops()

        aI = AnnotationsIntermediary()
        aI.set_hasty_annotations(hA=hA_crops)
        coco_path = aI.coco(output_path_dset / "coco_format.json")

        for i in hA_filtered.images:
            im = PIL.Image.open(output_path_dset / i.dataset_name / i.image_name)
            # convert to RGB
            if im.mode != "RGB":
                im = im.convert("RGB")
            # TODO fix that this swallows a bounding box and or polygon
            image_mappings, cropped_annotated_images, images_set = crop_out_individual_object(i,
                                                                                          offset=0,
                                                                                          im=im,
                                                                                          output_path=output_path_dset_object)

            # individual_object_cropped_labels.extend(cropped_annotated_images)

            image_mappings_constant, cropped_annotated_images_constant, images_set_constant = crop_out_individual_object(i,
                                                                                          width=crop_size,
                                                                                          height=crop_size,
                                                                                          im=im,
                                                                                          output_path=output_path_dset_object_const)
            individual_object_cropped_labels.extend(cropped_annotated_images_constant)



        l2 = [{"label_id": l.id,
              "image_name": i.image_name,
              "width": l.width,
              "area": l.width * l.height,
              "height": l.height, "bbox": l.bbox} for i in hA_crops.images for l in i.labels]



        df_cropped_label = pd.DataFrame([{"label_id": l.id,
              "crop_image_name": i.image_name,
              "dataset_name": i.dataset_name, "bbox": l.bbox} for i in individual_object_cropped_labels for l in i.labels])

        df_merged = df_cropped_label.merge(df_parameter, on=["label_id"], how="left")

        fig = plot_image_grid_by_visibility(df_merged,
                                            image_dir=output_path_dset_object_const,
                                            max_images_per_visibility=5,
                                            width_scale=1.2)
        plt.show()

        """
        [AnnotatedImage(image_name='Fer_FCD01-02-03_20122021_single_images___DJI_0079_FCD01.JPG', image_id='13978b69-5571-4da1-9a6e-769...3_20122021_single_images___DJI_0079_FCD01.JPG', image_status='COMPLETED', tags=['easy', 'sand', 'segment'], image_mode='YCbCr'), AnnotatedImage(image_name='FLBB01_28012023___DJI_0220.JPG', image_id='14e7c0c9-37c2-4184-a432-1b1c51908495', labels=[ImageLabe...image_name='FLBB01_28012023___DJI_0220.JPG', image_status='COMPLETED', tags=['hard', 'segment', 'unsharp'], image_mode='YCbCr'), AnnotatedImage(image_name='SRPB06 1053 - 1112 falcon_25.01.20___SRPB06 1053 - 1112 falcon_dem_translate_0_0.jpg', image_id='2a...con_25.01.20___SRPB06 1053 - 1112 falcon_dem_translate_0_0.jpg', image_status='COMPLETED', tags=['segment'], image_mode='CMYK'), AnnotatedImage(image_name='FLMO02_28012023___DJI_0195.JPG', image_id='378779b4-3fda-4cde-bfdb-c837ccc56078', labels=[ImageLabe...name='Default', ds_image_name='FLMO02_28012023___DJI_0195.JPG', image_status='COMPLETED', tags=['segment'], image_mode='YCbCr'), AnnotatedImage(image_name='San_STJB06_12012023___STJB06_12012023_Santiago_m_2_7_DJI_0128.JPG', image_id='4c94b2df-4050-44aa-89...2012023___STJB06_12012023_Santiago_m_2_7_DJI_0128.JPG', image_status='COMPLETED', tags=['hard', 'segment'], image_mode='YCbCr'), AnnotatedImage(image_name='FLMO02_28012023___DJI_0185.JPG', image_id='5b19e3f5-126f-484f-a466-2c09ce02c9ab', labels=[ImageLabe...name='Default', ds_image_name='FLMO02_28012023___DJI_0185.JPG', image_status='COMPLETED', tags=['segment'], image_mode='YCbCr'), AnnotatedImage(image_name='Floreana_02.02.21_FMO01___DJI_0942.JPG', image_id='6d45aa09-247a-4a6f-8019-3e8741ed2ed9', labels=[I...fault', ds_image_name='Floreana_02.02.21_FMO01___DJI_0942.JPG', image_status='COMPLETED', tags=['segment'], image_mode='YCbCr'), AnnotatedImage(image_name='FLMO02_28012023___DJI_0237.JPG', image_id='6dfb1201-d74b-4d03-8c92-6412b7d54f13', labels=[ImageLabe...name='Default', ds_image_name='FLMO02_28012023___DJI_0237.JPG', image_status='COMPLETED', tags=['segment'], image_mode='YCbCr'), AnnotatedImage(image_name='FLMO02_28012023___DJI_0285.JPG', image_id='b45e5b6c-fb92-44ae-a476-cb745b61f106', labels=[ImageLabe...name='Default', ds_image_name='FLMO02_28012023___DJI_0285.JPG', image_status='COMPLETED', tags=['segment'], image_mode='YCbCr'), AnnotatedImage(image_name='Floreana_02.02.21_FMO01___DJI_0924.JPG', image_id='e6a96136-ecdf-4f58-b248-26605cff2d21', labels=[I...fault', ds_image_name='Floreana_02.02.21_FMO01___DJI_0924.JPG', image_status='COMPLETED', tags=['segment'], image_mode='YCbCr'), AnnotatedImage(image_name='FLMO02_28012023___DJI_0097.JPG', image_id='f40cbdef-7292-430d-88b9-2dda80d84e4d', labels=[ImageLabe...ds_image_name='FLMO02_28012023___DJI_0097.JPG', image_status='COMPLETED', tags=['good_quality', 'segment'], image_mode='YCbCr')]
        """

        # convert_HastyAnnotationV2_to_HastyAnnotationV2flat(hA_filtered)
        # TODO build a histogram
