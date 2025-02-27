from active_learning.config.dataset_filter import DatasetFilterConfig
from com.biospheredata.converter.HastyConverter import AnnotationType, ClassName

test_fixes = DatasetFilterConfig(**{
        "dset": "test_fixed",
        # "images_filter": ["DJI_0935.JPG", "DJI_0972.JPG", "DJI_0863.JPG"],
        # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
        "dataset_filter": ["FLMO02_28012023", "FMO05"],
        # "dataset_filter": None,
        # "num": 1,
        "class_filter": ["iguana_point"],
        "annotation_types": [AnnotationType.KEYPOINT],
        "empty_fraction": 0.0,

    })

train_segments_fernanandina_1 = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "train_segments_1",
    "images_filter": ["DJI_0366.JPG"],
    "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images"],
    "class_filter": ["iguana"],
    "annotation_types": [AnnotationType.POLYGON],
    "tag_filter": ["segment"],
    "empty_fraction": 0.0,
})
train_segments_fernanandina_12 = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "segments_12",
    "images_filter": ["DJI_0366.JPG", "STJB06_12012023_Santiago_m_2_7_DJI_0128.JPG", "DJI_0009.JPG", "DJI_0893.JPG", "DJI_0924.JPG",
                      "DJI_0942.JPG", "DJI_0417.JPG", "DJI_0097.JPG", "DJI_0185.JPG", "DJI_0195.JPG", "DJI_0285.JPG"],
    "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images", "San_STJB06_12012023", "Floreana_02.02.21_FMO01", "Floreana_03.02.21_FMO06", "FLMO02_28012023",
                       ],
    "class_filter": ["iguana"],
    "annotation_types": [AnnotationType.POLYGON],
    "tag_filter": ["segment"],
    "empty_fraction": 0.0,
})

train_segments_points_fernanandina = DatasetFilterConfig(**{
    "dset": "train",
    "dataset_name": "segments_points_many",
    "images_filter": [],
    "images_exclude": ["DJI_0079_FCD01.JPG", "DJI_0395.JPG"],
    "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images", "San_STJB06_12012023",
                       "Floreana_02.02.21_FMO01", "Floreana_03.02.21_FMO06", "FLMO02_28012023",
                       ],
    "class_filter": [ClassName.iguana, ClassName.iguana_point],
    "annotation_types": [AnnotationType.POLYGON, AnnotationType.KEYPOINT],
    "image_tags": ["points", "segment"],
    "empty_fraction": 0.0,
})

val_segments_fernandina_1 = DatasetFilterConfig(**{
    "dset": "val",
    "dataset_name": "segments_12",
    "images_filter": ["DJI_0079_FCD01.JPG"],
    "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images"],
    "class_filter": ["iguana"],
    "annotation_types": [AnnotationType.POLYGON],
    "tag_filter": ["segment"],
    "empty_fraction": 0.0,
})
test_segments_fernandina_1 = DatasetFilterConfig(**{
    "dset": "test",
    "dataset_name": "segments_12",
    "images_filter": ["DJI_0395.JPG"],
    "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images"],
    "class_filter": ["iguana"],
    "annotation_types": [AnnotationType.POLYGON],
    "tag_filter": ["segment"],
    "empty_fraction": 0.0,
})

train_fernandina = DatasetFilterConfig(**{
    "dset": "train_fernandina",
    "dataset_filter": ["Fer_FCD01-02-03_20122021_single_images"], # Fer_FCD01-02-03_20122021_single_images
    # "num": 1,
    "empty_fraction": 0.0
})

train_floreana_small = DatasetFilterConfig(**{
    "dset": "train_floreana_small",
    "dataset_filter": ["FMO03", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # small Floreans subset
    # "num": 1,
    "empty_fraction": 0.0
})

train_floreana_big = DatasetFilterConfig(**{
    "dset": "train_floreana_big",
    "dataset_filter": ["FMO03", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # big Floreana subset
    # "num": 1,
    "empty_fraction": 0.0
})

val_fmo03 = DatasetFilterConfig(**{
        "dset": "val",
        # "images_filter": ["DJI_0465.JPG"],
        "dataset_filter": ["FMO03"],
        # "dataset_filter": None,
        "empty_fraction": 0.0

    })

test_fernandina_m = DatasetFilterConfig(**{
        "dset": "test",
        # "images_filter": ["DJI_0465.JPG"],
        "dataset_filter": ["Fer_FCD01-02-03_20122021"],
        # "dataset_filter": None,
        "empty_fraction": 0.0

    })

test_fmo02 = DatasetFilterConfig(**{
        "dset": "test",
        # "images_filter": ["DJI_0554.JPG"],
        "dataset_filter": ["FMO02"],
        "empty_fraction": 0.0

    })


datasets = [
    {
        "dset": "train",
        # "images_filter": ["DJI_0935.JPG", "DJI_0972.JPG", "DJI_0863.JPG"],
        # "dataset_filter": ["FMO05", "FSCA02", "FMO04", "Floreana_03.02.21_FMO06", "Floreana_02.02.21_FMO01"], # Fer_FCD01-02-03_20122021_single_images
        "dataset_filter": ["FMO05"],
        # "dataset_filter": None,
        "num": 1,
        "empty_fraction": 0.0
    },
    {
        "dset": "val",
        # "images_filter": ["DJI_0465.JPG"],
        "dataset_filter": ["FMO03"],
        # "dataset_filter": None,
        "empty_fraction": 0.0

    },
    {
        "dset": "test",
        # "images_filter": ["DJI_0554.JPG"],
        "dataset_filter": ["FMO02"],
        "empty_fraction": 0.0

    }
]

## Data preparation based on segmentation masks
train_segments = DatasetFilterConfig(**{
    "dset": "train",
    "images_filter": ["STJB06_12012023_Santiago_m_2_7_DJI_0128.JPG", "DJI_0079_FCD01.JPG", "DJI_0924.JPG", "DJI_0942.JPG",
                      "DJI_0097.JPG", "SRPB06 1053 - 1112 falcon_dem_translate_0_0.jpg", "DJI_0097.JPG", "DJI_0185.JPG",
                      "DJI_0195.JPG", "DJI_0237.JPG", "DJI_0285.JPG", "DJI_0220.JPG",
                      ],
    "empty_fraction": 0.0,
    "image_tags": ["segment"]
})

val_segments = DatasetFilterConfig(**{
    "dset": "val",
    "images_filter": ["DJI_0395.JPG", "DJI_0009.JPG", "DJI_0893.JPG", "DJI_0417.JPG"],
    "empty_fraction": 0.0,
    "image_tags": ["segment"]
})