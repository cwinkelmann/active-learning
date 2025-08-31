from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


def get_dataset_stats(dataset_path: Path):
    """
    Get statistics of the dataset including number of images, dimensions, and unique image IDs.
    :param dataset_path: Path to the dataset directory.
    :return: Dictionary with dataset statistics.
    """
    hA_loaded = HastyAnnotationV2.from_file(dataset_path)

    dict_some_stats = {}
    dict_some_stats["images_total"] = len(hA_loaded.images)
    dict_some_stats["labels_total"] = sum(len(img.labels) for img in hA_loaded.images if img.labels)
    return  dict_some_stats
