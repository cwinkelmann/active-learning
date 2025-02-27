from pathlib import Path

from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation, evaluate_predictions
from com.biospheredata.converter.HastyConverter import AnnotationType
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
import fiftyone as fo
from loguru import logger

def hastyAnnotation2fiftyOne(hA: HastyAnnotationV2, images_path: Path, annoation_type: AnnotationType) -> fo.Dataset:
    """
    Visualise the hasty annotation in fiftyone
    :param hA:
    :param images_path:
    :return:
    """
    images_set = [images_path / i.image_name for i in hA.images]
    for img_path in images_set:
        if not img_path.exists():
            logger.warning("Image path does not exist: %s", img_path)
    dataset_name = "temp"
    try:
        fo.delete_dataset(dataset_name)
    except:
        logger.warning(f"Dataset {dataset_name} does not exist")
    finally:
        # Create an empty dataset, TODO put this away so the dataset is just passed into this
        dataset:fo.Dataset = fo.Dataset(name=dataset_name)
        dataset.persistent = False

    samples = [fo.Sample(filepath=path) for path in images_set]
    dataset.add_samples(samples)

    for sample in dataset:
        # create dot annotations
        sample = evaluate_predictions(
            predictions=hA.images,
            sample=sample,
            sample_field_name="detection",
            # images_set=images_set,
            type=annoation_type,
        )
        sample.save()

    return dataset