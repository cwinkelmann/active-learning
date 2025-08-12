"docker build -f ./container/training/Dockerfile --tag dockerkartok/yolo-train:$CI_COMMIT_SHORT_SHA ."
import json

import os

from pathlib import Path
from loguru import logger
from com.biospheredata.converter.HastyToYoloConverter import whole_workflow

base_path = Path(os.getenv("BASE_PATH", "/Users/christian/data/training_data/iguanas_2024_03_03"))

training_data_location = Path(os.getenv("TRAINING_DATA_PATH", "/Users/christian/data/training_data/iguanas_2024_03_03"))
AMOUNT_TRAINING_IMAGES = json.loads(os.getenv("AMOUNT_TRAINING_IMAGES", "[10]"))
BACKGROUNDS = json.loads(os.getenv("BACKGROUNDS", "[0]"))
batch_size = int(os.getenv("BATCH_SIZE", 10))
PREFIX = os.getenv("PREFIX", "local_testing")
## should work via hnee gpu
storage = os.getenv("STORAGE_LOCATION", "/data/mnt/storage/ai-core/data/object_detection/training_logs")
SHELLSCRIPT_LOCATION = os.getenv("SHELLSCRIPT_LOCATION", "/app/shellscript.sh")
initial_weights = os.getenv("MODEL", "yolov5n6.pt")
KEEP_CLASSES = os.getenv("KEEP_CLASSES", "iguana").split(",")
HYP_YAML = os.getenv("HYP_YAML", None)
with_augmentations = int(os.getenv("WITH_AUGMENTATION", 0))

hasty_annotations_labels_zipped = "labels.zip"
hasty_annotations_images_zipped = "images.zip"
# initial_weights = "best_model_9b949e49ae778b1263dd898ce44da8a4.pt"

epochs = int(os.getenv("EPOCHS", 5))
imgsz = 2500
# SHELLSCRIPT_LOCATION = "./shellscript.sh"
patience = int(os.getenv("PATIENCE", 5))
extension = "jpg"

def generate_training_runs():

    generated_datasets = whole_workflow(base_path,
                                        hasty_annotations_labels_zipped,
                                        hasty_annotations_images_zipped,
                                        amount_training_images=AMOUNT_TRAINING_IMAGES,
                                        backgrounds=BACKGROUNDS,
                                        # backgrounds=[0],
                                        fulls=True,
                                        slice_size=imgsz,
                                        training_data_location=training_data_location,
                                        prefix=PREFIX,
                                        keep_classes=KEEP_CLASSES,
                                        folds=5,
                                        extension=extension,
                                        with_augmentations=with_augmentations
                                        )

    configs = []




    # with open("../../../playground/shellscript.sh", 'w') as w:
    ### TODO this is fine for now in the docker container
    with open(SHELLSCRIPT_LOCATION, 'w') as w:
        w.writelines("#! /bin/bash\n")
        n = 0
        for gd in generated_datasets:
            # print(gd)

            name = gd["ds"]
            data = f"{str(base_path)}/{name}/fold_{gd['fold']}/data.yaml"

            validation_weights = f"{storage}/runs/train/{name}/weights/best.pt"

            # delete_statement = f"docker rm {name} || true"
            # run_statement = f"docker run --gpus all --name {name} --ipc=host -v {storage}:{storage} dockerkartok/yolo-train:$CI_COMMIT_SHORT_SHA /bin/bash -c"
            docker_run_train = f"python3 yolov5/train.py --img {imgsz} --batch {batch_size} --epochs {epochs} --data {data} --weights {initial_weights} --name {name} --project={storage}/runs/train --patience={patience}"
            # docker_run_train = f"python3 yolov5/train.py --img {imgsz} --freeze 10 --batch {batch_size} --epochs {epochs} --data {data} --weights {initial_weights} --name {name} --project={storage}/runs/train --patience={patience}"
            if HYP_YAML:
                docker_run_train = docker_run_train + f" --hyp {HYP_YAML}"

            docker_run_val = f"python3 yolov5/val.py --data {data} --batch {batch_size} --img {imgsz} --weights {validation_weights} --name {name} --task test --conf-thres 0.5 --augment --save-txt --project={storage}/runs/val"

            # the_command = delete_statement + " && " + run_statement + " '" + docker_run_train + " && " + docker_run_val + "'\n"
            the_command = docker_run_train + " && " + docker_run_val + "\n"
            print(the_command)
            w.writelines(the_command)
            logger.info(the_command)


        sleep_command = "sleep 60 to make sure everything is cleaned up and synced."
        w.writelines(sleep_command)

    ## Convert to Hasty.ai format to check if the slicing worked as expected.
    from pathlib import Path

    from com.biospheredata.converter.YoloToHastyConverter import YoloToHastyConverter
    from com.biospheredata.types.HastyAnnotation import HastyAnnotation


    annotations_labels_path = base_path.joinpath("tiled/images_with_objects")
    annotations_images_path = base_path.joinpath("tiled/images_with_objects")
    class_name_mapping_path = base_path.joinpath("unzipped_yolo_format/class_names.txt")
    yHC = YoloToHastyConverter(base_path=base_path,
                               annotations_labels_path=annotations_labels_path,
                               annotations_images_path=annotations_images_path,
                               class_name_mapping_path=class_name_mapping_path,
                               ext=extension
                               )

    list_labels = yHC.transform_coordinates()

    hA = HastyAnnotation(
        project_name="test_case", create_date="2023-01-030T20:30:50Z", export_date="2023-01-030T20:30:50Z",
    )
    hA = yHC.transform(hA)
    hA.set_label_classes([
            {
              "class_id": "f77a0654-a822-4f21-9240-27ac6143a4e1",
              "parent_class_id": None,
              "class_name": "iguana",
              "class_type": "object",
              "color": "#a603034d",
              "norder": 10.0,
              "icon_url": None,
              "attributes": []
            },
            # {
            #   "class_id": "6a6e780e-8422-4065-a7ac-9c22407524c8",
            #   "parent_class_id": None,
            #   "class_name": "crab",
            #   "class_type": "object",
            #   "color": "#f5d03a4d",
            #   "norder": 11.0,
            #   "icon_url": None,
            #   "attributes": []
            # },
            # {
            #   "class_id": "8ceef749-71b4-43c6-a5a3-ac98be6a2ac2",
            #   "parent_class_id": None,
            #   "class_name": "turtle",
            #   "class_type": "object",
            #   "color": "#4175054d",
            #   "norder": 12.0,
            #   "icon_url": None,
            #   "attributes": []
            # },
            # {
            #   "class_id": "137a0e5e-342c-4775-91e0-0e95945b1943",
            #   "parent_class_id": None,
            #   "class_name": "seal",
            #   "class_type": "object",
            #   "color": "#734c144d",
            #   "norder": 13.0,
            #   "icon_url": None,
            #   "attributes": []
            # },
            # {
            #   "class_id": "8180dc71-6fb1-435c-a779-195353c694f1",
            #   "parent_class_id": None,
            #   "class_name": "ugly_stone",
            #   "class_type": "object",
            #   "color": "#cea9174d",
            #   "norder": 14.0,
            #   "icon_url": None,
            #   "attributes": []
            # },
            # {
            #   "class_id": "e53dea84-7a6a-43b7-a299-15ad02c847ee",
            #   "parent_class_id": None,
            #   "class_name": "trash",
            #   "class_type": "object",
            #   "color": "#2626264d",
            #   "norder": 15.0,
            #   "icon_url": None,
            #   "attributes": []
            # },
            # {
            #   "class_id": "c5d90dd6-81cc-4ec9-bedf-18c3690dc7dd",
            #   "parent_class_id": None,
            #   "class_name": "not_iguana_but_similar_look",
            #   "class_type": "object",
            #   "color": "#1f78b44d",
            #   "norder": 16.0,
            #   "icon_url": None,
            #   "attributes": []
            # }
          ],)
    hasty_annotation_file = hA.persist(base_path)

    logger.info(f"finished training preparation. build training commands here: {SHELLSCRIPT_LOCATION}")

    return hasty_annotation_file


if __name__ == "__main__":
    generate_training_runs()