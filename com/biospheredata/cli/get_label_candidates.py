"""
Use a pretrained model to predict annotations of a drone mission.
These will be converted later to hasty.ai format


"""


import json
import shutil
from pathlib import Path

from com.biospheredata.helper.candidate_proposal import candidate_proposal_prediction_from_mission
from com.biospheredata.types.HastyAnnotation import HastyAnnotation
from com.biospheredata.types.Mission import Mission

from loguru import logger


yolov5_model_path = "/home/christian/work/ag-dataplatform/object-detection-pytorch/tests_data/models/turtle_2022_10_15_0f89da13bcbfae02a5d0f4b61465e081.pt"

# base_path = Path("/media/christian/2TB/ai-core/data/Madagascar_expedition/2022-08-04/") ## DONE
# mission_path = base_path.joinpath("101MEDIA") ## DONE

base_path = Path("/home/christian/data/madagascar_expedition/2022-08-01/")
mission_path = base_path.joinpath("1")

CRS = "EPSG:4326"
# Open a Mission from a Folder
# mission = Mission.init(base_path=Path(mission_path), CRS=CRS, suffix="JPG")
#
# hA, image_names, sliced_image_path = candidate_proposal_prediction_from_mission(
#     mission=mission,
#     model_path=yolov5_model_path,
#     image_size=832,
#     logger=logger
# )
# with open("./hasty_annotation_tmp.json", 'w') as f:
#     json.dump(hA.to_dict(), f)

mission = Mission.open(base_path=Path(mission_path))
hA = HastyAnnotation.from_file("./hasty_annotation_tmp.json")

images_with_more_than_zero_turtles = hA.get_images_with_more_than(searched_label="turtle", threshold=1)

image_with_objects = Path("image_with_objects")
image_with_objects.mkdir(exist_ok=True)
for ip in images_with_more_than_zero_turtles:
    shutil.copy(mission_path.joinpath(ip), image_with_objects.joinpath(ip))

logger.info(f"your data is ready to be uploaded to hasty.ai and you can check and correct the annotations.")