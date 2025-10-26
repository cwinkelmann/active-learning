from pathlib import Path

from scripts.inferencing.pipeline import geospatial_inference_pipeline, get_config

orthomosaic_list_3 = [
    "/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Gen_GES10to15_05122021.tif",
    # "/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Flo_FLBB02_28012023.tif",
    # "/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Esp_ESCK05_10022023.tif",
    # "/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Esp_ESCK04_10022023.tif",
    # "/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Esp_ESCK02-03_10022023.tif",
    # "/raid/cwinkelmann/Manual_Counting/Drone Deploy orthomosaics/Esp_ESCH01-02_10022023.tif",
]

orthomosaic_list_3 = [Path(o) for o in orthomosaic_list_3]
hydra_cfg = get_config(config_name="genovesa_dla34", config_path="../configs/submission")
for o in orthomosaic_list_3:
    geospatial_inference_pipeline(o, hydra_cfg=hydra_cfg)