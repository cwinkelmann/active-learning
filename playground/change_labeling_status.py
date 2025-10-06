from pathlib import Path

# change the labelling Status of all images in a dataset
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.types.status import LabelingStatus

labels_path = Path("/home/cwinkelmann/work/Herdnet/data/2025_09_28_orthomosaic_data")
# labels_path = Path("/Users/christian/data/training_data/2025_08_10_endgame")

# hacky way
# get a list of images which are in a dataset
labels_file_path = labels_path / "unzipped_hasty_annotation/2025_09_19_orthomosaic_data_combined_corrections_3.json"

hA = HastyAnnotationV2.from_file(labels_file_path)


maps = {
'ha_corrected_fer_fnj01_19122021': "Fernandina",
    'ha_fer_fnj01_19122021': "Fernandina",
    'ha_corrected_fer_fpe09_18122021': "Fernandina",
    'ha_fer_fpe09_18122021': "Fernandina",
    'ha_corrected_fer_fnd02_19122021': "Fernandina",
    'ha_fer_fnd02_19122021': "Fernandina",
    'ha_fer_fni03_04_19122021': "Fernandina",
    'ha_corrected_fer_fni03_04_19122021': "Fernandina",
    'ha_corrected_fer_fna01_02_20122021': "Fernandina",
    'ha_fer_fna01_02_20122021': "Fernandina",
    'ha_corrected_fer_fef01_02_20012023': "Fernandina",
    'ha_fer_fef01_02_20012023': "Fernandina",


    'ha_corrected_flo_flpo02_04022021': "Floreana",
    'ha_flo_flpo02_04022021': "Floreana",
    'ha_corrected_flo_flpc04_22012021': "Floreana",
    'ha_flo_flpc04_22012021': "Floreana",
    'ha_corrected_flo_flpo01_04022021': "Floreana",
    'ha_flo_flpo01_04022021': "Floreana",
    'ha_corrected_flo_flpc03_22012021': "Floreana",
    'ha_flo_flpc03_22012021': "Floreana",
    'ha_corrected_flo_flbb01_28012023': "Floreana",
    'ha_flo_flbb01_28012023': "Floreana",
    'ha_corrected_flo_flbb02_28012023': "Floreana",
    'ha_flo_flbb02_28012023': "Floreana",
    }



for img in hA.images:
    img.image_status

    if maps.get(img.dataset_name, None):
        print(f"Changing {img.image_name} in dataset {img.dataset_name} from {img.image_status} to {LabelingStatus.COMPLETED}")
        img.image_status = LabelingStatus.COMPLETED

hA.save(file_path = labels_path / "unzipped_hasty_annotation/2025_09_19_orthomosaic_data_combined_corrections_3.json")