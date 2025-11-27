import pandas as pd
from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/IFA_training_data/2025_10_11_labels.json")

hA = HastyAnnotationV2.from_file(path)

hA_stats = hA.dataset_statistics()

# change the status of all images with status "Done" to "COMPLETED"

for i in hA.images:
    if i.image_status == "Done" and i.dataset_name in ["ha_corrected_fer_fwk01_20122021", "ha_corrected_fer_fe01_02_20012023"] 	:
        i.image_status = "COMPLETED"

hA.save(path)

pd.DataFrame(hA_stats)