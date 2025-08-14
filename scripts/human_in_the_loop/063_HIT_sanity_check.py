import json


from pathlib import Path
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

hA = HastyAnnotationV2.from_file(Path("/Users/christian/data/training_data/2025_08_10_label_correction/fernandina_s_correction_hasty_corrected_1.json"))

# chck how many images got cvat attribute labels

cvat_labels = 0
for image in hA.images:
    for label in image.labels:
        if label.attributes is not None:
            for attribute in label.attributes:
                if attribute == "cvat":
                    cvat_labels += 1
                    print(cvat_labels)