import pandas as pd
from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

path = Path("/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_09_28_orthomosaic_data/2025_09_19_orthomosaic_data_combined_corrections_4.json")

hA = HastyAnnotationV2.from_file(path)

hA_stats = hA.delete_dataset()

hA_stats

pd.DataFrame(hA_stats).to_csv(path.parent / f"{path.stem}_dataset_stats.csv", index=False)