from pathlib import Path

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.types.ImageStatistics import ImageCounts

corrected_path = "/Users/christian/data/training_data/2025_07_10_refined/label_correction_floreana_2025_07_10_review_hasty_corrected_formatted.json"
corrected_filtered_path = Path("/Users/christian/data/training_data/2025_07_10_refined/label_correction_floreana_2025_07_10_review_hasty_corrected_formatted_filtered.json")

hA = HastyAnnotationV2.from_file(corrected_path)
hA.images = hA.get_image_by_name("DJI_0093.JPG")

hA

hA.save(corrected_filtered_path)