"""
The orthomosaics are not very well organised. This script will reorganise the orthomosaics into a more structured form
Each island code gets its own folder, drone deploy orthomosaic are converted into COGs and copied to the new folder
"""
import shutil
from loguru import logger

from pathlib import Path

from active_learning.util.geospatial_transformations import batch_convert_to_cog

# dd_cog_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog')
# dd_path = Path('/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/.shortcut-targets-by-id/1u0tmSqWpyjE3etisjtWQ83r3cS2LEk_i/Manual Counting /Drone Deploy orthomosaics')
metashape_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Agisoft orthomosaics')

# dd = [f for f in dd_path.glob('*.tif') if not f.name.startswith('.')]
# dd_cog = [f for f in dd_cog_path.glob('*.tif')  if not f.name.startswith('.')]
ms = [f for f in metashape_path.glob('*/*.tif')  if not f.name.startswith('.')]

# copy images to new folder
# output_DD_dir = dd_cog_path
output_MS_dir = Path("/Volumes/2TB/Manual_Counting/AgisoftMetashape Orthomosaics")
# output_DD_dir.mkdir(exist_ok=True)
output_MS_dir.mkdir(exist_ok=True)

# for i, img in enumerate(ms):
#     img_name = img.name
#     island_code = img_name.split('_')[0]
#     output_MS_dir.joinpath(island_code).mkdir(exist_ok=True, parents=True)
#     output_file = output_MS_dir / island_code / img_name
#
#     if output_file.exists():
#         logger.info(f"Metashape created File already exists: {output_file}")
#     else:
#         logger.info(f"creating cog out of metashape {img} to {output_file}")
#         convert_to_cog(img, output_file)

# for i, img in enumerate(dd_cog):
#     img_name = img.name
#     island_code = img_name.split('_')[0]
#     output_MS_dir.joinpath(island_code).mkdir(exist_ok=True, parents=True)
#     output_file = output_MS_dir / island_code / img_name
#     if output_file.exists():
#         logger.info(f"File already exists: {output_file}")
#     else:
#         logger.info(f"Copying Drone deploy COG {img} to {output_MS_dir / island_code / img_name}")
#         shutil.copy(img, output_file)


output_files = []
input_files = []
for i, img in enumerate(ms):
    img_name = img.name
    island_code = img_name.split('_')[0]
    output_MS_dir.joinpath(island_code).mkdir(exist_ok=True, parents=True)
    output_file = output_MS_dir / island_code / img_name
    if output_file.exists():
        logger.info(f"MS File already exists: {output_file}")
    else:
        input_files.append(img)
        output_file = output_MS_dir / island_code / img_name
        output_files.append(output_file)

        logger.info(f" {i}/{len(ms):} convert {img} to cog at {output_MS_dir / island_code / img_name}")
        # try:
        #     convert_to_cog(img, output_DD_dir / island_code / img_name)
        # except Exception as e:
        #     logger.error(f"Error converting {img} to cog: {e}")

batch_convert_to_cog(input_files=input_files, output_files=output_files, output_dir=output_MS_dir, max_workers=3)


