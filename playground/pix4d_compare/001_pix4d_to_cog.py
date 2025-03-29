"""
The orthomosaics are not very well organised. This script will reorganise the orthomosaics into a more structured form
Each island code gets its own folder, drone deploy orthomosaic are converted into COGs and copied to the new folder
"""
import shutil
from loguru import logger
import rasterio
from pathlib import Path
import re
from geospatial_transformations import convert_to_cog, batch_convert_to_cog, warp_to_epsg, batch_warp_to_epsg
from osgeo import gdal

pix4d_path = Path('/Volumes/G-DRIVE/Iguanas_From_Above_Orthomosaics/Genovesa_122021/exports/ortho')
pix4d_cog_path = Path(pix4d_path / 'cog')


pix4d = [f for f in pix4d_path.glob('*.tiff') if not f.name.startswith('.')]
pix4d = [f for f in pix4d_path.glob('*.tiff')
         if not f.name.startswith('.') and f.stat().st_size > 10 * 1024 * 1024]


# copy images to new folder
output_DD_dir = pix4d_cog_path


def extract_epsg_from_named_projcs(wkt_string,):
    # Match the named PROJCS block

    KNOWN_PROJCS_EPSG = {
        "WGS 84 / UTM zone 16N": "EPSG:32616",
        "WGS 84 / UTM zone 15N": "EPSG:32615",
        "WGS 84 / UTM zone 16S": "EPSG:32716",
        "WGS 84 / UTM zone 15S": "EPSG:32715",
    }

    def extract_epsg_from_named_projcs(wkt_string):
        for projcs_name, epsg_code in KNOWN_PROJCS_EPSG.items():
            if f'PROJCS["{projcs_name}"' in wkt_string:
                return epsg_code
        raise ValueError("EPSG Not found")

    return extract_epsg_from_named_projcs(wkt_string)

output_files = []
output_prj_files = []

input_files = []
for i, img in enumerate(pix4d):


    # check for EPSG Code
    with rasterio.open(img) as src:
        # Get the CRS (Coordinate Reference System)
        crs = src.crs

        wkt = src.crs.wkt
        epsg = extract_epsg_from_named_projcs(wkt)

    img_name = img.name
    island_code = "GES"
    output_DD_dir.joinpath(island_code).mkdir(exist_ok=True, parents=True)
    output_file = output_DD_dir / island_code / img_name
    output_prj_files = output_DD_dir / island_code / Path(img_name.split(".")[0]).with_suffix(".prj")

    input_files.append(img)
    img_name_cog = f"{Path(img_name).stem}_cog.tiff"
    img_name_proj = f"{Path(img_name).stem}_cog.tiff"
    output_file_proj = output_DD_dir / island_code / img_name
    output_files.append(output_file)

    logger.info(f"convert {img} to cog at {output_DD_dir / island_code / img_name}")
    try:
        logger.info(f"warp {img} to proj at {epsg}")
        warp_to_epsg(img, output_DD_dir / island_code / img_name, target_epsg=epsg)

        logger.info(f"convert {img} to cog at {output_prj_files}")

        convert_to_cog(output_DD_dir / island_code / img_name, output_DD_dir / island_code / img_name_cog, overwrite=True)

        # gdal.Warp(output_DD_dir / island_code / img_name, output_DD_dir / island_code / img_name, dstSRS=epsg)
    except Exception as e:
        logger.error(f"Error converting {img} to cog: {e}")

batch_warp_to_epsg(input_files=input_files, output_files=output_files, output_dir=output_DD_dir, max_workers=3, target_epsg=epsg)
# batch_convert_to_cog(input_files=input_files, output_files=output_files, output_dir=output_DD_dir, max_workers=3)





