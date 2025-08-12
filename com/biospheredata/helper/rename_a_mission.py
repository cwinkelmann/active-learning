import shutil
from pathlib import Path

from com.biospheredata.helper.filenames import get_image_name
from com.biospheredata.types.Mission import Mission

def rename_mission(mission_path: Path, mission_path_renamed: Path, CRS = "EPSG:4326", suffix="JPG", delete=False):
    """
    Assign a unique id to the images

    @param mission_path:
    @param mission_path_renamed:
    @param CRS:
    @param suffix:
    @return:
    """
    try:
        m = Mission.open(mission_path)
    except Exception as e:
        images_list = mission_path.glob(f"*.{suffix}")
        from com.biospheredata.helper.image.identifier import get_image_id

        for mp in images_list:
            iid = get_image_id(mp)
            new_image_name = get_image_name(iid, suffix)

            if(mp != mission_path_renamed.joinpath(new_image_name)):

                if delete:
                    shutil.move(mp, mission_path_renamed.joinpath(new_image_name))
                else:
                    shutil.copyfile(mp, mission_path_renamed.joinpath(new_image_name))
        m = Mission.init(mission_path, CRS=CRS, suffix=suffix)
        m.persist()

    return m.base_path


if __name__ == '__main__':
    mission_path = Path("/home/christian/data/missions/2022_demo_cw/02.02.21/FMO01_small")
    mission_path_renamed_parts = list(mission_path.parts)
    mission_path_renamed_parts[-3] = "2022_demo_cw_renamed"
    mission_path_renamed = Path(*mission_path_renamed_parts)
    mission_path_renamed.mkdir(exist_ok=True, parents=True)

    # rename_mission(Path(mission_path), Path(mission_path_renamed))
    rename_mission(Path(mission_path), Path(mission_path))