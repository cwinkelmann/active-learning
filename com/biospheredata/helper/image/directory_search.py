import glob
from pathlib import Path


def find_orthophoto(MY_DIRECTORY):
    """
    look for all orthophotos in an expedition date folder
    :param MY_DIRECTORY:
    """
    for filename in glob.glob(f"{MY_DIRECTORY}/*/output/odm_orthophoto/odm_orthophoto.tif"):
        # for filename in os.listdir(MY_DIRECTORY):
        filepath = Path(filename).resolve()
        # if os.path.isfile(filepath):
        base_path = filepath.parts[:-2]
        file_name = filepath.parts[-2:]
        mission_name = filepath.parts[-4]
        yield str(Path(*base_path)), str(Path(*file_name)), str(mission_name)


if __name__ == '__main__':
    for x, y in find_orthophoto("/home/christian/mount/hetzner_webdav/ai-core/data/object_detection/live.biospheredata.com/temp/missions"):
        print(x)
        print(y)
