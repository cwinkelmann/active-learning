from pathlib import Path

import pytest

from active_learning.types.Mission import MissionV2
from active_learning.types.MissionStatistics import MissionStatistics


def test_setup_mission():
    """
    Load a flight mission into a Mission object, calculate some statistics on top of it
    :return:
    """
    image_folder = Path("/Users/christian/data/2TB/ai-core/data/fake_test_data/Floreana/FLPC02_22012023")
    mission_name = image_folder.parts[-1]
    m = MissionV2.init(base_path=image_folder, CRS="EPSG:32715", suffix="JPG")

    assert m.mission_name == "FLPC02_22012023"

    gdf = m.get_geodata_frame(projected_CRS="EPSG:32715")

    ms = MissionStatistics(m, projected_CRS="EPSG:32715")

    assert ms.photo_series_length == 580.0, "580 seconds between first and last image"