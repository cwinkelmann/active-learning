from pathlib import Path

import pytest

from active_learning.types.Mission import MissionV2


def test_setup_mission():
    image_folder = Path("/Users/christian/data/2TB/ai-core/data/fake_test_data/Floreana/FLPC02_22012023")
    mission_name = image_folder.parts[-1]
    m = MissionV2.init(base_path=image_folder, CRS="EPSG:32715", suffix="JPG")

    assert m.mission_name == "FLPC02_22012023"