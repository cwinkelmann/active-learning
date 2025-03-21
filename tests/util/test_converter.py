from pathlib import Path

import pytest

from active_learning.util.converter import dota2coco


@pytest.fixture()
def dota_annoatations():
    input_data = """imagesource:GoogleEarth
    gsd:0.504225526997
    8184.0 364.0 8189.0 364.0 8188.0 375.0 8183.0 376.0 small-vehicle 1
    4825.0 1314.0 4821.0 1314.0 4821.0 1304.0 4824.0 1304.0 small-vehicle 1
    4875.0 1166.0 4873.0 1161.0 4884.0 1158.0 4885.0 1163.0 small-vehicle 1
    4867.0 1178.0 4866.0 1173.0 4876.0 1168.0 4877.0 1173.0 small-vehicle 1
    4853.0 1186.0 4851.0 1181.0 4858.0 1176.0 4861.0 1181.0 small-vehicle 1
    4841.0 1198.0 4837.0 1195.0 4844.0 1186.0 4847.0 1189.0 small-vehicle 1
    4831.0 1210.0 4827.0 1207.0 4831.0 1199.0 4834.0 1202.0 small-vehicle 1
    4851.0 1274.0 4847.0 1272.0 4847.0 1259.0 4851.0 1260.0 small-vehicle 1
    4845.0 1271.0 4839.0 1270.0 4839.0 1259.0 4843.0 1259.0 small-vehicle 1
    4825.0 1297.0 4819.0 1296.0 4820.0 1285.0 4824.0 1285.0 small-vehicle 1
    4891.0 1177.0 4891.0 1173.0 4900.0 1171.0 4901.0 1175.0 small-vehicle 1"""

    return input_data

def dota_path():
    return Path(__file__).parent.parent / "tests/util/data/dota/P1868_hbb.txt"

def test_dota2coco(dota_annoatations):
    res = dota2coco(dota_path)

