import shutil

import os
from loguru import logger
from pathlib import Path


def get_testdata_base_path() -> Path:
    """
    resolve the location of the testdata based on the currently set env variable
    @return:
    """

    current_path = Path(__file__).parent
    base_path = Path(
        os.getenv("TEST_DATA_PATH",
        current_path.joinpath("./data/").resolve())
    )
    return base_path

def get_integration_testdata_base_path() -> Path:
    """
    resolve the location of the testdata based on the currently set env variable
    @return:
    """

    current_path = Path(__file__).parent
    base_path = Path(
        os.getenv("INTEGRATION_TEST_DATA_PATH",
        current_path.joinpath("../data/").resolve())
    )
    return base_path



def get_test_odm_host():
    odm_host = os.getenv("TEST_ODM_HOST", "live.biospheredata.com")
    return odm_host


def get_test_odm_port():
    odm_port = os.getenv("TEST_ODM_PORT", "3001")
    return odm_port


def cleanup_test_tmp_directory() -> Path:
    testdata_base_path = get_testdata_base_path().joinpath("tmp")
    try:
        shutil.rmtree(testdata_base_path)
    except Exception as e:
        logger.info(e)

    testdata_base_path.mkdir()

    return testdata_base_path


def cleanup_tmp_path(tmp_path):
    try:
        shutil.rmtree(tmp_path)
        logger.info(f"deleted {tmp_path}")
    except Exception as e:
        logger.info(f"problem with deleteing tmp folder: {e}")