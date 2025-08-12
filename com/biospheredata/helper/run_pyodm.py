## docker run -ti -p 3000:3000 opendronemap/nodeodm

"""

docker run -ti --rm -v "/media/christian/2TB/tmp":/datasets/code opendronemap/odm --project-path /datasets --orthophoto-resolution 1
"""
import copy
import glob
import os
from pathlib import Path
import random

from pyodm import Node
import random

from time import sleep


def get_images(base_path):
    image_list = glob.glob(
        str(base_path.joinpath("*.JPG")))

    return image_list


PRESET_PREDICTION_GRADE_QUALITY = {
    'dsm': True, # Digital Surface Model
    'cog': True, # Cloud Optimized GeoTIFF
    'orthophoto-resolution': 0.6, # Ground Sample Distance of Orthophoto
    # "orthophoto-kmz": True,
    'build-overviews': True, # Internal lower resolution overview images to speed up display
    'tiles': True, # split orthophoto into Tiles
    "verbose": True,
    "max_concurrency": 4,
    # https://docs.opendronemap.org/sw/arguments/dem-decimation/#
    "dem-decimation": 50, # reduce points for DEM generation
    # https://docs.opendronemap.org/arguments/fast-orthophoto/
    "feature-quality": "high",
    "split": 300,
    "split-overlap": 100
}

PRESET_MEDIUM_GRADE_QUALITY = {
    'dsm': False,
    # 'dtm': True,
    'cog': True,
    'orthophoto-resolution': 2.0,
    # "orthophoto-kmz": True,
    'build-overviews': True,
    'tiles': False,
    "verbose": True,
    # "max_concurrency": 4,
    # "skip-3dmodel": True,
    # https://docs.opendronemap.org/sw/arguments/dem-decimation/#
    # "dem-decimation": 50,
    # https://docs.opendronemap.org/arguments/fast-orthophoto/
    "fast-orthophoto": True,
    # "radiometric-calibration": "camera" # https://docs.opendronemap.org/arguments/radiometric-calibration/

    # 'orthophoto-compression': 'JPEG',

    # "pc-las": True,
    # "pc-geometric": True,
    # "pc-rectify": True,
    # "pc-quality": "high",

    # "mesh-octree-depth": 13,
    # "mesh-size": 300000,

    "feature-quality": "low",
    # "min-num-features": 12000,
    # "texturing-data-term": "area",
    # "use-3dmesh": True,

    "split": 100,
    "split-overlap": 50
}

PRESET_OVERVIEW_QUALITY = {
    'dsm': False,
    # 'dtm': True,
    'cog': True,
    'orthophoto-resolution': 5,
    # "orthophoto-kmz": True,
    'build-overviews': True,
    'tiles': True,
    "verbose": True,
    # "max_concurrency": 4,
    "skip-3dmodel": True,
    # https://docs.opendronemap.org/sw/arguments/dem-decimation/#
    "dem-decimation": 100,
    # https://docs.opendronemap.org/arguments/fast-orthophoto/
    "fast-orthophoto": True,
    "feature-quality": "low"

}


def find_empty_node(hostname, ports):
    """

    @param hostname:
    @param ports:
    @return:
    """
    ports_copy = list(copy.deepcopy(ports))
    random.shuffle(ports_copy)
    shortest_queue_length = 10000
    port_of_shortest_q_length = 0
    for port in ports_copy:
        node = Node(hostname, port)
        queue_length = node.info().task_queue_count
        if queue_length < shortest_queue_length:
            shortest_queue_length = queue_length
            port_of_shortest_q_length = port

    return port_of_shortest_q_length


def build_orthophoto(images, output_path, logger, host="localhost", port=3000, preset=PRESET_PREDICTION_GRADE_QUALITY):
    """
    submit images to opendronemap
    :param images:
    :param output_path:
    :param logger:
    :return:
    """
    host = os.getenv("OPENDRONEMAP_HOST", host)
    port = int(os.getenv("OPENDRONEMAP_PORT", port))
    if host is None:
        raise ValueError("set OPENDRONEMAP_HOST")
    n_1 = Node(host, port)

    logger.info(f"node 1 info: {n_1.info()} before submission.")
    logger.info(f"create task about this mission {images}")
    images = [str(i) for i in images]

    # n = n_1
    task_node = n_1
    #    task_node = n_1 if n_1.info().task_queue_count <= n_2.info().task_queue_count else n_2

    logger.info(f"create task with {len(images)} images at host: {task_node.host} and port {port}")

    # sleep(random.randint(1, 50))

    task = task_node.create_task(files=images,
                                 ## https://docs.opendronemap.org/arguments/orthophoto-compression/
                                 # https://docs.opendronemap.org/arguments/
                                 options=preset,
                                 max_retries=15, retry_timeout=15
                                 )

    def status_callback(d):
        logger.info(f"status of odm task: {d}")
        ## todo logging to grafana

    logger.info(f"wait for the process to finish.")
    logger.info(f"node info: {task_node.info()} after submission.")
    task.wait_for_completion(status_callback=status_callback, interval=60)

    logger.info(f"download assets")
    task.download_assets(output_path)

    task.remove()  ## remove the output after it was downloaded to prevent disk overflow

    logger.info(f"done with the mosaicing")
    logger.info(f"wrote result to {output_path}")

    return Path(output_path).joinpath("odm_orthophoto/odm_orthophoto.tif")
