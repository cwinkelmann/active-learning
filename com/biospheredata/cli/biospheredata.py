from com.biospheredata.converter.HastyToYoloConverter import whole_workflow
from com.biospheredata.converter.MissionMetaData import MissionMetaData
from com.biospheredata.converter.YoloTiler import YoloTiler
from com.biospheredata.helper.candidate_proposal import geospatial_prediction
from com.biospheredata.types.CandidateProposal import CandidateProposal
from com.biospheredata.cli.expedition_group import *
from com.biospheredata.cli.zooniverse_group import *
from com.biospheredata.cli.orthophoto_group import *
from com.biospheredata.cli.groups import *

"""
Command line interface for all major tasks in the biospheredata project

These are
* create a mission from a folder
* create/refresh a whole expedition which multiple missions
* generate a usable orthomosaic
* Generate training data from corrected annotations (hasty or yolo) by slicing, train/test split and config data generation
* candidate proposals to be corrected by an annotation tool

in a development context you should add the code path to PYTHONPATH:

PYTHONPATH=$PYTHONPATH:../../../ python ./biospheredata.py mission create --name test_bla

"""

a_global_variable = None


@click.option(
    "--base_path",
    default="/tmp",
    prompt='folder which contains the downloaded zips/folders',
    help="""The name of the mission. When creating consider something very unique""",
)
@click.option(
    "--labels",
    default="/labels.zip",
    help="""The name of the label folder or zip file which were downloaded from hasty""",
)
@click.option(
    "--images",
    default="/images.zip",
    help="""The name of the images folder or zip file which were downloaded from hasty""",
)
@click.option(
    "--amount_training_images",
    default="[20]",
    help="""a list of numbers which depicts the amount of images in the training data""",
)
@click.option(
    "--max_background_images",
    default="[0]",
    help="""a list of numbers which background images""",
)
@click.option(
    "--folds",
    default="2",
    help="""how many folds do you want to have""",
)
@click.option(
    "--slice_size",
    default=1280,
    help="""size of the training images""",
)
@click.option(
    "--prefix",
    default="rtu_tds_150_400",
    help="""the folder naming prefix with no spaces""",
)
@click.option(
    "--full_data_processing",
    default=False,
    help="""should the images be unpacked and sliced? Chone False if that is already done.""",
)
@click.option(
    "--extension",
    default="JPG",
    help="""Are the images JPG, Jpg, jpg or png or ...""",
)
@cli.command()
def hasty2yolo(base_path, labels, images,
               amount_training_images, max_background_images,
               folds, slice_size, prefix, full_data_processing, extension
               ):
    """
    PYTHONPATH=$PYTHONPATH:./ python3 ./com/biospheredata/cli/biospheredata.py hasty2yolo --base_path=/home/christian/work/object-detection-pytorch/tests_data/hasty_turtles_2022-08-06 --labels=MRCI_turtles_labels.zip --images=MRCI_turtles_images.zip
    @param base_path:
    @param labels:
    @param images:
    @return:
    """

    full_data_processing = bool(full_data_processing)

    result_workflow = whole_workflow(
        Path(base_path),
        Path(labels),
        Path(images),
        amount_training_images=json.loads(amount_training_images),
        backgrounds=json.loads(max_background_images),
        folds=int(folds),
        fulls=full_data_processing,
        slice_size=slice_size,
        prefix=prefix,
        extension=extension
        )
    logger.info(list(result_workflow))


@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where should the mission be saved to',
    help="""The folder the images of the missions are in.""",
)
@click.option(
    "--metadata_path",
    default=None,
    help="""The location extracted metadata should be saved to. Defaults to the "metadata" in the mission folder""",
)
@click.option(
    "--suffix",
    default="JPG",
    help="""Coordindate Reference System (CRS)""",
)
@click.option(
    "--crs",
    default="EPSG:4326",
    # type=click.Choice(["create", "load"], case_sensitive=True),
    help="""Coordindate Reference System (CRS)""",
)
@mission.command()
def create(base_path, metadata_path, suffix, crs):
    """ create a mission """
    print("mission init")

    m = Mission.init(base_path=Path(base_path), suffix=suffix, CRS=crs)
    m.persist()

    if metadata_path is None:
        metadata_path = m.base_path.joinpath("metadata")
    else:
        metadata_path = Path(metadata_path)
    mmd = MissionMetaData(mission=m, metadata_folder=metadata_path)
    mmd.extract_mission_metadata()


@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where should the mission be saved to',
    help="""The name of the mission. When creating consider something very unique""",
)
@click.option(
    "--format",
    default="GeoJSON",
    help="""convert a mission into something""",
    prompt="Shapefile GeoJSON or KML"
)
@mission.command()
def export(base_path):
    """ export a mission metadata definition to another format """

    m = Mission.open(base_path=Path(base_path))
    georeferenced_image_metadata = m.get_georeferenced_image_metadata()
    # TODO implement this correctly
    ## TODO this refers to the metadata export




@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where is the mission folder with the images',
)
@click.option(
    "--model_path",
    default="/tmp",
    prompt='Location of the model',
)
@click.option(
    "--label",
    prompt='the label you look for',
)
@click.option(
    "--output",
    prompt='where should they be stored',
)
@click.option(
    "--slice_size",
    prompt='the size of the image slice which YOLO is trained on',
    default="1280",
)
@click.option(
    "--image_size",
    prompt='the size of the image the image is tiled into',
    default=-1,
)
@click.option(
    "--mosaic",
    prompt='if the mosaic should be used',
    default=False,
)
@mission.command()
def candidate(base_path, model_path, label, output, slice_size, image_size, mosaic):
    """
    generate a list of candidate proposals
     PYTHONPATH=$PYTHONPATH:../../../ python3 biospheredata.py mission candidate --base_path=/tmp/missions/2022_demo_cw/12.01.21/EGB01  --model_path=/home/christian/data/models/iguana_n6l_82b6a0257af284c58c9dc84452c1509b.pt --label=iguana --output=/tmp/candidate_proposals/

    """

    if image_size == -1:
        image_size = None
    else:
        image_size = int(image_size)


    m = Mission.open(base_path=Path(base_path))
    logger.info(m)

    candidate_proposal_path = CandidateProposal.generate_candidate_proposals(missions=[m],
                                                                             yolov5_model_path=model_path,
                                                                             searched_label=label,
                                                                             output=Path(output),
                                                                             image_size=image_size,
                                                                             slice_size=int(slice_size),
                                                                             mosaic=mosaic
                                                                             )
    logger.info(f"Wrote candidate proposal to {candidate_proposal_path}")

    m.persist()


@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where is the mission folder with the images',
)
@click.option(
    "--model_path",
    default="/tmp",
    prompt='Location of the model',
)
# @click.option(
#     "--output",
#     prompt='where should they be stored',
# )
@click.option(
    "--slice_size",
    prompt='the size of the image slice which YOLO is trained on',
    default="640",
)
@click.option(
    "--image_size",
    prompt='the size of the image the image is tiled into',
    default=5000,
)

@mission.command()
def geospatial(base_path, model_path, slice_size, image_size):
    """
    geospatial prediction
    """

    if image_size == -1:
        image_size = None
    else:
        image_size = int(image_size)

    slice_size = int(slice_size)


    m = Mission.open(base_path=Path(base_path))
    logger.info(m)

    predictions_geojson, image_names, sliced_image_path = geospatial_prediction(base_path=m.base_path,
                                                                                image_name=m.orthophoto_path,
                                                                                model_path=model_path,
                                                                                image_size=image_size,
                                                                                logger=logger,
                                                                                slice_size=slice_size
                                                                                )

    predictions_geojson_path = str(sliced_image_path.joinpath(predictions_geojson))

    logger.info(f"Wrote candidate proposal to {predictions_geojson_path}")

    m.persist()

    return m, predictions_geojson_path

@click.option(
    "--mission_path",
    default="/tmp",
    prompt='Where is the mission folder with the images',
)
@click.option(
    "--output",
    prompt='where should the slices be stored',
)
@click.option(
    "--slice_size",
    prompt='the size of the image slice which YOLO is trained on',
    default="1200",
)
@mission.command()
def simple_slice(mission_path, output, slice_size):
    """
    generate a list of candidate proposals
     PYTHONPATH=$PYTHONPATH:../../../ python3 biospheredata.py mission candidate --base_path=/tmp/missions/2022_demo_cw/12.01.21/EGB01  --model_path=/home/christian/data/models/iguana_n6l_82b6a0257af284c58c9dc84452c1509b.pt --label=iguana --output=/tmp/candidate_proposals/

    @param action:
    @param name:
    @return:
    """
    m = Mission.open(base_path=Path(mission_path))
    logger.info(m)

    image_slices = YoloTiler.simple_images_slice(
        slice_size=int(slice_size),
        newpath=Path(output),
        images_paths=m.get_images(absolute=True)
    )

    logger.info(f"generated images {image_slices}")
    logger.info(f"Wrote sliced images to {output}")

    m.persist()
    return m




if __name__ == "__main__":
    #  PYTHONPATH=$PYTHONPATH:../../../ python ./biospheredata.py mission --base_path="/tmp/missions/2022_demo_a/2022-10-31/with_iguanas_big/" add_orthophoto --host=q6600 --port 3001
    cli()
