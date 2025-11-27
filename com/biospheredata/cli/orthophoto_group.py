from com.biospheredata.cli.groups import *
from com.biospheredata.helper.run_pyodm import build_orthophoto
from com.biospheredata.types.Mission import Mission
from com.biospheredata.helper.image.manipulation.slice import slice_very_big_raster


@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where should the mission be saved to',
    help="""The name of the mission. When creating consider something very unique""",
)
@click.option(
    "--host",
    default="localhost",
    prompt='The machine on which nodeodm is running',
    help="""The machine on which nodeodm is running""",
)
@click.option(
    "--port",
    default="3000",
    prompt='the port of nodeodm',
    help="""The port of node odm""",
)
@click.option(
    "--preset_path",
    default=None,
    prompt="path to json defining the setting for opendronemap",
    help="""settings like feature-quality, orthophoto-resolution""",
)
@orthophoto.command()
def create(base_path, host, port, preset_path = None):
    """
    create an orthophoto based on the simple images.

    @param action:
    @param name:
    @return:
    """
    m = Mission.open(base_path=Path(base_path))
    logger.info(m)

    with open(preset_path, 'r') as f:
        preset = json.load(f)

    absolute_image_paths = m.get_images(absolute=True)
    output_path = build_orthophoto(absolute_image_paths,
                                   output_path=m.base_path.joinpath("odm"),
                                   logger=logger,
                                   host=host,
                                   port=port, preset=preset)

    m.orthophoto_path = Path(output_path)
    m.persist()
    return m


@click.option(
    "--orthophoto_path",
    default="/tmp",
    prompt='Where is the orthophoto',
)
# @click.option(
#     "--output",
#     prompt='where should the slices be stored',
# )
@click.option(
    "--slice_size",
    prompt='the size of the image slice',
    default="1200",
)
@orthophoto.command()
def slice(orthophoto_path, slice_size):
    """
    slice an orthophoto into smaller parts
    @param base_path:
    @param host:
    @param port:
    @param preset_path:
    @return:
    """
    base_path = Path(orthophoto_path)
    image_name = base_path.parts[-1]
    base_path = base_path.parent

    sliced_path, image_names, slice_dict = slice_very_big_raster(base_path, image_name, x_size=int(slice_size), y_size=int(slice_size),
                          FORCE_UPDATE=False, file_type="JPEG")

    logger.info(f"wrote slices to {sliced_path}")
    logger.info(f"wrote # {len(image_names)} slices.")