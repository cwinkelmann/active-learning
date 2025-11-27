from com.biospheredata.cli.groups import *
from com.biospheredata.helper.run_pyodm import build_orthophoto
from com.biospheredata.types.ExpeditionOverviewV2 import ExpeditionOverviewV2
from com.biospheredata.types.Expedition import Expedition




@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where should the mission be saved to',
    help="""The name of the mission. When creating consider something very unique""",
)
@click.option(
    "--expedition_name",
    help="""The name of the expedition you want to create""",
)
@click.option(
    "--crs",
    default="EPSG:4326",
    # type=click.Choice(["create", "load"], case_sensitive=True),
    help="""Coordindate Reference System (CRS)""",
)
@click.option(
    "--suffix",
    default="JPG",
    help="""suffix of the images. If you have images from different sources and some are named *.jgp, some *.JPEG you have problem.""",
)
@expedition.command()
def create(base_path, expedition_name, crs, suffix):
    """
    create an Expedition. An Expedition contains many missions on multiple dates.

    :param base_path:
    :param expedition_name:
    :param crs:
    :param suffix:
    :return:
    """
    print("expedition init")

    e = Expedition(
        expedition_name=expedition_name,
        expedition_path=base_path)

    e.resolve_missions()
    e.persist()


@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where should the mission be saved to',
    help="""The name of the mission. When creating consider something very unique""",
)
@click.option(
    "--host",
    default="localhost",
    prompt="The machine on which nodeodm is running",
    help="""The machine on which nodeodm is running""",
)
@click.option(
    "--port",
    default="3000",
    prompt='the port of nodeodm',
    help="""The port of node odm""",
)
@expedition.command()
def orthophoto(base_path, host, port):
    """
    generate the orthophoto right in here
    WARNING: it takes a long time to do this and if your expedition is quite big you don't want to do this, especially as this doesn't keep the state after a failure
    @param base_path:
    @param host:
    @param port:
    @return:
    """
    e = Expedition.open(expedition_folder=Path(base_path))
    missions = []
    for m in e.get_missions():
        absolute_image_paths = m.get_images(absolute=True)
        output_path = build_orthophoto(absolute_image_paths,
                                       output_path=m.base_path.joinpath("odm"),
                                       logger=logger,
                                       host=host,
                                       port=port)

        m.orthophoto_path = Path(output_path)
        m.persist()
        missions.append(m)

    e.set_missions(missions)
    e.persist()



@click.option(
    "--base_path",
    default="/tmp",
    prompt='Where should the Expedition Base Path which includes all missions',
    help="""Where should the Expedition Base Path which includes all missions""",
)
@click.option(
    "--report_folder",
    default="./report",
    prompt='where the report should be saved to',
)
@click.option(
    "--expedition_name_schema",
    default=Expedition.ISLAND_LOCATIONCODEDATE,
    prompt=f"The Name schema of the Expedition. {Expedition.ISLAND_LOCATIONCODEDATE} is the default for Jan 2023",
    help=f"The Name schema of the Expedition. {Expedition.ISLAND_LOCATIONCODEDATE} is the default for Jan 2023",
)
@click.option(
    "--source_crs",
    default="EPSG:4326",
    prompt=f"projected Coordinate reference system for measuring i.e. lengths, area",
    help=f"projected Coordinate reference system for measuring i.e. lengths, area",
)
@click.option(
    "--projected_crs",
    default="EPSG:32715",
    prompt=f"projected Coordinate reference system for measuring i.e. lengths, area",
    help=f"projected Coordinate reference system for measuring i.e. lengths, area",
)
@click.option(
    "--init",
    default=True,
    prompt=f"Are the missions new or should the metadata be updated?",
    help=f"Are the missions new or should the metadata be updated?",
)
@expedition.command()
def overview(base_path, report_folder, source_crs, projected_crs, expedition_name_schema, init):
    """
    generate the overview of expeditions either for each image individually

    expedition overview --base_path="/Users/christian/data/missions/Jan 2023/Isabela/ISTOSI01_17012023" --init=True --projected_CRS=EPSG:32715 --source_CRS=EPSG:4326 --expedition_name_schema=ISLAND_LOCATIONCODEDATE


    :param source:
    :param base_path:
    :param level:
    :param report_folder:
    :return:
    """


    exp = Expedition(expedition_name="B_Name",
                       expedition_path=base_path,
                       schema=expedition_name_schema,
                       source_CRS=source_crs)
    exp.resolve_missions(init=init)

    exp_Overview = ExpeditionOverviewV2(
        expeditions=[exp],
        expedition_overview_path=Path(report_folder)
    )
    exp_Overview.persist()

    exp_Overview.to_dataframe()
    exp_O_df = exp_Overview.expedition_level_statistics(projected_CRS=projected_crs)

    return exp_O_df.to_dict()



