from dataclasses import dataclass

import fiona
from loguru import logger
from pathlib import Path

from com.biospheredata.helper.geospatial.GeospatialAnalysis import convert_points_to_lines, process_coastlines
from com.biospheredata.types.ExpeditionOverviewV2 import ExpeditionOverviewV2
from com.biospheredata.types.Mission import Mission


@dataclass()
class MissionMetaData(object):
    """
    take all image of a mission and calculate interesting metadata like flight path, area... then convert it to format you are interested in
    """

    def __init__(self, mission: Mission, metadata_folder: Path, projected_CRS="EPSG:32715"):
        self.mission = mission

        metadata_folder.mkdir(exist_ok=True, parents=True)
        self.metadata_folder = metadata_folder

        self.projected_CRS = projected_CRS


    def extract_mission_metadata(self, ):
        """
        TODO implement this right when its done
        :return:
        """

        logger.info(f"generating metadata in suitable form. return : {self.mission}")

        mission_name = self.mission.mission_name
        mission_date = self.mission.creation_date
        expedition_name = self.mission.expedition_name

        logger.info(f"copy geospatial metadata into ")

        # metadata_folder = Path(context.op_config["metadata_folder"])
        logger.info(f"packed_path : {self.metadata_folder}")

        self.mission.to_dataframe().to_file(str(Path(self.metadata_folder)
                                           .joinpath(f"./{mission_name}_photo_points.geojson")),
                                       driver="GeoJSON")

        gdf = convert_points_to_lines(gdf_photopoints=self.mission.to_dataframe(),
                                      projected_CRS=self.projected_CRS,
                                      output_path=self.metadata_folder,
                                      prefix=mission_name)


        self.mission.to_dataframe().to_file(str(Path(self.metadata_folder)
                                           .joinpath(f"./{mission_name}_photo_points.geojson")),
                                       driver="GeoJSON")
        self.mission.to_dataframe().to_csv(str(Path(self.metadata_folder)
                                                .joinpath(f"./{mission_name}_photo_points.csv"),
                                            ), index=False
                                            )
        fiona.supported_drivers['KML'] = 'rw'
        kml_file = Path(self.metadata_folder).joinpath(f"./{mission_name}_photo_points.KML")
        if kml_file.is_file():
            ## For a strange reason the library can't overwrite the kml file
            kml_file.unlink()
        self.mission.to_dataframe().to_file(str(kml_file), driver='KML')

        return gdf

    def get_expedition_level_summary(self, expedition_base_paths, metadat_path):
        """
        summarise an expedition into a dataframe

        :param expedition_base_paths:
        :param metadat_path:
        :return:
        """
        statistics_result = {}

        exp_Overview = ExpeditionOverviewV2(
            expeditions=expedition_base_paths,
            expedition_overview_path=metadat_path
        )

        exp_O_df = exp_Overview.expedition_level_statistics()
        logger.info(exp_O_df)
        return exp_O_df

