from __future__ import annotations

import fiona
import json
from pathlib import Path

from loguru import logger

from com.biospheredata.helper.geospatial.GeospatialAnalysis import convert_points_to_lines, convert_points_to_buffer
from com.biospheredata.types.Mission import Mission, NoImagesException
import pandas as pd


class Expedition(object):
    """
    Summary a whole expedition including multiple Missions on multiple days
    """

    mission_paths = []
    expedition_name = None
    expedition_path = None
    EXPEDITION_FILE_NAME = "expedition_metadata.json"
    missions = []

    ISLAND_DATE_LOCATIONCODE = "ISLAND_DATE_LOCATIONCODE"
    ISLAND_LOCATIONCODEDATE = "ISLAND_LOCATIONCODEDATE"
    ISLAND_DRONE_DATE_LOCATIONCODE = "ISLAND_DRONE_DATE_LOCATIONCODE"

    def __init__(self, expedition_name,
                 expedition_path: Path,
                 schema: str,
                 suffix: str = "JPG",
                 source_CRS: str = "EPSG:4326"
                 ):
        """

        :param expedition_name:
        :param expedition_path:
        :param schema:
        :param suffix: i.e. JPG without dot or *
        :param source_CRS:
        """
        self.expedition_name = expedition_name
        self.expedition_path = Path(expedition_path)
        self.suffix = suffix
        self.CRS = source_CRS
        self.schema = schema
        self.missions = []
        self._resolve_missions()

    def add_mission(self, mission: Mission):
        self.missions.append(mission)
        self.mission_paths.append(mission.base_path)

    def _resolve_missions(self):
        if self.schema == self.ISLAND_DATE_LOCATIONCODE:
            self.mission_paths = find_mission_folders(self.expedition_path)
        elif self.schema == self.ISLAND_LOCATIONCODEDATE:
            self.mission_paths = find_mission_folders_Jan_2023(self.expedition_path)
        elif self.schema == self.ISLAND_DRONE_DATE_LOCATIONCODE:
            self.mission_paths = find_mission_folders_Dec2021(self.expedition_path)
        else:
            raise Exception("wrong schema supplied")

        self.resolve_missions()

    def get_missions(self):
        return self.missions

    def resolve_missions(self, init=False):
        """

        :param init:
        :return:
        """
        # self.mission_paths = self._resolve_missions()
        self.missions = []
        for mp in self.mission_paths:

            if init:
                mission = Mission.init(mp, suffix=self.suffix, CRS=self.CRS)
                mission.persist()
            try:
                self.missions.append(
                    Mission.from_json_file(Path(mp).joinpath(Mission.metadata_fp)))
            except Exception as e:
                logger.info(f"No Mission metadata config found at {mp}")
                try:
                    mission = Mission.init(mp,
                                           suffix=self.suffix,
                                           CRS=self.CRS)

                    self.missions.append(mission)
                    logger.info(f"persist NEW missions to disk: {mp}")

                    mission.persist()
                except NoImagesException as e:
                    logger.warning(e)

        # logger.warning(f"DISABLED: persist all missions on disk")
        # persist all missions on disk
        #logger.info(f"persist all missions on disk")
        # [m.persist() for m in self.missions]
        #logger.info(f"DONE persisting all missions on disk")

    def load_all_mission(self):
        self.mission_paths = self._resolve_missions()
        self.missions = [Mission.open(m) for m in self.mission_paths]

    def to_dict(self):
        return {
            "expedition_name": self.expedition_name,
            "expedition_path": str(self.expedition_path),
            # "missions": [m.to_dict() for m in self.missions],
            "CRS": self.CRS,
            "mission_paths": [str(p) for p in self.mission_paths],
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(expedition_dict) -> Expedition:
        expedition_name = expedition_dict["expedition_name"]
        expedition_path = expedition_dict["expedition_path"]
        CRS = expedition_dict["CRS"]
        expedition = Expedition(expedition_name=expedition_name,
                                expedition_path=expedition_path,
                                source_CRS=CRS,
                                suffix="JPG", schema=Expedition.ISLAND_DATE_LOCATIONCODE)  # TODO this does not make sense

        missions = []

        for m in expedition_dict["mission_paths"]:
            try:
                obj_mission = Mission.from_json_file(Path(m).joinpath("mission_metadata.json"))
                missions.append(obj_mission)
            except FileNotFoundError as e:
                logger.warning(f"the was an empty folder in {m}")

        expedition.set_missions(missions)
        return expedition

    @staticmethod
    def from_json_file(filepath: Path):
        with open(filepath, 'r') as f:
            expedition_dict = json.load(f)

        return Expedition.from_dict(expedition_dict)

    def set_mission_paths(self, mission_paths):
        missions = [Mission.open(m) for m in mission_paths]
        self.set_missions(missions)
        self.mission_paths = mission_paths

    def set_missions(self, missions: list[Mission]):
        self.missions = missions

    @staticmethod
    def get_missions_from_expedition_folder(base_path):
        """

        @param base_path:
        @return:
        """
        import glob
        from pathlib import Path
        from com.biospheredata.types.Mission import Mission

        ### Now we have a lot of metadata jsons which store information like image dimension etc.
        missions = []

        missions_json_reps = glob.glob(f"{base_path}/**/mission_metadata.json", recursive=True)
        # print(missions_json_reps)

        for missions_json_rep in missions_json_reps:
            mission_loaded = Mission.from_json_file(Path(missions_json_rep))
            missions.append(mission_loaded)

        return missions

    def persist(self):
        """

        safe the metadata from the expedition

        @param path:
        @return:
        """
        self.mission_metadata_paths = [str(m.persist()) for m in self.missions]

        metadata_path = Path(self.expedition_path).joinpath(self.EXPEDITION_FILE_NAME)
        with open(metadata_path, 'w+') as f:
            json.dump(self.to_dict(), fp=f)

        return metadata_path

    def get_mission_by_name(self, mission_name):
        for m in self.missions:
            if mission_name == m.mission_name:
                return m

    @staticmethod
    def open(expedition_folder):
        expedition_metadata = expedition_folder.joinpath(Expedition.EXPEDITION_FILE_NAME)
        e1 = Expedition.from_json_file(expedition_metadata)

        return e1

    def to_dataframe(self):
        """
        build a flat representation of the Expedition
        :return:
        """
        m_df = []
        ## TOOD get all missions
        for m in self.get_missions():
            assert isinstance(m, Mission)
            m_df = m_df + m.to_dataframe().to_dict(orient="records")

        m_df = pd.DataFrame.from_records(m_df)
        m_df["expedition_name"] = self.expedition_name
        return m_df

    def mission_summary(self, metadata_folder: Path,
                        projected_CRS: str = "EPSG:32715",
                        type: str = "buffer"):
        """
        summarises each mission into a line, buffer and multipoint

        :metadata_folder: Folder to
        :return:
        """

        buffer_list = []
        multipoint_list = []
        line_list = []
        dissolved_buffer_list = []
        images_taken = 0
        list_images_taken = []
        photo_point_list = []

        ## TODO don't I have this in MissionMetaData
        for m in self.get_missions():
            assert isinstance(m, Mission)
            m.to_dataframe()

            ## TODO all the code is implement here, but juggling with the filenames sucks. I have to do it later to finish my presentation. Sorry to my future self if this is still in here by end of 2023.
            ## TODO there is a circular import too
            #mmd = MissionMetaData(mission=m, metadata_folder=metadata_folder)
            #mmd.extract_mission_metadata()

            dissolved_buffer = convert_points_to_buffer(gdf_photopoints=m.to_dataframe(),
                                                                projected_CRS=projected_CRS,
                                                                output_path=metadata_folder,
                                                                prefix=m.mission_name)

            lines, buffer, multipoint = convert_points_to_lines(gdf_photopoints=m.to_dataframe(),
                                                                projected_CRS=projected_CRS,
                                                                output_path=metadata_folder,
                                                                prefix=m.mission_name)

            m.to_dataframe().to_file(str(Path(metadata_folder)
                                                    .joinpath(f"./{m.mission_name}_photo_points.geojson")),
                                                driver="GeoJSON")
            m.to_dataframe().to_csv(str(Path(metadata_folder)
                                                   .joinpath(f"./{m.mission_name}_photo_points.csv"),
                                                   ), index=False
                                               )
            fiona.supported_drivers['KML'] = 'rw'
            kml_file = Path(metadata_folder).joinpath(f"./{m.mission_name}_photo_points.KML")
            if kml_file.is_file():
                ## For a strange reason the library can't overwrite the kml file
                kml_file.unlink()
            m.to_dataframe().to_file(str(kml_file), driver='KML')

            images_taken += len(m.to_dataframe())
            list_images_taken.append({
                "mission_name": m.mission_name,
                "images_taken": len(m.to_dataframe())}
            )
            photo_point_list.append(m.to_dataframe())
            buffer_list.append(buffer)
            multipoint_list.append(multipoint)
            line_list.append(lines)
            dissolved_buffer_list.append(dissolved_buffer)

        # merge all the single items to one dataframe but keep the features separate
        gdf_buffer = pd.concat(buffer_list)
        gdf_dissolved_buffer = pd.concat(dissolved_buffer_list)
        gdf_multipoint = pd.concat(multipoint_list)
        gdf_lines = pd.concat(line_list)
        gdf_photo_point = pd.concat(photo_point_list)

        gdf_buffer.to_file(metadata_folder.joinpath(f"summary_buffer.geojson"), driver="GeoJSON")
        gdf_buffer.to_csv(str(Path(metadata_folder).joinpath(f"summary_buffer.csv")), index=False)


        gdf_multipoint.to_file(metadata_folder.joinpath(f"summary_multipoint.geojson"), driver="GeoJSON")
        gdf_multipoint.to_csv(str(Path(metadata_folder).joinpath(f"summary_multipoint.csv")), index=False)


        gdf_lines.to_file(metadata_folder.joinpath(f"summary_lines.geojson"), driver="GeoJSON")
        gdf_lines.to_csv(str(Path(metadata_folder).joinpath(f"summary_lines.csv")), index=False)


        gdf_dissolved_buffer.to_file(metadata_folder.joinpath(f"summary_dissolved_buffer.geojson"), driver="GeoJSON")
        gdf_dissolved_buffer.to_csv(str(Path(metadata_folder).joinpath(f"summary_dissolved_buffer.csv")), index=False)

        gdf_photo_point.to_file(metadata_folder.joinpath(f"summary_photopoints.geojson"), driver="GeoJSON")
        gdf_photo_point.to_csv(str(Path(metadata_folder).joinpath(f"summary_photo_points.csv")), index=False)

        return {
            "buffer": gdf_buffer,
            "multipoint": gdf_multipoint,
            "lines": gdf_lines,
            "dissolved_buffer": gdf_dissolved_buffer,
            "images_taken": images_taken
                }
def find_mission_folders(base_path) -> [Path]:
    """
    find Mission Paths in nested expedition folders.

    For instance, it is "2021 Dec/06.12.21/MBN06"
    which relates to <Expedition Name>/<Date>/<Mission Name>
    @param base_path: of an Expedition. Which is usually the island name
    @return: [PosixPath]
    """
    mission_paths = []

    p = Path(base_path).glob('*')
    ## find the subfolder which store the dates
    mission_dates = [x for x in p if not x.is_file()]
    if len(mission_dates) == 0:
        error_message = f"no dates found in expedition folder: {base_path}. Structure inside should be: ./<Date>/<Mission_Name>"
        logger.error(error_message)
        # raise ValueError(error_message)

    ## iterate over each date and find the missions in there
    for mission_date in mission_dates:
        current_date_missions = mission_date.glob('*')
        mission_paths = mission_paths + [x for x in current_date_missions if not x.is_file()]
        # TODO if it is a date
    if len(mission_paths) == 0:
        raise ValueError(
            f"no missions found in expedition folder: {base_path}. Structure should be: BASE_PATH/ExpeditionName/Date/<MissionName>")

    return mission_paths


def find_mission_folders_Jan_2023(base_path):
    """
    find Images in nested expedition folders.

    For instance, it is "Fernandina/FENA01_06122021"
    which relates to <Expedition Name>/<Date>/<Mission Name>
    @param base_path:
    @return: [PosixPath]
    """
    missions = []

    p = Path(base_path).glob('*')
    ## find the subfolder which store the dates
    missions_path_dates = [x for x in p if not x.is_file()]
    missions_path_dates = [x for x in missions_path_dates if not x.name == "metadata"]
    if len(missions_path_dates) == 0:
        error_message = f"no dates found in expedition folder: {base_path}. Structure inside should be: ./<Date>/<Mission_Name>"
        logger.error(error_message)
        # raise ValueError(error_message)

    if len(missions_path_dates) == 0:
        raise ValueError(
            f"no missions found in expedition folder: {base_path}. Structure should be: BASE_PATH/ExpeditionName/<MissionName>_<Date>")
    logger.info(f"found {len(missions_path_dates)} folders")
    return missions_path_dates


def find_mission_folders_Dec2021(base_path):
    """
    The Folder structure in 2021 has changed. At every flight date multiple drones where flying, so it is:
    For instance it is "2021 Dec/Drone/06.12.21/Albatross/MBN06"
    which relates to <Expedition Name>/<Placeholder>/<Date>/<Drone Name>/<Mission Name>

    @param base_path:
    @return:
    """
    missions = []

    mission_dates = Path(base_path).glob('*')
    ## find the subfolder which store the dates
    mission_dates = [x for x in mission_dates if not x.is_file()]

    for mission_date in mission_dates:
        # get the drone name
        current_date_drones = mission_date.glob('*')
        for drone in current_date_drones:
            current_date_and_drone_missions = drone.glob('*')
            missions = missions + [x for x in current_date_and_drone_missions if not x.is_file()]

    return missions


if __name__ == '__main__':
    mission_paths = Expedition.get_missions_from_expedition_folder("/home/christian/Downloads/expeditions/Floreana")

    mission_paths
