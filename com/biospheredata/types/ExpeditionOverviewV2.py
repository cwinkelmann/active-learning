"""
Database to find images

one database wraps around multiple Expeditions, which store missions, wich store images


"""

import glob
import json
import tempfile
from geopandas import GeoDataFrame
from pathlib import Path
from shutil import rmtree
import zipfile

import pandas as pd
from loguru import logger

from com.biospheredata.types.Expedition import Expedition
from com.biospheredata.types.HastyAnnotation import HastyAnnotation
from com.biospheredata.types.Mission import Mission


class ExpeditionOverviewV2(object):
    """
    load all expeditions missions you can find.
    """

    base_path = "./"
    metadata_file = "expedition_overview.json"
    expeditions: [Expedition]

    def __init__(self,
                 expeditions: [Expedition],
                 expedition_overview_path: Path):

        self.expeditions = expeditions
        self.expedition_overview_path = expedition_overview_path
        self.expedition_overview_path.mkdir(exist_ok=True, parents=True)
    def open(self, folder: Path):
        """
        open a folder with the prediction zip files
        @param folder:
        @return:
        """
        eO = ExpeditionOverviewV2()

        eO.hasty_prediction_files = glob.glob(f'{folder}/*/hasty_annotation.json')

        for x in eO.hasty_prediction_files:
            eO.add_hasty_annotation(HastyAnnotation.from_file(x))

        return eO

    def get_missions(self) -> [Mission]:
        """

        :return: [Mission]
        """
        missions = []

        # TODO fix the type annotations here, the list screws it up
        for e in self.expeditions:
            for m in e.missions:
                missions.append(m)

        return missions

    def to_dataframe(self) -> pd.DataFrame:
        """
        generate a dateframe with the path, annotation json, metadata, like counts
        @return:
        """

        df_expeditions = []

        for expedition in self.expeditions:
            df_expeditions.append(expedition.to_dataframe())

        single_image_metadata = []

        for mission in self.get_missions():
            assert isinstance(mission, Mission)
            # hasty_annotations.append(HastyAnnotation.from_file(mission.get_hasty_annotation_file_mosaic()))
            single_image_metadata = single_image_metadata + mission.to_dataframe().to_dict(orient="records")

        merged_image_metadata_df = pd.DataFrame.from_records(single_image_metadata)

        return merged_image_metadata_df

    def expedition_level_statistics(self, projected_CRS="EPSG:32715"):
        """
        """
        statistics_result = []

        for expedition in self.expeditions:
            mission_summary = expedition.mission_summary(metadata_folder=expedition.expedition_path.joinpath("metadata"),
                                                 projected_CRS=projected_CRS,
                                                         type="buffer")

            logger.info(f"summarise {expedition.expedition_name}")
            statistics_result.append(
                {
                    "expedition_name": expedition.expedition_name,
                    "duration": int(mission_summary["buffer"]["duration"].round().sum()),
                    "distance": int(mission_summary["buffer"]["distance"].round().sum()),
                    "area": int(mission_summary["dissolved_buffer"].area.sum()),
                    "flights": int(mission_summary["buffer"]["mission_name"].count()),
                    "images_taken": mission_summary["images_taken"],
                 }
            )

        logger.info(statistics_result)

        dictionary = pd.DataFrame(statistics_result).to_dict(orient="records")


        with open(self.expedition_overview_path.joinpath("expedition_overview.json"), 'w') as f:
            json.dump(dictionary, f)
            logger.info(f"Wrote json expedition_overview.json to {self.expedition_overview_path.resolve()}")

        with open(self.expedition_overview_path.joinpath("expedition_overview.csv"), 'w') as f:
            pd.DataFrame(statistics_result).to_csv(f)
            logger.info(f"Wrote json expedition_overview.csv to {self.expedition_overview_path}")

        with open(self.expedition_overview_path.joinpath("expedition_overview.tex"), 'w') as f:
            pd.DataFrame(statistics_result).to_latex(f, index=False)
            logger.info(f"Wrote json expedition_overview.tex to {self.expedition_overview_path}")

        return pd.DataFrame(statistics_result)


    def persist(self):
        """
        write the dataframe to disk
        @return:
        """
        self.to_dataframe().to_csv(self.expedition_overview_path.joinpath("all_images_of_expedition.csv"), index=False, header=True)
        return self.expedition_overview_path


if __name__ == '__main__':
    path = "/home/christian/mount/hetzner_webdav_od/predictions"
    path = Path("/home/christian/mount/hetzner_webdav_od/predictions_exp_1_2")
    # path = "/media/christian/2TB/ai-core/data/iguanas_from_above/hasty"

    eO = ExpeditionOverviewV2.open(path)

    good_annotations = eO.find_good_missions(label="iguana", label_threshold=50, image_threshold=5)

    print(good_annotations)

    for good_annotation in good_annotations:
        print(good_annotation.project_name)
        print(good_annotation.images)
