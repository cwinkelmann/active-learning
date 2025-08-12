import glob
import json
import tempfile
from pathlib import Path
from shutil import rmtree
import zipfile

import pandas as pd
from loguru import logger

from com.biospheredata.types.HastyAnnotation import HastyAnnotation


class ExpeditionOverview(object):
    """
    load all missions you can find
    """
    missions = []

    hasty_annotations = []
    hasty_prediction_files = []

    def __init__(self):
        self.missions = []
        self.hasty_annotations = []
        self.hasty_prediction_files = []

    @staticmethod
    def open(folder: Path):
        """
        open a folder with the prediction zip files
        @param folder:
        @return:
        """
        eO = ExpeditionOverview()

        eO.hasty_prediction_files = glob.glob(f'{folder}/*/hasty_annotation.json')

        for x in eO.hasty_prediction_files:
            eO.add_hasty_annotation(HastyAnnotation.from_file(x))

        return eO

    @staticmethod
    def openFromZip(path):
        eO = ExpeditionOverview()

        try:
            tempdir = tempfile.mkdtemp(prefix="predicitons_zip_files")
            zip_files = glob.glob(f'{path}/*.zip')
            for zip_file in zip_files:
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        logger.info(f"opening: {zip_file} to retrieve hasty_annotation.json")
                        hasty_annotation = zip_ref.read('hasty_annotation.json')
                        hasty_annotation = json.loads(hasty_annotation)
                        eO.add_hasty_annotation(HastyAnnotation.from_dict(hasty_annotation))
                except AttributeError as e:
                    logger.error(e)
            return eO

        except Exception as e:
            logger.error(e)
            logger.error(e.__traceback__)
        finally:
            rmtree(tempdir)


    def to_dataframe(self):
        """
        generate a dateframe with the path, annotation json, metadata, like counts
        @return:
        """
        annotation_dictionary = {
            "hasty_prediction_files": self.hasty_prediction_files
        }

        annotation_df = pd.DataFrame.from_records(annotation_dictionary)

        statistics_for_all = []
        for annotation in self.hasty_annotations:
            statistics_for_all.append(annotation.get_statistic())

        statistics_df = pd.DataFrame.from_records(statistics_for_all)

        statistics_df = pd.concat([annotation_df, statistics_df], axis=1)

        statistics_df = statistics_df.fillna(0)
        # convert the statistic columns to integer
        cols = [i for i in statistics_df.columns if i not in ["hasty_prediction_files"]]
        for col in cols:
            statistics_df[col] = statistics_df[col].astype(int)

        return statistics_df



    def add_hasty_annotation(self, hA: HastyAnnotation):
        self.hasty_annotations.append(hA)


    def find_good_missions(self, label, label_threshold, image_threshold = None):
        """
        find missions, which contain at least these amounts
        @param label_threshold: amount of individual labels per image slice
        @param image_threshold: amount of
        @return:
        """

        good_missions = []

        for annotation in self.hasty_annotations:
            images = annotation.get_images_with_more_than(label, threshold=label_threshold)

            if image_threshold is not None and len(images) > image_threshold:
                good_missions.append(annotation)

            if image_threshold is None:
                good_missions.append(annotation)

        return good_missions




if __name__ == '__main__':
    path = "/home/christian/mount/hetzner_webdav_od/predictions"
    path = Path("/home/christian/mount/hetzner_webdav_od/predictions_exp_1_2")
    # path = "/media/christian/2TB/ai-core/data/iguanas_from_above/hasty"

    eO = ExpeditionOverview.open(path)

    good_annotations = eO.find_good_missions(label="iguana", label_threshold=50, image_threshold=5)

    print(good_annotations)

    for good_annotation in good_annotations:
        print(good_annotation.project_name)
        print(good_annotation.images)