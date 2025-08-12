import tempfile
import unittest
import random
from loguru import logger
from pathlib import Path

import pandas as pd
import sklearn

import numpy as np
from sklearn.model_selection import train_test_split, KFold

from com.biospheredata.types.exceptions.NotEnoughImagesException import NotEnoughImagesException


def generate_k_folds(
        df_train: pd.DataFrame,
        # amount_train_size = None,
        # min_train_size=None,
        # test_size = 0.1,
        folds=5
):
    """
    from images select a good split

    # TODO implement this https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

    @param df_with_object: absolute path names to the images and their labels
    @param df_background:
    @return:
    :param image_paths:
    :param max_train_size:
    :param test_size:
    :param folds:
    """
    assert isinstance(df_train, pd.DataFrame)

    # label_paths = [image_path.with_suffix('.txt') for image_path in image_paths]

    # TODO it is not guaranteed those labels exists, especially in background images
    # df_with_object = pd.DataFrame({
    #     "filenames": image_paths,
    #     # "labels": label_paths
    # })
    #
    # df_test, df_train = get_test_set(amount_train_size, df_with_object, test_size)

    k_folds = []
    # Now we have the dataset we want to train on. But we need to cross validate it now.
    # See https://scikit-learn.org/stable/modules/cross_validation.html
    kf = KFold(n_splits=folds)
    for df_train_k, df_valid_k in kf.split(df_train):
        # print("%s %s" % (train, test))

        #if len(df_train_k) < amount_train_size:
        #    raise ValueError(f"train_size of {train_size} is too small after cutting of test set. df_with_object lenght: {len(df_with_object)}")

        logger.info(f"df_train_k: {len(df_train_k)}")
        logger.info(f"df_valid_k: {len(df_valid_k)}")

        k_folds.append(
            {
                "train": df_train.take(list(df_train_k)),
                "valid": df_train.take(list(df_valid_k)),
            }
        )

    return k_folds


def get_test_set(amount_train_size, df_with_object, test_size):
    """
    retrieve a test from the overall dataset
    :param amount_train_size:
    :param df_with_object:
    :param test_size:
    :return:
    """
    absolute_test_size = int(len(df_with_object) * test_size)
    logger.info(f"The absolute test set size will be {absolute_test_size}")
    train_size = len(df_with_object) - absolute_test_size
    if amount_train_size is not None:
        train_size = min(train_size, amount_train_size)

        if train_size < amount_train_size:
            raise NotEnoughImagesException(
                f"train_size of {train_size} is too small after cutting of test set. df_with_object length: {len(df_with_object)}")
    logger.info(f"The absolute train+valid set size will be {train_size}")
    # add the objects
    df_train, df_test = train_test_split(df_with_object, shuffle=True, train_size=train_size, test_size=test_size)
    return df_test, df_train