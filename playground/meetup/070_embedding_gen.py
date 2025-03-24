from pathlib import Path

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import timm
from timm.data import resolve_data_config, create_transform

import torch
from PIL import Image
from sklearn.preprocessing import normalize

from active_learning.feature_extraction.embedding_clustering import get_tsn_embeddings
from active_learning.feature_extraction.timm_feature_extraction import FeatureExtractor

# Example usage
if __name__ == "__main__":
    model_name = "vit_base_patch16_224"
    model_name = "timm/resnet34.a1_in1k"
    model_name = "dla34.in1k"

    # extractor = FeatureExtractor(model_name)
    model_path = "final_model_iguanas_2025_03_23.pth"

    model_path = "/Users/christian/PycharmProjects/hnee/active_learning/active_learning/pipelines/final_model_cats_2025_03_23.pth"
    extractor = FeatureExtractor(model_path=model_path, model_name=model_name)

    # cats and dogs
    train_dir = Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224_train_val_small/train/")
    val_dir = Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224_train_val_small/val/")

    # image_root_dir = "/Users/christian/data/WAID-main/WAID/images/train_sample"  # This should contain subfolders (ClassA, ClassB, ...)
    image_root_dir = Path("/Users/christian/data/training_data/2025_02_22_HIT/01_segment_pretraining/segments_12/val/classification")  # This should contain subfolders (ClassA, ClassB, ...)
    image_root_dir = Path("/Users/christian/data/training_data/2025_02_22_HIT/03_all_other/val/classification")
    image_root_dir = train_dir

    save_path = "embeddings_with_labels.csv"

    images_list = [i for i in image_root_dir.rglob("*.jpg") if not str(i).startswith(".")]
    class_labels = [x.parent.stem for x in images_list]

    df_features = extractor.extract_from_image_list(images_list, class_labels=class_labels, save_path=save_path)

    df_features

    df_tsne_embeddings = get_tsn_embeddings(save_path)

    df_tsne_embeddings

    # from active_learning.feature_extraction.embedding_clustering import display_misclassified_grid

    # display_misclassified_grid(df_tsne_embeddings, image_dir=image_root_dir, cols=5, figsize=(15, 3), true_class_column="class")