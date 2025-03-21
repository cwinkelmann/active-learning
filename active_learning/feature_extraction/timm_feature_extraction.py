from typing import List, Optional, Union

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


class FeatureExtractor:
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        if model_name is not None:
            self.model = timm.create_model(
                model_name, pretrained=True, num_classes=0, global_pool="avg"
            ).to(self.device)
        elif model_path is not None:
            self.model = torch.load(model_path).to(self.device)
        else:
            raise ValueError("Must provide either a model name or a model path.")

        self.model.eval()

        self.preprocess = create_transform(**resolve_data_config({}, model=self.model))

    def extract_features(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extract features from a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Normalized feature vector or None if extraction fails
        """
        # Convert to Path if string
        image_path = Path(image_path) if isinstance(image_path, str) else image_path

        try:
            input_image = Image.open(image_path).convert("RGB")
            input_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor).to("cpu")

            feature_vector = output.squeeze().numpy()
            return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_from_directory(self, root_dir: Union[str, Path], save_path: Union[str, Path]) -> pd.DataFrame:
        """Extract features from all images in a directory structure organized by class.

        Args:
            root_dir: Path to the directory containing class subdirectories
            save_path: Path where to save the resulting CSV file

        Returns:
            DataFrame containing extracted features and class labels
        """
        # Convert to Path if string
        root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        save_path = Path(save_path) if isinstance(save_path, str) else save_path

        embeddings = []
        filenames = []
        class_labels = []

        # Traverse class folders
        for class_path in root_dir.iterdir():
            if class_path.is_dir():
                class_name = class_path.name
                for img_path in tqdm(list(class_path.iterdir()), desc=f"Processing {class_name}"):
                    if img_path.is_file() and img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                        features = self.extract_features(img_path)
                        if features is not None:
                            embeddings.append(features)
                            filenames.append(img_path.name)
                            class_labels.append(class_name)  # Assign label based on folder name

        # Create DataFrame and save as CSV
        df = pd.DataFrame(embeddings, index=filenames)
        df["class"] = class_labels  # Add class label column

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index_label="filename")
        print(f"Saved embeddings to {save_path}")

        return df

    def extract_from_image_list(self,
                                image_paths: List[Union[str, Path]],
                                class_labels: Optional[List[str]] = None,
                                save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Extract features from a list of images.

        Args:
            image_paths: List of paths to image files
            class_labels: Optional list of class labels corresponding to each image
            save_path: Optional path to save the resulting CSV file

        Returns:
            DataFrame containing extracted features and optionally class labels
        """
        if class_labels is not None and len(class_labels) != len(image_paths):
            raise ValueError(
                f"Number of class labels ({len(class_labels)}) doesn't match number of images ({len(image_paths)})")

        embeddings = []
        filenames = []
        valid_indices = []

        # Process each image
        for i, img_path in enumerate(tqdm(image_paths, desc="Extracting features")):
            # Convert to Path if string
            img_path = Path(img_path) if isinstance(img_path, str) else img_path

            features = self.extract_features(img_path)
            if features is not None:
                embeddings.append(features)
                filenames.append(img_path.name)
                valid_indices.append(i)

        # Create DataFrame
        df = pd.DataFrame(embeddings, index=filenames)

        # Add class labels if provided
        if class_labels is not None:
            df["class"] = [class_labels[i] for i in valid_indices]

        # Save if path provided
        if save_path is not None:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index_label="filename")
            print(f"Saved embeddings to {save_path}")

        return df

