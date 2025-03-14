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

class FeatureExtractor:
    def __init__(self, model_name: str):

        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, global_pool="avg"
        ).to("cuda" if torch.cuda.is_available() else "mps")
        self.model.eval()
        self.preprocess = create_transform(**resolve_data_config({}, model=self.model))

    def extract_features(self, image_path: str) -> np.ndarray:
        import torch
        from PIL import Image
        from sklearn.preprocessing import normalize

        try:
            input_image = Image.open(image_path).convert("RGB")
            input_tensor = self.preprocess(input_image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "mps")

            with torch.no_grad():
                output = self.model(input_tensor).to("cpu")

            feature_vector = output.squeeze().numpy()
            return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_from_directory(self, root_dir: str, save_path: str):
        embeddings = []
        filenames = []
        class_labels = []

        # Traverse class folders
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        features = self.extract_features(img_path)
                        if features is not None:
                            embeddings.append(features)
                            filenames.append(img_name)
                            class_labels.append(class_name)  # Assign label based on folder name

        # Save as CSV with class labels
        df = pd.DataFrame(embeddings, index=filenames)
        df["class"] = class_labels  # Add class label column
        df.to_csv(save_path, index_label="filename")
        print(f"Saved embeddings to {save_path}")

# Example usage
if __name__ == "__main__":
    model_name = "vit_base_patch16_224"
    extractor = FeatureExtractor(model_name)

    image_root_dir = "/Users/christian/Downloads/WAID-main/WAID/images/train_sample"  # This should contain subfolders (ClassA, ClassB, ...)
    save_path = "embeddings_with_labels.csv"

    extractor.extract_from_directory(image_root_dir, save_path)
