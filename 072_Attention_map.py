import torch
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from timm.data import resolve_data_config, create_transform

class ViTAttentionCAM:
    def __init__(self, model_name="vit_base_patch16_224"):
        """
        Initializes a Vision Transformer (ViT) model for extracting attention-based CAMs.
        """
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()

        # Get preprocessing pipeline
        self.config = resolve_data_config({}, model=self.model)
        self.preprocess = create_transform(**self.config)

    def get_attention_map(self, image_path):
        """
        Computes the attention-based activation map for a given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Smoothed attention map.
        """
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0)  # Add batch dimension

        # Forward pass: Extract features and attention weights
        with torch.no_grad():
            outputs = self.model.forward_features(img_tensor)

            # Extract attention weights from the last transformer block
            last_block = self.model.blocks[-1]  # Get last transformer block
            attention_weights = last_block.attn.get_attention_weights()  # FIXED: Extract actual attention

        # Convert attention tensor to numpy
        attention_weights = attention_weights.cpu().numpy()

        # Extract CLS token attention (first token)
        cls_attention = attention_weights[:, 0, 1:].mean(axis=0)  # Mean across heads

        # Reshape into patch grid
        num_patches = int(cls_attention.shape[0] ** 0.5)  # Assumes square patches
        attention_map = cls_attention.reshape(num_patches, num_patches)

        # Resize to match the original image size
        attention_map = cv2.resize(
            attention_map, (img.size[0], img.size[1]), interpolation=cv2.INTER_CUBIC
        )

        # Normalize for visualization
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        return attention_map

    def overlay_heatmap(self, image_path, attention_map, alpha=0.5):
        """
        Overlays the attention map on the original image.

        Args:
            image_path (str): Path to the input image.
            attention_map (np.ndarray): Computed attention map.
            alpha (float): Blending factor for overlay.

        Returns:
            np.ndarray: Image with overlay.
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert attention map to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

        # Blend heatmap with image
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return overlay

# Example usage
if __name__ == "__main__":
    model_name = "vit_base_patch16_224"
    cam_extractor = ViTAttentionCAM(model_name)

    image_path = "/Users/christian/Downloads/WAID-main/WAID/images/train/0d702197-87d5-4d74-8ca5-fc2b55479dba_jpg.rf.af49eaf1fedf85b51f2972668da07b0d.jpg"
    attention_map = cam_extractor.get_attention_map(image_path)
    overlay_img = cam_extractor.overlay_heatmap(image_path, attention_map)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_img)
    plt.axis("off")
    plt.title("ViT Attention CAM")
    plt.show()
