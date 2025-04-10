import numpy as np
import matplotlib.pyplot as plt
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, Optional, Tuple, List
import cv2
from pathlib import Path
import random

import pandas as pd
import torch
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from typing import Union, Tuple, List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from PIL import Image

class TimmModel(object):

    """
    A base class for timm models.
    """

    def __init__(self,             model_name: str = 'resnet34.a1_in1k',
            pretrained: bool = True,
            checkpoint_path: Optional[Union[str, Path]] = None,
            device: Optional[str] = None,
            num_classes: int = 0,
                 global_pool: str = 'avg'):
        """
        Initialize the TimmModel with a specified model name and whether to use a pretrained version.

        Args:
            model_name: Name of the timm model (e.g., 'resnet50', 'resnet101')
            pretrained: Whether to use a pretrained model
        """

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else
                                       "cpu")
        else:
            self.device = torch.device(device)

        # Create model based on whether checkpoint is provided
        if checkpoint_path is None:
            # Initialize with pretrained weights from timm
            print(f"Loading pretrained model: {model_name}")
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                global_pool=global_pool
            ).to(self.device)
        else:
            # Load from checkpoint
            checkpoint_path = Path(checkpoint_path) if not isinstance(checkpoint_path, Path) else checkpoint_path
            print(f"Loading checkpoint from: {checkpoint_path}")

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            try:
                # First, create the base model with the correct architecture
                # Note: For fine-tuned models, we create the model with desired num_classes
                # rather than the original architecture's class count
                base_model = timm.create_model(
                    model_name,
                    pretrained=False,  # Don't load pretrained weights, we'll load from checkpoint
                    num_classes=num_classes,  # Set this to your fine-tuned class count
                    global_pool=global_pool
                ).to(self.device)

                # Load the checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        # Standard format with state_dict key
                        state_dict = checkpoint["state_dict"]
                    elif "model" in checkpoint and isinstance(checkpoint["model"], torch.nn.Module):
                        # Training checkpoint with model object
                        state_dict = checkpoint["model"].state_dict()
                    else:
                        # Assume it's already a state_dict
                        state_dict = checkpoint
                elif isinstance(checkpoint, torch.nn.Module):
                    # It's a full model
                    state_dict = checkpoint.state_dict()
                else:
                    raise ValueError(f"Unsupported checkpoint format")

                # Try to load the state dict - use strict=False to allow for classifier differences
                missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)

                # Log any issues with loading
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {missing_keys}")
                    # Check specifically for classifier keys
                    classifier_keys = [k for k in missing_keys if 'classifier' in k or 'fc' in k or 'head' in k]
                    if classifier_keys:
                        print(f"Note: Missing classifier keys: {classifier_keys}")
                        print("This may be expected if your model was fine-tuned with a different number of classes.")

                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

                self.model = base_model

            except Exception as e:
                raise ValueError(f"Failed to load checkpoint: {e}")

        # Set model to evaluation mode
        self.model.eval()

        # Create preprocessing transform
        self.data_config = resolve_data_config({}, model=self.model)
        self.transforms = create_transform(**self.data_config, is_training=False)

        # Verify model output size
        dummy_input = torch.randn(1, 3, *self.data_config.get('input_size')[1:]).to(self.device)
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
        actual_num_classes = dummy_output.shape[1]

        print(f"Model loaded successfully. Input size: {self.data_config.get('input_size')}")
        print(f"Output shape: {dummy_output.shape} (number of classes: {actual_num_classes})")

        if num_classes > 0 and actual_num_classes != num_classes:
            print(f"WARNING: Requested {num_classes} classes but model outputs {actual_num_classes} classes!")
            print("This may indicate issues with the checkpoint loading or model configuration.")

class GradCAMVisualizer(TimmModel):
    """
    A class to visualize Grad-CAM activations for timm ResNet models.
    """

    def __init__(self, model_name: str = 'resnet34.a1_in1k',
            pretrained: bool = True,
            checkpoint_path: Optional[Union[str, Path]] = None,
            device: Optional[str] = None,
            num_classes: int = 0,
                 global_pool: str = 'avg'):
        """
        Initialize the GradCAM visualizer with a ResNet model from timm.

        Args:
            model_name: Name of the timm model (e.g., 'resnet50', 'resnet101')
            pretrained: Whether to use a pretrained model
        """
        # Load the model
        super().__init__(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path,
                         device=device, num_classes=num_classes, global_pool=global_pool)

        # Get the target layer - for ResNet, this is typically the last convolutional layer
        if 'resnet' in model_name:
            # For ResNet, the last conv layer is in layer4
            self.target_layer = self.model.layer4[-1]
        else:
            raise ValueError(f"Model {model_name} is not supported or is not a ResNet variant.")

        # Register hooks to get activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _register_hooks(self):
        """Register forward and backward hooks to get activations and gradients."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Preprocess the image for the model."""
        image = Image.open(image_path).convert('RGB')

        input_tensor = self.transform(image).to(self.device)
        return input_tensor.unsqueeze(0)  # Add batch dimension

    def generate_cam(self, image_path: Union[str, Path], target_class: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray, int, float]:
        """
        Generate the Grad-CAM for an image.

        Args:
            image_path: Path to the input image
            target_class: Target class for visualization. If None, uses the predicted class

        Returns:
            Tuple containing:
                - Original image (numpy array)
                - CAM heatmap (numpy array)
                - Predicted class index
                - Prediction confidence
        """
        # Preprocess the image
        input_tensor = self._preprocess_image(image_path)

        # Forward pass
        output = self.model(input_tensor)

        # Get the prediction
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

        # Use predicted class if target_class is not specified
        if target_class is None:
            target_class = predicted_class.item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target = torch.zeros_like(output)
        target[0, target_class] = 1
        output.backward(gradient=target)

        # Get the mean gradients
        gradients = self.gradients.detach().cpu().numpy()[0]

        # Calculate weights based on global average pooling
        weights = np.mean(gradients, axis=(1, 2))

        # Get activations
        activations = self.activations.detach().cpu().numpy()[0]

        # Compute weighted activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU to focus on features that have a positive influence
        cam = np.maximum(cam, 0)

        # Normalize the CAM
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        # Load the original image for visualization
        original_image = cv2.imread(str(image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        return original_image, cam, predicted_class.item(), confidence.item()

    def visualize(self,
                  image_path: Union[str, Path],
                  target_class: Optional[int] = None,
                  alpha: float = 0.5,
                  colormap: str = 'jet',
                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize the Grad-CAM for an image.

        Args:
            image_path: Path to the input image
            target_class: Target class for visualization. If None, uses the predicted class
            alpha: Transparency level for the heatmap overlay
            colormap: Matplotlib colormap for the heatmap
            save_path: Optional path to save the visualization

        Returns:
            matplotlib Figure object
        """
        # Generate the CAM
        original_image, cam, predicted_class, confidence = self.generate_cam(image_path, target_class)

        # Get class names if ImageNet model
        class_name = f"Class {predicted_class}"
        if hasattr(self.model, 'default_cfg') and 'label' in self.model.default_cfg:
            idx_to_class = self.model.default_cfg.get('label')
            if idx_to_class and predicted_class < len(idx_to_class):
                class_name = idx_to_class[predicted_class]

        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

        # Create heatmap
        heatmap = plt.cm.get_cmap(colormap)(cam_resized)[:, :, :3]  # Drop alpha channel

        # Create overlay
        overlay = original_image.astype(float) / 255.0
        overlay = (1 - alpha) * overlay + alpha * heatmap

        # Clip overlay to valid range
        overlay = np.clip(overlay, 0, 1)

        # Create figure with original image and overlay
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # CAM overlay
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM: {class_name} ({confidence:.2f})")
        axes[1].axis('off')

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        return fig

    def visualize_batch(self,
                        image_paths: List[Union[str, Path]],
                        grid_size: Optional[Tuple[int, int]] = None,
                        target_classes: Optional[List[int]] = None,
                        alpha: float = 0.5,
                        colormap: str = 'jet',
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Visualize Grad-CAM for multiple images in a grid.

        Args:
            image_paths: List of paths to input images
            grid_size: Optional tuple of (rows, cols). If None, determined automatically
            target_classes: Optional list of target classes. If None, uses predicted classes
            alpha: Transparency level for the heatmap overlay
            colormap: Matplotlib colormap for the heatmap
            save_path: Optional path to save the visualization

        Returns:
            matplotlib Figure object
        """
        n_images = len(image_paths)

        # Determine grid size if not provided
        if grid_size is None:
            cols = min(4, int(np.ceil(np.sqrt(n_images))))
            rows = int(np.ceil(n_images / cols))
            grid_size = (rows, cols)
        else:
            rows, cols = grid_size

        # Create figure
        fig = plt.figure(figsize=(6 * cols, 4 * rows))

        # Process each image
        for i, image_path in enumerate(image_paths):
            if i >= rows * cols:
                break

            # Get target class if provided
            target_class = None
            if target_classes and i < len(target_classes):
                target_class = target_classes[i]

            # Generate the CAM
            original_image, cam, predicted_class, confidence = self.generate_cam(image_path, target_class)

            # Get class name
            class_name = f"Class {predicted_class}"
            if hasattr(self.model, 'default_cfg') and 'label' in self.model.default_cfg:
                idx_to_class = self.model.default_cfg.get('label')
                if idx_to_class and predicted_class < len(idx_to_class):
                    class_name = idx_to_class[predicted_class]

            # Resize CAM to match image size
            cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

            # Create heatmap
            heatmap = plt.cm.get_cmap(colormap)(cam_resized)[:, :, :3]

            # Create overlay
            overlay = original_image.astype(float) / 255.0
            overlay = (1 - alpha) * overlay + alpha * heatmap
            overlay = np.clip(overlay, 0, 1)

            # Calculate position in grid
            r, c = divmod(i, cols)

            # Original image
            ax1 = plt.subplot2grid((rows, cols * 2), (r, c * 2))
            ax1.imshow(original_image)
            ax1.set_title("Original")
            ax1.axis('off')

            # CAM overlay
            ax2 = plt.subplot2grid((rows, cols * 2), (r, c * 2 + 1))
            ax2.imshow(overlay)
            ax2.set_title(f"{class_name} ({confidence:.2f})")
            ax2.axis('off')

        plt.suptitle("Grad-CAM Visualizations", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Save if requested
        if save_path:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved batch visualization to {save_path}")

        return fig


if __name__ == '__main__':

    model_path = "/Users/christian/PycharmProjects/hnee/pytorch-image-models/output/iguanas_empty_resnet34.a1_in1k/model_best.pth.tar"

    cam_vis = GradCAMVisualizer(model_name='resnet34',  # Base architecture
        checkpoint_path=model_path,  # Your trained weights
        num_classes=2,  # Number of classes in your model
    )

    empty_image_path = "/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_112/val/empty/Flo_FLPC04_22012021_empty_117322_9864145_117322_9864146.jpg"
    iguana_image_path = "/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_112/val/iguana/Scris_SRL12_10012021_centered_207975.4858857841_9898061.550476288.jpg"
    # Visualize CAM for a single image
    fig = cam_vis.visualize(
        # image_path=empty_image_path,
        image_path=iguana_image_path,
        save_path='cam_visualization.png'
    )
    plt.show()

