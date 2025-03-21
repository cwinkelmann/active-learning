"""
fine tune a classifier using


"""

import matplotlib.pyplot as plt
import numpy as np
import time
import timm
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from timm.data import create_transform
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union

from active_learning.pipelines.data_splitting import split_dataset


class CustomImageDataset(Dataset):
    """Custom dataset for image classification with timm models."""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (integers)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label


class ModelFineTuner:
    """Class for fine-tuning timm models for image classification."""

    def __init__(
            self,
            model_name: str,
            num_classes: int,
            pretrained: bool = True,
            checkpoint_path: Optional[str] = None,
            freeze_base: bool = False,
            device: Optional[str] = None
    ):
        """
        Initialize the fine-tuner.

        Args:
            model_name: Name of the timm model
            num_classes: Number of classes for classification
            pretrained: Whether to use pretrained weights
            checkpoint_path: Path to checkpoint to resume from
            freeze_base: Whether to freeze base model weights
            device: Device to use (will auto-detect if None)
        """
        # Set device
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create model
        self.model_name = model_name
        self.num_classes = num_classes

        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model = torch.load(checkpoint_path)
        else:
            print(f"Creating model: {model_name} with {num_classes} classes")
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )

        # Move model to device
        self.model = self.model.to(self.device)

        # Freeze base if needed
        if freeze_base:
            self._freeze_base_model()

        # Get recommended transforms
        self.train_transform = create_transform(
            input_size=self.model.default_cfg['input_size'][1:],
            is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1'
        )

        self.val_transform = create_transform(
            input_size=self.model.default_cfg['input_size'][1:],
            is_training=False
        )

        # History for tracking metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }

    def _freeze_base_model(self):
        """Freeze all layers except the final classifier."""
        for name, param in self.model.named_parameters():
            if 'head' not in name and 'fc' not in name:  # Don't freeze head/classifier
                param.requires_grad = False

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})")

    def unfreeze(self):
        """Unfreeze all model layers."""
        for param in self.model.parameters():
            param.requires_grad = True

        print("All model parameters unfrozen.")

    @staticmethod
    def load_from_directory(
            directory: Union[str, Path],
            valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
            class_to_idx: Optional[Dict[str, int]] = None
    ) -> Tuple[List[Path], List[int], Dict[str, int]]:
        """
        Load image paths and labels from a directory structure where each subfolder is a class.

        Args:
            directory: Root directory containing class subfolders
            valid_extensions: Tuple of valid image file extensions
            class_to_idx: Optional mapping of class names to indices
                          (if None, will be created automatically)

        Returns:
            Tuple of (image_paths, labels, class_to_idx_mapping)
        """
        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        # Get all subdirectories (classes)
        classes = sorted([d.name for d in directory.iterdir()
                          if d.is_dir() and not d.name.startswith('.')])

        if not classes:
            raise ValueError(f"No class directories found in {directory}")

        print(f"Found {len(classes)} classes: {', '.join(classes)}")

        # Create class to index mapping if not provided
        if class_to_idx is None:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # Collect all image paths and labels
        image_paths = []
        labels = []

        for cls_name, idx in class_to_idx.items():
            cls_dir = directory / cls_name
            if not cls_dir.exists() or not cls_dir.is_dir():
                print(f"Warning: Class directory {cls_dir} not found, skipping")
                continue

            # Find all image files in this class directory
            cls_images = [p for p in cls_dir.glob('**/*')
                          if p.is_file() and p.suffix.lower() in valid_extensions
                          and not p.name.startswith('.')]

            if not cls_images:
                print(f"Warning: No images found for class {cls_name}")
                continue

            # Add paths and labels
            image_paths.extend(cls_images)
            labels.extend([idx] * len(cls_images))

            print(f"  - {cls_name}: {len(cls_images)} images")

        if not image_paths:
            raise ValueError(f"No valid images found in {directory}")

        return image_paths, labels, class_to_idx

    def prepare_data(
            self,
            train_dir: Optional[Union[str, Path]] = None,
            val_dir: Optional[Union[str, Path]] = None,
            train_image_paths: Optional[List[Union[str, Path]]] = None,
            train_labels: Optional[List[int]] = None,
            val_image_paths: Optional[List[Union[str, Path]]] = None,
            val_labels: Optional[List[int]] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            val_split: float = 0.0,
            random_seed: int = 42
    ):
        """
        Prepare datasets and dataloaders. Can load from directory structure or explicit lists.

        Args:
            train_dir: Directory containing training images in class subfolders
            val_dir: Directory containing validation images in class subfolders
            train_image_paths: List of paths to training images (alternative to train_dir)
            train_labels: List of training labels (required if train_image_paths is provided)
            val_image_paths: List of paths to validation images (alternative to val_dir)
            val_labels: List of validation labels (required if val_image_paths is provided)
            batch_size: Batch size
            num_workers: Number of workers for data loading
            val_split: Fraction of training data to use for validation if val_dir and val_image_paths are None
            random_seed: Random seed for train/val split
        """
        # Dictionary to store class mapping
        self.class_to_idx = None

        # Load from directory structure if provided
        if train_dir is not None:
            print(f"Loading training data from directory: {train_dir}")
            train_paths, train_lbls, self.class_to_idx = self.load_from_directory(train_dir)

            # Convert Path objects to strings for compatibility
            train_image_paths = [str(p) for p in train_paths]
            train_labels = train_lbls

            print(f"Loaded {len(train_image_paths)} training images")

        # Validate that we have training data
        if train_image_paths is None or train_labels is None:
            raise ValueError("You must provide either train_dir or both train_image_paths and train_labels")

        if len(train_image_paths) != len(train_labels):
            raise ValueError(
                f"Number of training images ({len(train_image_paths)}) and labels ({len(train_labels)}) must match")

        # Load validation data if directory provided
        if val_dir is not None:
            print(f"Loading validation data from directory: {val_dir}")
            val_paths, val_lbls, _ = self.load_from_directory(val_dir, class_to_idx=self.class_to_idx)

            # Convert Path objects to strings for compatibility
            val_image_paths = [str(p) for p in val_paths]
            val_labels = val_lbls

            print(f"Loaded {len(val_image_paths)} validation images")

        # Create train/val split if needed
        if (val_dir is None and val_image_paths is None and val_split > 0):
            print(f"Creating validation split ({val_split:.1%} of training data)")

            # Create train/val indices
            indices = list(range(len(train_image_paths)))

            # Shuffle indices
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            # Calculate split
            val_size = int(len(indices) * val_split)
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]

            # Create validation data
            val_image_paths = [train_image_paths[i] for i in val_idx]
            val_labels = [train_labels[i] for i in val_idx]

            # Update training data
            train_image_paths = [train_image_paths[i] for i in train_idx]
            train_labels = [train_labels[i] for i in train_idx]

            print(f"Split dataset: {len(train_image_paths)} training, {len(val_image_paths)} validation images")

        # Create training dataset
        self.train_dataset = CustomImageDataset(
            train_image_paths, train_labels, transform=self.train_transform
        )

        # Create training dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        # Create validation dataset and dataloader if data is available
        if val_image_paths is not None and val_labels is not None:
            if len(val_image_paths) != len(val_labels):
                raise ValueError(
                    f"Number of validation images ({len(val_image_paths)}) and labels ({len(val_labels)}) must match")

            self.val_dataset = CustomImageDataset(
                val_image_paths, val_labels, transform=self.val_transform
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
            print("No validation data provided")

    def train(
            self,
            epochs: int = 10,
            optimizer_name: str = 'adam',
            learning_rate: float = 3e-4,
            weight_decay: float = 1e-4,
            lr_scheduler: str = 'cosine',
            mixed_precision: bool = True,
            save_path: Optional[Union[str, Path]] = None,
            early_stopping_patience: int = 0,
            clip_grad_norm: float = 0
    ):
        """
        Train the model.

        Args:
            epochs: Number of epochs
            optimizer_name: Name of optimizer
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            lr_scheduler: Type of learning rate scheduler
            mixed_precision: Whether to use mixed precision
            save_path: Path to save the best model
            early_stopping_patience: Early stopping patience (0 to disable)
            clip_grad_norm: Max norm for gradient clipping (0 to disable)

        Returns:
            Best validation accuracy
        """
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create optimizer
        optimizer = create_optimizer_v2(
            self.model,
            opt=optimizer_name,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Create scheduler
        if lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
        else:
            scheduler = None

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # For mixed precision
        scaler = None

        # Training loop
        best_val_acc = 0
        no_improve_epochs = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Training phase
            self.model.train()
            train_loss = 0

            for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
                images, targets = images.to(self.device), targets.to(self.device)

                # Zero gradients
                optimizer.zero_grad()
                # Standard precision training
                outputs = self.model(images)
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping if enabled
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)

                # Update weights
                optimizer.step()

                # Update running loss
                train_loss += loss.item()

            # Calculate average training loss
            avg_train_loss = train_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_train_loss)

            # Validation phase if we have validation data
            val_loss = 0
            val_acc = 0

            if self.val_loader:
                self.model.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for images, targets in tqdm(self.val_loader, desc="Validation"):
                        images, targets = images.to(self.device), targets.to(self.device)

                        # Forward pass
                        outputs = self.model(images)
                        loss = criterion(outputs, targets)

                        # Calculate accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                        # Update running loss
                        val_loss += loss.item()

                # Calculate metrics
                avg_val_loss = val_loss / len(self.val_loader)
                val_acc = correct / total

                # Update history
                self.history['val_loss'].append(avg_val_loss)
                self.history['val_accuracy'].append(val_acc)

                # Update LR scheduler if using ReduceLROnPlateau
                if lr_scheduler == 'reduce_on_plateau':
                    scheduler.step(avg_val_loss)

            # Update LR scheduler if not ReduceLROnPlateau
            if scheduler and lr_scheduler != 'reduce_on_plateau':
                scheduler.step()

            # Store current learning rate
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            # Calculate epoch time
            epoch_time = time.time() - start_time

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.1f}s - Train Loss: {avg_train_loss:.4f}", end="")
            if self.val_loader:
                print(f" - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                print("")

            # Save best model if we have validation data
            if self.val_loader and val_acc > best_val_acc:
                best_val_acc = val_acc

                if save_path:
                    torch.save(self.model, save_path)
                    print(f"Model saved to {save_path}")

                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Early stopping
            if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model if we saved it
        if save_path and save_path.exists() and self.val_loader:
            self.model = torch.load(save_path)
            print(f"Loaded best model from {save_path}")

        return best_val_acc if self.val_loader else None

    def evaluate(self, test_loader=None, detailed=True):
        """
        Evaluate the model.

        Args:
            test_loader: DataLoader for test data (uses val_loader if None)
            detailed: Whether to show detailed metrics

        Returns:
            Dictionary of metrics
        """
        # Use validation loader if test loader not provided
        loader = test_loader or self.val_loader

        if not loader:
            print("No evaluation data available")
            return None

        self.model.eval()

        # Initialize variables
        all_preds = []
        all_targets = []

        # Evaluation loop
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Evaluating"):
                images, targets = images.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                # Store predictions and targets
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        acc = np.mean(np.array(all_preds) == np.array(all_targets))

        # Print results
        print(f"Accuracy: {acc:.4f}")

        if detailed:
            # Generate classification report
            print("\nClassification Report:")
            print(classification_report(all_targets, all_preds))

            # Generate confusion matrix
            cm = confusion_matrix(all_targets, all_preds)

            # Plot confusion matrix if number of classes is reasonable
            if self.num_classes <= 20:
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.tight_layout()
                plt.show()

        return {
            'accuracy': acc,
            'predictions': all_preds,
            'targets': all_targets
        }

    def plot_training_history(self):
        """Plot training and validation metrics."""
        if not self.history['train_loss']:
            print("No training history available")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot losses
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot learning rate and accuracy if available
        ax2.plot(self.history['learning_rates'], label='Learning Rate', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate', color='green')
        ax2.set_title('Learning Rate and Validation Accuracy')
        ax2.tick_params(axis='y', labelcolor='green')

        # Add second y-axis for accuracy
        if self.history['val_accuracy']:
            ax3 = ax2.twinx()
            ax3.plot(self.history['val_accuracy'], label='Validation Accuracy', color='red')
            ax3.set_ylabel('Accuracy', color='red')
            ax3.tick_params(axis='y', labelcolor='red')

            # Create custom legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        else:
            ax2.legend()

        ax2.grid(True)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def predict(self, image_path, return_probs=False):
        """
        Predict the class of a single image.

        Args:
            image_path: Path to the image
            return_probs: Whether to return all class probabilities

        Returns:
            Predicted class (and probabilities if return_probs=True)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.val_transform(img).unsqueeze(0).to(self.device)

        # Set model to eval mode
        self.model.eval()

        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)

            if return_probs:
                probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
                pred_class = np.argmax(probs)
                return pred_class, probs
            else:
                _, pred_class = torch.max(outputs, 1)
                return pred_class.item()

    def predict_batch(self, image_paths, return_probs=False):
        """
        Predict classes for a batch of images.

        Args:
            image_paths: List of paths to images
            return_probs: Whether to return all class probabilities

        Returns:
            List of predicted classes (and probabilities if return_probs=True)
        """
        # Create a dataset and dataloader for the images
        dummy_labels = [0] * len(image_paths)  # Dummy labels
        dataset = CustomImageDataset(image_paths, dummy_labels, transform=self.val_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        # Set model to eval mode
        self.model.eval()

        # Make predictions
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Predicting"):
                images = images.to(self.device)
                outputs = self.model(images)

                if return_probs:
                    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    all_probs.extend(probs)
                else:
                    _, preds = torch.max(outputs, 1)
                    preds = preds.cpu().numpy()

                all_preds.extend(preds)

        if return_probs:
            return all_preds, all_probs
        else:
            return all_preds

    def save_model(self, path):
        """
        Save the model to the specified path.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, path)
        print(f"Model saved to {path}")

    def export_to_onnx(self, path, input_shape=None):
        """
        Export the model to ONNX format.

        Args:
            path: Path to save the ONNX model
            input_shape: Input shape (if None, use model's default)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get input shape if not provided
        if input_shape is None:
            input_shape = (1, 3, *self.model.default_cfg['input_size'][1:])

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"Model exported to ONNX format at {path}")

    def get_class_activation_map(self, image_path, save_path=None):
        """
        Generate a class activation map (CAM) for an image.

        Args:
            image_path: Path to the image
            save_path: Path to save the visualization

        Returns:
            Tuple of (original image, CAM image)
        """
        # Check if the model supports CAM
        if not hasattr(self.model, 'forward_features'):
            print("Model doesn't support CAM generation")
            return None

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.val_transform(img).unsqueeze(0).to(self.device)

        # Set model to eval mode
        self.model.eval()

        # Get original image as numpy array for plotting
        orig_img = np.array(img)

        # Register hook to get feature maps
        feature_maps = None

        def hook_fn(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()

        # Try to register hook to the last convolutional layer
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn)
                break

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, pred_class = torch.max(outputs, 1)
            pred_class = pred_class.item()

        # Get class-specific weights from the classifier
        if feature_maps is None:
            print("Could not generate CAM - feature maps were not captured")
            return None

        # Get weights from the final layer
        classifier = None
        for name, module in self.model.named_modules():
            if "head" in name or "fc" in name or "classifier" in name:
                if isinstance(module, nn.Linear):
                    classifier = module
                    break

        if classifier is None:
            print("Could not find classifier layer")
            return None

        # Get weights for predicted class
        class_weights = classifier.weight[pred_class].detach().cpu().numpy()

        # Reshape weights if needed
        if len(class_weights.shape) == 1 and len(feature_maps.shape) == 4:
            # Reshape to match the number of feature maps
            class_weights = class_weights.reshape(-1, 1, 1)

        # Calculate CAM
        cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)

        # Multiply each activation map with corresponding weight
        for i, w in enumerate(class_weights):
            # Handle case where weights might be for each position
            if isinstance(w, np.ndarray):
                cam += feature_maps[0, i].cpu().numpy() * w.mean()
            else:
                cam += feature_maps[0, i].cpu().numpy() * w

        # Apply ReLU to focus on positive contributions
        cam = np.maximum(cam, 0)

        # Normalize CAM
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)

        # Resize CAM to match input image size
        from scipy.ndimage import zoom
        zoom_factor = (orig_img.shape[0] / cam.shape[0], orig_img.shape[1] / cam.shape[1])
        cam = zoom(cam, zoom_factor)

        # Create heatmap
        import matplotlib.cm as cm
        heatmap = cm.jet(cam)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)

        # Overlay heatmap on original image with transparency
        overlay = orig_img * 0.7 + heatmap * 0.3
        overlay = overlay.astype(np.uint8)

        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(orig_img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(overlay)
        ax2.set_title(f'Class Activation Map (Class: {pred_class})')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()

        return orig_img, overlay


if __name__ == "__main__":
    # Initialize fine-tuner
    fine_tuner = ModelFineTuner(
        model_name="timm/resnet34.a1_in1k",  # Use any timm model
        num_classes=2,  # Number of your classes
        freeze_base=True,  # Start with frozen base for transfer learning
        device=torch.device("mps")
    )

    # iguanas
    train_dir = Path("/Users/christian/data/training_data/2025_02_22_HIT/01_segment_pretraining/segments_12/train/classification")
    val_dir = Path("/Users/christian/data/training_data/2025_02_22_HIT/01_segment_pretraining/segments_12/val/classification")
    model_path = "final_model_iguanas.pth"
    model_stage1_path = "final_model_iguanas_stage1.pth"

    # iguanas
    train_dir = Path("/Users/christian/data/training_data/2025_02_22_HIT/03_all_other/train_floreana_big/classification")
    val_dir = Path("/Users/christian/data/training_data/2025_02_22_HIT/03_all_other/val/classification")
    model_path = "final_model_iguanas_points.pth"
    model_stage1_path = "final_model_iguanas_points_stage1.pth"

    # # cats and dogs
    # train_dir = Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224_train_val/train")
    # val_dir = Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224_train_val/val/")
    # model_path = "final_model_cats_docs.pth"
    # model_stage1_path = "final_model_cats_docs.pth"

    # Or specify separate validation directory
    fine_tuner.prepare_data(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=10
    )


    # Train with frozen base first
    fine_tuner.train(
        epochs=5,
        learning_rate=3e-4,
        save_path=model_stage1_path
    )

    # Unfreeze layers for fine-tuning
    fine_tuner.unfreeze()

    # Continue training with lower learning rate
    fine_tuner.train(
        epochs=5,
        learning_rate=1e-4,
        save_path=model_path,
        early_stopping_patience=3
    )

    # Evaluate the model
    results = fine_tuner.evaluate(detailed=True)

    # Plot training history
    fine_tuner.plot_training_history()

    # Save the final model
    fine_tuner.save_model(model_path)

    # Generate class activation map for an image
    fine_tuner.get_class_activation_map("/Users/christian/data/training_data/2025_02_22_HIT/01_segment_pretraining/segments_12/train/classification/iguana/Floreana_02.02.21_FMO01___DJI_0942_x1344_y2240.jpg")

