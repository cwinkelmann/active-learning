import torch
import timm
from timm.data import create_transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import os
import time
import json


class ModelInference:
    """Class for performing inference with timm models."""

    def __init__(
            self,
            model_source: Union[str, Path],
            num_classes: Optional[int] = None,
            class_map: Optional[Dict[str, int]] = None,
            pretrained: bool = False,
            device: Optional[str] = None,
            transforms: Optional[callable] = None
    ):
        """
        Initialize the inference engine.

        Args:
            model_source: Either a path to saved model (.pth file) or a timm model name
            num_classes: Number of classes when creating a new model (not needed for loaded models)
            class_map: Optional dictionary mapping class names to indices
            pretrained: Whether to use pretrained weights (only for new models)
            device: Device to use ('cuda', 'cpu', 'mps' or None for auto-detect)
            transforms: Optional custom transforms (if None, will use model's default)
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        model_source = str(model_source)  # Convert to string if Path
        self.model_name = model_source

        # Determine if model_source is a path or model name
        is_path = os.path.exists(model_source) or model_source.endswith('.pth')

        # Load or create model
        if is_path:
            # Load saved model
            print(f"Loading model from {model_source}")
            self.model = torch.load(model_source, map_location=self.device)

            # Try to get model name
            if hasattr(self.model, 'model_name'):
                self.model_name = self.model.model_name

            # Try to load class mapping
            self.class_to_idx = class_map
            class_map_path = Path(model_source).parent / f"{Path(model_source).stem}_classes.json"
            if self.class_to_idx is None and class_map_path.exists():
                try:
                    with open(class_map_path, 'r') as f:
                        self.class_to_idx = json.load(f)
                    print(f"Loaded class mapping from {class_map_path}")
                except:
                    print(f"Failed to load class mapping from {class_map_path}")
        else:
            # Create new model using timm
            if num_classes is None:
                raise ValueError("num_classes must be provided when creating a new model")

            print(f"Creating new model: {model_source} with {num_classes} classes")
            self.model = timm.create_model(
                model_source,
                pretrained=pretrained,
                num_classes=num_classes
            )

            # Use provided class mapping
            self.class_to_idx = class_map

        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)

        # Create reverse mapping
        if self.class_to_idx:
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            print(f"Using class mapping with {len(self.class_to_idx)} classes")
        else:
            self.idx_to_class = None
            print("No class mapping provided - results will use numeric indices")

        # Get or create transforms
        if transforms is not None:
            self.transforms = transforms
        else:
            # Use default transforms from model
            self.transforms = create_transform(
                input_size=self.model.default_cfg['input_size'][1:],
                is_training=False
            )

    @classmethod
    def from_timm(
            cls,
            model_name: str,
            num_classes: int,
            class_names: Optional[List[str]] = None,
            pretrained: bool = False,
            device: Optional[str] = None
    ) -> 'ModelInference':
        """
        Create an inference engine with a fresh timm model.

        Args:
            model_name: Name of the timm model to use
            num_classes: Number of output classes
            class_names: Optional list of class names (in order of indices)
            pretrained: Whether to use pretrained weights
            device: Compute device to use

        Returns:
            ModelInference instance
        """
        # Create class mapping if class names provided
        class_map = None
        if class_names:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Number of class names ({len(class_names)}) doesn't match num_classes ({num_classes})")
            class_map = {name: idx for idx, name in enumerate(class_names)}

        return cls(
            model_source=model_name,
            num_classes=num_classes,
            class_map=class_map,
            pretrained=pretrained,
            device=device
        )

    @classmethod
    def from_saved_model(
            cls,
            model_path: Union[str, Path],
            class_map: Optional[Dict[str, int]] = None,
            device: Optional[str] = None
    ) -> 'ModelInference':
        """
        Create an inference engine from a saved model.

        Args:
            model_path: Path to the saved model file
            class_map: Optional class mapping dictionary
            device: Compute device to use

        Returns:
            ModelInference instance
        """
        return cls(
            model_source=model_path,
            class_map=class_map,
            device=device
        )

    @classmethod
    def from_fine_tuner(
            cls,
            fine_tuner,
            save_path: Optional[Union[str, Path]] = None
    ) -> 'ModelInference':
        """
        Create an inference engine from a ModelFineTuner instance.

        Args:
            fine_tuner: A ModelFineTuner instance
            save_path: Optional path to save the model

        Returns:
            ModelInference instance
        """
        # Save model if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save(fine_tuner.model, save_path)

            # Save class mapping if available
            if hasattr(fine_tuner, 'class_to_idx') and fine_tuner.class_to_idx:
                class_map_path = save_path.parent / f"{save_path.stem}_classes.json"
                with open(class_map_path, 'w') as f:
                    json.dump(fine_tuner.class_to_idx, f)
        else:
            # Create temporary file
            import tempfile
            temp_dir = tempfile.gettempdir()
            save_path = Path(temp_dir) / f"temp_model_{int(time.time())}.pth"
            torch.save(fine_tuner.model, save_path)

        # Create inference engine
        class_map = fine_tuner.class_to_idx if hasattr(fine_tuner, 'class_to_idx') else None
        return cls.from_saved_model(save_path, class_map=class_map)

    def predict_single(
            self,
            image_path: Union[str, Path],
            return_probs: bool = False
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Predict the class of a single image.

        Args:
            image_path: Path to the image
            return_probs: Whether to return all class probabilities

        Returns:
            Predicted class name or tuple of (class_name, probabilities_dict)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
            pred_idx = int(np.argmax(probs))

        # Get class name if available
        if self.idx_to_class is not None:
            pred_class = self.idx_to_class[pred_idx]
        else:
            pred_class = str(pred_idx)

        if return_probs:
            # Create dictionary of class probabilities
            if self.idx_to_class is not None:
                prob_dict = {self.idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
            else:
                prob_dict = {str(i): float(probs[i]) for i in range(len(probs))}

            return pred_class, prob_dict
        else:
            return pred_class

    def predict_batch(
            self,
            image_paths: List[Union[str, Path]],
            batch_size: int = 32,
            return_probs: bool = False,
            progress_bar: bool = True
    ) -> Union[List[str], Tuple[List[str], List[Dict[str, float]]]]:
        """
        Predict classes for a batch of images.

        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing
            return_probs: Whether to return all class probabilities
            progress_bar: Whether to show progress bar

        Returns:
            List of predicted class names or tuple of (class_names, probabilities_dicts)
        """
        all_preds = []
        all_probs = []

        # Process in batches
        num_batches = (len(image_paths) + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if progress_bar:
            iterator = tqdm(iterator, desc="Predicting")

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]

            # Load and preprocess batch
            batch_tensors = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transforms(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    # Add a placeholder tensor
                    batch_tensors.append(torch.zeros(3, *self.model.default_cfg['input_size'][1:]))

            # Stack tensors and move to device
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(self.device)

                # Make predictions
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    batch_probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    batch_pred_idx = np.argmax(batch_probs, axis=1)

                # Convert indices to class names if available
                if self.idx_to_class is not None:
                    batch_preds = [self.idx_to_class[int(idx)] for idx in batch_pred_idx]
                else:
                    batch_preds = [str(int(idx)) for idx in batch_pred_idx]

                all_preds.extend(batch_preds)

                if return_probs:
                    # Create dictionaries of class probabilities
                    batch_prob_dicts = []
                    for probs in batch_probs:
                        if self.idx_to_class is not None:
                            prob_dict = {self.idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
                        else:
                            prob_dict = {str(i): float(probs[i]) for i in range(len(probs))}
                        batch_prob_dicts.append(prob_dict)

                    all_probs.extend(batch_prob_dicts)

        if return_probs:
            return all_preds, all_probs
        else:
            return all_preds

    def predict_directory(
            self,
            directory: Union[str, Path],
            recursive: bool = True,
            valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
            batch_size: int = 32,
            return_probs: bool = False,
            save_results: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Predict classes for all images in a directory.

        Args:
            directory: Directory containing images
            recursive: Whether to search subdirectories
            valid_extensions: Tuple of valid image file extensions
            batch_size: Batch size for processing
            return_probs: Whether to include probability scores in results
            save_results: Optional path to save results CSV

        Returns:
            DataFrame with prediction results
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        # Find all images
        image_paths = []

        if recursive:
            for ext in valid_extensions:
                image_paths.extend(directory.glob(f'**/*{ext}'))
                image_paths.extend(directory.glob(f'**/*{ext.upper()}'))
        else:
            for ext in valid_extensions:
                image_paths.extend(directory.glob(f'*{ext}'))
                image_paths.extend(directory.glob(f'*{ext.upper()}'))

        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError(f"No valid images found in {directory}")

        print(f"Found {len(image_paths)} images in {directory}")

        # Make predictions
        if return_probs:
            predictions, probabilities = self.predict_batch(
                image_paths, batch_size=batch_size, return_probs=True
            )

            # Create DataFrame with results
            results = pd.DataFrame({
                'image_path': [str(p.relative_to(directory) if p.is_relative_to(directory) else p) for p in
                               image_paths],
                'predicted_class': predictions
            })

            # Add top probability
            top_probs = [max(p.values()) for p in probabilities]
            results['probability'] = top_probs

            # Add all class probabilities if not too many
            if len(probabilities[0]) <= 10:
                for class_name in probabilities[0].keys():
                    results[f'prob_{class_name}'] = [p[class_name] for p in probabilities]

        else:
            predictions = self.predict_batch(
                image_paths, batch_size=batch_size, return_probs=False
            )

            # Create DataFrame with results
            results = pd.DataFrame({
                'image_path': [str(p.relative_to(directory) if p.is_relative_to(directory) else p) for p in
                               image_paths],
                'predicted_class': predictions
            })

        # Save results if requested
        if save_results:
            save_path = Path(save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")

        return results

    def save_model(
            self,
            path: Union[str, Path],
            save_class_map: bool = True
    ):
        """
        Save the model and optionally the class mapping.

        Args:
            path: Path to save the model
            save_class_map: Whether to save the class mapping
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model, path)
        print(f"Model saved to {path}")

        # Save class mapping if available and requested
        if save_class_map and self.class_to_idx:
            class_map_path = path.parent / f"{path.stem}_classes.json"
            with open(class_map_path, 'w') as f:
                json.dump(self.class_to_idx, f)
            print(f"Class mapping saved to {class_map_path}")


# Example usage - directly with timm

"""
# Method 1: Create inference engine with a new timm model
inference = ModelInference.from_timm(
    model_name="resnet18",
    num_classes=2,
    class_names=["cat", "dog"],
    pretrained=True
)

# Method 2: Create with a saved model
inference = ModelInference.from_saved_model('model.pth')

# Method 3: Create from a fine-tuner
inference = ModelInference.from_fine_tuner(fine_tuner, save_path='model.pth')

# Method 4: Direct initialization (old way)
inference = ModelInference('model.pth')  # Load saved model
# OR
inference = ModelInference('resnet18', num_classes=2)  # Create new timm model

# Predict with any of the methods
prediction = inference.predict_single('test_image.jpg')
"""