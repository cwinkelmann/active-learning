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

class LogitExtractor:
    def __init__(
            self,
            model_name: str = 'resnet34.a1_in1k',
            pretrained: bool = True,
            checkpoint_path: Optional[Union[str, Path]] = None,
            device: Optional[str] = None,
            num_classes: int = 0,
            global_pool: str = "avg"
    ):
        """
        Initialize the LogitExtractor with a timm model.

        Args:
            model_name: Name of the timm model to use
            pretrained: Whether to use pretrained weights (ignored if checkpoint_path is provided)
            checkpoint_path: Path to a custom checkpoint file
            device: Device to use (None for auto-detection)
            num_classes: Number of output classes (0 for default)
            global_pool: Global pooling type ("avg", "max", etc.)
        """
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else
                                       "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

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

    def extract_logits(
            self,
            image: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        """
        Extract raw logits from an image.

        Args:
            image: Input image (PIL Image or path to image)

        Returns:
            torch.Tensor: Raw logits from the model
        """
        # Load the image if a path is provided
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')

        # Preprocess and run inference
        with torch.no_grad():
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            logits = self.model(input_tensor)

        return logits

    def extract_probabilities(
            self,
            image: Union[str, Path, Image.Image],
            top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract softmax probabilities and top class indices from an image.

        Args:
            image: Input image (PIL Image or path to image)
            top_k: Number of top probabilities to return

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Top-k probabilities and corresponding class indices
        """
        # Get logits first
        logits = self.extract_logits(image)

        # Calculate softmax probabilities and get top-k results
        probs = logits.softmax(dim=1) * 100  # Convert to percentage
        top_probs, top_indices = torch.topk(probs, k=top_k)

        return top_probs, top_indices

    def predict_image(
            self,
            image: Union[str, Path, Image.Image],
            class_map: Union[str, Path, List[str]] = None,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict classes for an image and return human-readable results.

        Args:
            image: Input image (PIL Image or path to image)
            class_map: Path to class map file or list of class names
            top_k: Number of top predictions to return

        Returns:
            List[dict]: List of dictionaries with class names and probabilities
        """
        # Get class names if provided
        class_names = None
        if class_map is not None:
            if isinstance(class_map, (str, Path)):
                with open(class_map, 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
            else:
                class_names = class_map

        # Extract probabilities
        probs, indices = self.extract_logits(image, return_probs=True, top_k=top_k)

        # Convert to list of dictionaries
        results = []
        for prob, idx in zip(probs[0], indices[0]):
            idx_int = idx.item()
            class_name = class_names[idx_int] if class_names and idx_int < len(class_names) else f"class_{idx_int}"
            results.append({
                "class": class_name,
                "probability": prob.item()
            })

        return results

    def batch_extract_logits(
            self,
            images: List[Union[str, Path, Image.Image]]
    ) -> List[torch.Tensor]:
        """
        Extract logits from a batch of images.

        Args:
            images: List of input images (PIL Images or paths to images)

        Returns:
            List[torch.Tensor]: List of raw logits for each image
        """
        return [self.extract_logits(image) for image in images]

    def batch_extract_probabilities(
            self,
            images: List[Union[str, Path, Image.Image]],
            top_k: int = 5
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract probabilities from a batch of images.

        Args:
            images: List of input images (PIL Images or paths to images)
            top_k: Number of top probabilities to return

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of (probabilities, indices) tuples
        """
        return [self.extract_probabilities(image, top_k) for image in images]

    def extract_from_image_list(
            self,
            image_paths: List[Union[str, Path]],
            class_labels: Optional[List[str]] = None,
            class_mapping: Optional[List[str]] = None,
            save_path: Optional[Union[str, Path]] = None,
            use_tqdm: bool = True
    ) -> pd.DataFrame:
        """
        Extract logits from a list of images and optionally save to a CSV file.

        Args:
            image_paths: List of paths to image files
            class_labels: Optional list of class labels corresponding to each image
            class_mapping: Optional list of class names mapping to logit indices
                           (e.g. ["negative", "positive"] for binary classification)
            save_path: Optional path to save the resulting CSV file
            use_tqdm: Whether to show a progress bar (requires tqdm package)

        Returns:
            DataFrame containing extracted logits and optionally class labels
        """
        if class_labels is not None and len(class_labels) != len(image_paths):
            raise ValueError(
                f"Number of class labels ({len(class_labels)}) doesn't match number of images ({len(image_paths)})")

        # Import tqdm conditionally to avoid dependency issues
        if use_tqdm:
            try:
                from tqdm import tqdm as progress_bar
            except ImportError:
                print("Warning: tqdm package not found. Disabling progress bar.")
                use_tqdm = False

        logits_list = []
        filenames = []
        valid_indices = []

        # Prepare iterator with or without progress bar
        iterator = progress_bar(image_paths, desc="Extracting logits") if use_tqdm else image_paths

        # Process each image
        for i, img_path in enumerate(iterator):
            # Convert to Path if string
            img_path = Path(img_path) if isinstance(img_path, str) else img_path

            try:
                # Extract raw logits
                raw_logits = self.extract_logits(img_path)

                # Convert to numpy array and flatten
                logits_np = raw_logits.cpu().numpy().flatten()

                # If this is a binary classifier, we can represent it with a single value
                # (the second logit or the difference between logits)
                if logits_np.shape[0] == 2:
                    # Option 1: Use both logits directly
                    logits_values = {f"logit_{i}": float(val) for i, val in enumerate(logits_np)}

                    # Option 2: Also add a derived confidence score (difference between positive and negative)
                    logits_values["confidence"] = float(logits_np[1] - logits_np[0])

                    # Option 3: Add softmax probabilities
                    probs = torch.nn.functional.softmax(raw_logits, dim=1).cpu().numpy().flatten()
                    logits_values["prob_0"] = float(probs[0])
                    logits_values["prob_1"] = float(probs[1])

                    # Add predicted class index (0 or 1)
                    predicted_idx = int(logits_np[1] > logits_np[0])
                    logits_values["predicted_idx"] = predicted_idx

                    # Add human-readable predicted class using class mapping if provided
                    if class_mapping is not None:
                        if len(class_mapping) >= 2:  # Make sure we have enough classes in the mapping
                            logits_values["predicted_class"] = class_mapping[predicted_idx]
                        else:
                            print(
                                f"Warning: class_mapping has {len(class_mapping)} classes but model outputs {logits_np.shape[0]} logits")

                else:
                    # For multi-class, use all logits
                    logits_values = {f"logit_{i}": float(val) for i, val in enumerate(logits_np)}

                    # Add softmax probabilities
                    probs = torch.nn.functional.softmax(raw_logits, dim=1).cpu().numpy().flatten()
                    for i, prob in enumerate(probs):
                        logits_values[f"prob_{i}"] = float(prob)

                    # Add predicted class index
                    predicted_idx = int(np.argmax(logits_np))
                    logits_values["predicted_idx"] = predicted_idx

                    # Add human-readable predicted class using class mapping if provided
                    if class_mapping is not None:
                        if len(class_mapping) >= len(logits_np):  # Make sure we have enough classes in the mapping
                            logits_values["predicted_class"] = class_mapping[predicted_idx]
                        else:
                            print(
                                f"Warning: class_mapping has {len(class_mapping)} classes but model outputs {logits_np.shape[0]} logits")

                # Add to our lists
                logits_list.append(logits_values)
                filenames.append(img_path.name)
                valid_indices.append(i)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Create DataFrame
        df = pd.DataFrame(logits_list, index=filenames)

        # Add class labels if provided
        if class_labels is not None:
            df["true_class"] = [class_labels[i] for i in valid_indices]

        # Save if path provided
        if save_path is not None:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index_label="filename")
            print(f"Saved logits to {save_path}")

        return df

    def visualize_binary_logits(
            self,
            dataframe: pd.DataFrame,
            true_label_column: str = "true_class",
            predicted_label_column: str = "predicted_class",
            confidence_column: str = "confidence",
            logit0_column: str = "logit_0",
            logit1_column: str = "logit_1",
            title: str = "Binary Classification Logits",
            save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Visualize binary logits from a DataFrame as a 2D scatter plot with error highlighting.

        Args:
            dataframe: DataFrame containing logits and class labels
            true_label_column: Name of the column containing true class labels
            predicted_label_column: Name of the column containing predicted labels
            confidence_column: Name of the column containing confidence values
            logit0_column: Name of the column containing the first logit
            logit1_column: Name of the column containing the second logit
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib.figure.Figure: The created figure object
        """

        # Check if required columns exist
        if logit0_column not in dataframe.columns or logit1_column not in dataframe.columns:
            raise ValueError(f"Logit columns '{logit0_column}' and/or '{logit1_column}' not found in DataFrame")

        # Check if label columns exist
        has_true_labels = true_label_column in dataframe.columns
        has_pred_labels = predicted_label_column in dataframe.columns
        has_confidence = confidence_column in dataframe.columns

        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # If predicted labels don't exist but we have true labels, infer predictions from logits
        if not has_pred_labels and has_true_labels:
            df[predicted_label_column] = df[logit1_column] > df[logit0_column]

            # Try to convert to the same type as true_labels
            unique_labels = df[true_label_column].unique()
            if len(unique_labels) == 2:
                # Map boolean to the actual labels
                label_map = {False: unique_labels[0], True: unique_labels[1]}
                df[predicted_label_column] = df[predicted_label_column].map(label_map)

            has_pred_labels = True

        # Calculate errors if we have both true and predicted labels
        highlight_errors = has_true_labels and has_pred_labels
        if highlight_errors:
            df['is_error'] = df[true_label_column] != df[predicted_label_column]
            num_errors = df['is_error'].sum()
            error_rate = (num_errors / len(df)) * 100

        # Create the figure
        fig = plt.figure(figsize=(12, 9))

        # Handle point sizes based on confidence
        if has_confidence:
            # Calculate sizes based on confidence
            confidence_values = df[confidence_column].abs()
            min_confidence = confidence_values.min()
            max_confidence = confidence_values.max()

            # Normalize confidence to a reasonable size range (50-300)
            if max_confidence > min_confidence:
                sizes = 50 + 250 * (confidence_values - min_confidence) / (max_confidence - min_confidence)
            else:
                sizes = 100  # Default if all values are the same
        else:
            sizes = 100  # Default size

        # Plot the data
        if has_true_labels:
            # Get unique labels and assign colors
            labels = df[true_label_column].unique()
            colors = {label: plt.cm.tab10(i / len(labels)) for i, label in enumerate(labels)}

            # Plot each class
            for label, color in colors.items():
                mask = df[true_label_column] == label

                if highlight_errors:
                    # Correct predictions
                    correct_mask = mask & ~df['is_error']
                    if correct_mask.any():
                        plt.scatter(
                            df.loc[correct_mask, logit0_column],
                            df.loc[correct_mask, logit1_column],
                            c=[color],
                            label=f"{label} (correct)",
                            alpha=0.4,
                            edgecolors='w',
                            s=sizes[correct_mask].values if has_confidence else sizes,
                            marker='o',
                            zorder=2
                        )

                    # Incorrect predictions
                    error_mask = mask & df['is_error']
                    if error_mask.any():
                        plt.scatter(
                            df.loc[error_mask, logit0_column],
                            df.loc[error_mask, logit1_column],
                            c=[color],
                            label=f"{label} (misclassified)",
                            alpha=0.9,
                            edgecolors='red',
                            linewidth=2,
                            s=sizes[error_mask].values if has_confidence else sizes,
                            marker='X',
                            zorder=10
                        )
                else:
                    # Standard plot without error highlighting
                    plt.scatter(
                        df.loc[mask, logit0_column],
                        df.loc[mask, logit1_column],
                        c=[color],
                        label=str(label),
                        alpha=1.0,
                        edgecolors='w',
                        s=sizes[mask].values if has_confidence else sizes
                    )

            plt.legend()
        else:
            # Simple scatter plot without class information
            plt.scatter(
                df[logit0_column],
                df[logit1_column],
                alpha=0.7,
                edgecolors='w',
                s=sizes.values if has_confidence else sizes
            )

        # Add decision boundary (x=y line for binary classifier)
        min_val = min(df[logit0_column].min(), df[logit1_column].min())
        max_val = max(df[logit0_column].max(), df[logit1_column].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Decision Boundary')

        # Add labels and title
        plt.xlabel(logit0_column)
        plt.ylabel(logit1_column)
        plt.title(title)
        plt.grid(alpha=0.3)

        # Make the plot square
        plt.axis('equal')

        # Add annotations if needed
        if highlight_errors:
            plt.figtext(0.5, 0.01, f"Error rate: {error_rate:.2f}% ({num_errors}/{len(df)} misclassified)",
                        ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

        if has_confidence:
            plt.figtext(0.5, 0.03 if highlight_errors else 0.01,
                        "Note: Point size represents confidence level",
                        ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

        # Save if requested
        if save_path:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.tight_layout()

        return fig

    def visualize_confident_errors(
            self,
            dataframe: pd.DataFrame,
            image_dir: Union[str, Path],
            true_label_column: str = "true_class",
            predicted_label_column: str = "predicted_class",
            confidence_column: str = "confidence",
            n_images: int = 16,
            grid_size: Tuple[int, int] = None,
            sort_by_confidence: bool = True,
            title: str = "Most Confident Misclassifications",
            save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Create a grid visualization of the most confident misclassifications.

        Args:
            dataframe: DataFrame containing prediction results with filenames as index
            image_dir: Directory containing the image files
            true_label_column: Name of the column containing true class labels
            predicted_label_column: Name of the column containing predicted labels
            confidence_column: Name of the column containing confidence values
            n_images: Maximum number of images to display
            grid_size: Optional tuple of (rows, cols). If None, determined automatically
            sort_by_confidence: Whether to sort errors by confidence (True) or alphabetically by filename (False)
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            import numpy as np
            from PIL import Image
        except ImportError:
            raise ImportError("Matplotlib and PIL are required. Install with 'pip install matplotlib pillow'")

        # Check if required columns exist
        for col in [true_label_column, predicted_label_column]:
            if col not in dataframe.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        # Convert image_dir to Path
        image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir

        # Get misclassified images
        errors_df = dataframe[dataframe[true_label_column] != dataframe[predicted_label_column]].copy()

        if len(errors_df) == 0:
            print("No misclassifications found!")
            return None

        # Sort by confidence if requested and available
        if sort_by_confidence and confidence_column in errors_df.columns:
            # Use absolute value of confidence to get most confident regardless of direction
            errors_df['abs_confidence'] = errors_df[confidence_column].abs()
            errors_df = errors_df.sort_values('abs_confidence', ascending=False)

        # Limit to n_images
        errors_df = errors_df.head(n_images)

        # Determine grid size if not provided
        if grid_size is None:
            cols = min(4, int(np.ceil(np.sqrt(len(errors_df)))))
            rows = int(np.ceil(len(errors_df) / cols))
            grid_size = (rows, cols)
        else:
            rows, cols = grid_size

        # Create figure
        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        gs = gridspec.GridSpec(rows, cols)

        # Counter for placed images
        count = 0

        # Load and place images
        for idx, row in errors_df.iterrows():
            if count >= rows * cols:
                break

            # Get filename from index
            filename = idx
            true_class = row[true_label_column]
            # Construct image path - try different extensions if needed
            img_path = None

            # If filename already has extension, try direct path
            direct_path = image_dir / true_class / filename
            if direct_path.exists():
                img_path = direct_path


                try:
                    # Read image
                    img = Image.open(img_path)

                    # Calculate grid position
                    r, c = divmod(count, cols)

                    # Create subplot
                    ax = plt.subplot(gs[r, c])

                    # Display image
                    ax.imshow(np.array(img))

                    # Create label
                    true_label = row[true_label_column]
                    pred_label = row[predicted_label_column]
                    conf_text = f" (Conf: {row[confidence_column]:.2f})" if confidence_column in row else ""

                    # Set title and turn off axis
                    ax.set_title(f"True: {true_label}\nPred: {pred_label}{conf_text}", fontsize=10)
                    ax.axis('off')

                    count += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

        # Set overall title
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save if requested
        if save_path:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved grid visualization to {save_path}")

        return fig


# Example usage
if __name__ == "__main__":
    from urllib.request import urlopen


    imgage_path_iguana = Path(
        "/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_112/val/iguana/Gen_GES01to09_04122021_centered_171913.0404174219_33284.56984052529.jpg")
    # Load an image from a URL
    img = Image.open(imgage_path_iguana)

    # # Example 1: Initialize with pretrained model
    # extractor = LogitExtractor(model_name='resnet34.a1_in1k', pretrained=True)
    # # Get top probabilities and class indices
    # logits = extractor.extract_logits(img)
    # top_probs, top_indices = extractor.extract_probabilities(img, top_k=2)
    # print("\nTop 5 predictions:")
    # for prob, idx in zip(top_probs[0], top_indices[0]):
    #     print(f"Class {idx.item()}: {prob.item():.2f}%")

    model_path = "/Users/christian/PycharmProjects/hnee/pytorch-image-models/output/iguanas_empty_resnet34.a1_in1k/model_best.pth.tar"

    # Initialize with a custom checkpoint
    extractor2 = LogitExtractor(
        model_name='resnet34',  # Base architecture
        checkpoint_path=model_path,  # Your trained weights
        num_classes=2,  # Number of classes in your model
    )

    # Use it the same way for inference

    # Get top probabilities and class indices
    logits = extractor2.extract_logits(img)
    top_probs, top_indices = extractor2.extract_probabilities(img, top_k=2)
    print("\nTop 5 predictions:")
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f"Class {idx.item()}: {prob.item():.2f}%")

    # image_root_dir = "/Users/christian/data/WAID-main/WAID/images/train_sample"  # This should contain subfolders (ClassA, ClassB, ...)
    embedding_save_path = "embeddings_with_labels.csv"
    tsne_embedding_save_path = "tsne_embeddings_with_labels.csv"

    image_root_dir = Path(
        "/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_112/val")

    images_list = [i for i in image_root_dir.rglob("*.jpg") if not str(i).startswith(".")]
    # randomply sample 100 images
    # images_list = random.sample(images_list, 500)
    class_labels = [x.parent.stem for x in images_list]

    results_df = extractor2.extract_from_image_list(
        image_paths=images_list,
        class_labels=class_labels,  # Optional
    class_mapping=["empty", "iguana"],  # Maps index 0 to "empty" and index 1 to "iguana"

        save_path='logits_results.csv'
    )

    fig = extractor2.visualize_binary_logits(dataframe=results_df)
    plt.show()
    results_df

    fig_2 = extractor2.visualize_confident_errors(dataframe=results_df, image_dir=image_root_dir)

    plt.show()
