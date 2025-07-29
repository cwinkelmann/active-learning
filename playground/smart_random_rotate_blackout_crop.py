import random
import numpy as np
from typing import Dict, List, Tuple, Any
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import cv2


class SmartRandomCrop(DualTransform):
    """
    Custom augmentation that performs random cropping with intelligent bounding box handling.

    Features:
    - Randomly crops the image
    - Removes annotations for boxes that are significantly cut off at edges
    - Blacks out remaining portions of removed boxes
    - Can skip images that would result in no remaining annotations

    Args:
        height (int): Target height of the crop
        width (int): Target width of the crop
        min_visibility (float): Minimum visibility ratio for keeping a bbox (0.0 to 1.0)
        blackout_partial (bool): Whether to black out remaining portions of removed boxes
        allow_empty (bool): Whether to allow crops with no remaining annotations
        max_empty_attempts (int): Maximum attempts before accepting an empty crop
        always_apply (bool): Whether to always apply this transform
        p (float): Probability of applying this transform
    """

    def __init__(
            self,
            height: int,
            width: int,
            min_visibility: float = 0.5,
            blackout_partial: bool = True,
            allow_empty: bool = False,
            max_empty_attempts: int = 10,
            always_apply: bool = False,
            p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.min_visibility = min_visibility
        self.blackout_partial = blackout_partial
        self.allow_empty = allow_empty
        self.max_empty_attempts = max_empty_attempts

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for the transform based on image and targets."""
        img = params["image"]
        h, w = img.shape[:2]

        # Try multiple times to get a crop with annotations (if allow_empty=False)
        for attempt in range(self.max_empty_attempts):
            # Generate random crop coordinates
            if h <= self.height:
                y_min = 0
                y_max = h
            else:
                y_min = random.randint(0, h - self.height)
                y_max = y_min + self.height

            if w <= self.width:
                x_min = 0
                x_max = w
            else:
                x_min = random.randint(0, w - self.width)
                x_max = x_min + self.width

            crop_params = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'orig_height': h,
                'orig_width': w
            }

            # If we allow empty crops or this isn't the last attempt,
            # check if we'd have any remaining boxes
            if self.allow_empty or attempt == self.max_empty_attempts - 1:
                return crop_params

            # Check if this crop would leave us with any valid annotations
            if 'bboxes' in params and len(params['bboxes']) > 0:
                if self._would_have_valid_boxes(params['bboxes'], crop_params):
                    return crop_params

        return crop_params

    def _would_have_valid_boxes(self, bboxes: List, crop_params: Dict) -> bool:
        """Check if the crop would result in any valid bounding boxes."""
        h, w = crop_params['orig_height'], crop_params['orig_width']
        x_min, y_min = crop_params['x_min'], crop_params['y_min']
        x_max, y_max = crop_params['x_max'], crop_params['y_max']

        # Normalize crop coordinates to match bbox format
        crop_x_min = x_min / w
        crop_y_min = y_min / h
        crop_x_max = x_max / w
        crop_y_max = y_max / h

        for bbox in bboxes:
            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox[:4]

            # Calculate intersection
            inter_x_min = max(bbox_x_min, crop_x_min)
            inter_y_min = max(bbox_y_min, crop_y_min)
            inter_x_max = min(bbox_x_max, crop_x_max)
            inter_y_max = min(bbox_y_max, crop_y_max)

            if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
                continue  # No intersection

            # Calculate visibility ratio
            bbox_area = (bbox_x_max - bbox_x_min) * (bbox_y_max - bbox_y_min)
            if bbox_area == 0:
                continue

            intersect_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            visibility = intersect_area / bbox_area

            if visibility >= self.min_visibility:
                return True

        return False

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply the crop to the image."""
        x_min = params['x_min']
        y_min = params['y_min']
        x_max = params['x_max']
        y_max = params['y_max']

        # Crop the image
        cropped = img[y_min:y_max, x_min:x_max]

        # Resize if necessary (in case the crop is smaller than target size)
        if cropped.shape[0] != self.height or cropped.shape[1] != self.width:
            cropped = cv2.resize(cropped, (self.width, self.height))

        # Always apply blackout for boxes that need to be blacked out
        boxes_to_blackout = params.get('boxes_to_blackout', [])
        for x_min_bo, y_min_bo, x_max_bo, y_max_bo in boxes_to_blackout:
            # Black out the partial boxes
            cropped[y_min_bo:y_max_bo, x_min_bo:x_max_bo] = 0

        return cropped

    def apply_to_bboxes(self, bboxes: List, **params) -> List:
        """Apply the crop to bounding boxes."""
        if not bboxes:
            return []

        h, w = params['orig_height'], params['orig_width']
        x_min, y_min = params['x_min'], params['y_min']
        x_max, y_max = params['x_max'], params['y_max']

        # Crop dimensions
        crop_width = x_max - x_min
        crop_height = y_max - y_min

        new_bboxes = []
        boxes_to_blackout = []

        for bbox in bboxes:
            # Bboxes are already normalized (0-1) by Albumentations
            bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox[:4]

            # Convert to absolute coordinates for intersection calculation
            abs_bbox_x_min = bbox_x_min * w
            abs_bbox_y_min = bbox_y_min * h
            abs_bbox_x_max = bbox_x_max * w
            abs_bbox_y_max = bbox_y_max * h

            # Calculate intersection with crop area
            inter_x_min = max(abs_bbox_x_min, x_min)
            inter_y_min = max(abs_bbox_y_min, y_min)
            inter_x_max = min(abs_bbox_x_max, x_max)
            inter_y_max = min(abs_bbox_y_max, y_max)

            # Skip if no intersection
            if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
                continue

            # Calculate visibility ratio
            bbox_area = (abs_bbox_x_max - abs_bbox_x_min) * (abs_bbox_y_max - abs_bbox_y_min)
            if bbox_area == 0:
                continue

            intersect_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            visibility = intersect_area / bbox_area

            if visibility >= self.min_visibility:
                # Keep this bbox, adjust coordinates relative to crop
                new_x_min = max(0, abs_bbox_x_min - x_min)
                new_y_min = max(0, abs_bbox_y_min - y_min)
                new_x_max = min(crop_width, abs_bbox_x_max - x_min)
                new_y_max = min(crop_height, abs_bbox_y_max - y_min)

                # Scale coordinates if we resized the crop
                if self.height != crop_height or self.width != crop_width:
                    scale_y = self.height / crop_height
                    scale_x = self.width / crop_width
                    new_x_min *= scale_x
                    new_y_min *= scale_y
                    new_x_max *= scale_x
                    new_y_max *= scale_y

                # Convert back to normalized coordinates (0-1)
                final_x_min = new_x_min / self.width
                final_y_min = new_y_min / self.height
                final_x_max = new_x_max / self.width
                final_y_max = new_y_max / self.height

                # Create new bbox, keeping additional fields
                new_bbox = [final_x_min, final_y_min, final_x_max, final_y_max] + list(bbox[4:])
                new_bboxes.append(new_bbox)

            else:
                # Check if the box is on the edge of the original image
                is_on_edge = (abs_bbox_x_min <= 0 or abs_bbox_y_min <= 0 or 
                             abs_bbox_x_max >= w or abs_bbox_y_max >= h)

                # Mark this box for blackout if it's partially visible and either:
                # 1. It's on the edge of the original image (always blackout split edge boxes)
                # 2. Blackout is enabled for all partial boxes with visibility > 0
                if (is_on_edge or (self.blackout_partial and visibility > 0)):
                    # Convert to crop-relative coordinates
                    blackout_x_min = max(0, abs_bbox_x_min - x_min)
                    blackout_y_min = max(0, abs_bbox_y_min - y_min)
                    blackout_x_max = min(crop_width, abs_bbox_x_max - x_min)
                    blackout_y_max = min(crop_height, abs_bbox_y_max - y_min)

                    # Scale if resized
                    if self.height != crop_height or self.width != crop_width:
                        scale_y = self.height / crop_height
                        scale_x = self.width / crop_width
                        blackout_x_min *= scale_x
                        blackout_y_min *= scale_y
                        blackout_x_max *= scale_x
                        blackout_y_max *= scale_y

                    boxes_to_blackout.append((
                        int(blackout_x_min), int(blackout_y_min),
                        int(blackout_x_max), int(blackout_y_max)
                    ))

        # Store boxes to blackout for the image transform
        params['boxes_to_blackout'] = boxes_to_blackout

        return new_bboxes

    def apply_with_blackout(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply crop and blackout removed boxes."""
        # First apply the regular crop
        img = self.apply(img, **params)

        # Then blackout the partial boxes
        boxes_to_blackout = params.get('boxes_to_blackout', [])
        for x_min, y_min, x_max, y_max in boxes_to_blackout:
            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, img.shape[1]))
            y_min = max(0, min(y_min, img.shape[0]))
            x_max = max(0, min(x_max, img.shape[1]))
            y_max = max(0, min(y_max, img.shape[0]))

            if x_max > x_min and y_max > y_min:
                img[y_min:y_max, x_min:x_max] = 0  # Black out

        return img

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "min_visibility", "blackout_partial",
                "allow_empty", "max_empty_attempts")


# Example usage with your data format
def create_augmentation_pipeline():
    """Create an example augmentation pipeline using SmartRandomCrop."""

    return A.Compose([
        SmartRandomCrop(
            height=512,
            width=512,
            min_visibility=0.6,  # Keep boxes that are at least 60% visible
            blackout_partial=True,  # Black out partial boxes
            allow_empty=False,  # Try to avoid empty images
            max_empty_attempts=5,  # Try up to 5 times to get non-empty crop
            p=0.8
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # Your data format: xmin, ymin, xmax, ymax
        label_fields=['labels'],  # Field name for your labels
        min_area=0,
        min_visibility=0
    ))


# Test function with your data format
def test_with_your_data():
    """Test the SmartRandomCrop transform with your data format."""
    import pandas as pd
    import matplotlib.pyplot as plt

    # Example of loading your CSV data
    # df = pd.read_csv('your_data.csv', names=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])

    # Mock your data format for testing
    image_path = "everglades_train___52850051_x640_y0.jpg"

    # Create a dummy image (replace with actual image loading)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Your bounding boxes in absolute coordinates (pascal_voc format)
    bboxes = [
        [36, 323, 86, 373],  # Bird 1
        [160, 200, 210, 250],  # Bird 2 (added for testing)
        [400, 50, 450, 100],  # Bird 3 (added for testing)
    ]
    labels = ['Bird', 'Bird', 'Bird']  # Your labels

    # Create the augmentation
    transform = create_augmentation_pipeline()

    # Apply the transform
    transformed = transform(
        image=image,
        bboxes=bboxes,
        labels=labels
    )

    print(f"Original image shape: {image.shape}")
    print(f"Original bboxes count: {len(bboxes)}")
    print(f"Original bboxes: {bboxes}")
    print(f"Transformed image shape: {transformed['image'].shape}")
    print(f"Transformed bboxes count: {len(transformed['bboxes'])}")
    print(f"Remaining bboxes: {transformed['bboxes']}")
    print(f"Remaining labels: {transformed['labels']}")

    return transformed


# Function to process your Michigan birds CSV data
def process_michigan_birds_data(csv_path=None, image_dir=None, output_dir=None, num_samples=None, save_output=False):
    """
    Process the Michigan birds dataset with augmentation.

    Args:
        csv_path: Path to michigan_train.csv
        image_dir: Directory containing the images
        output_dir: Directory to save augmented images and annotations (optional)
        num_samples: Number of images to process (None for all)
        save_output: Whether to save augmented images and annotations
    """
    import pandas as pd
    import os
    from collections import defaultdict
    import matplotlib.pyplot as plt

    # Default paths if not provided
    if csv_path is None:
        csv_path = '/Volumes/G-DRIVE/Datasets/deep_forest_birds/everglades/everglades_train.csv'
    if image_dir is None:
        image_dir = '/Volumes/G-DRIVE/Datasets/deep_forest_birds/everglades'

    print(f"Loading data from: {csv_path}")
    print(f"Looking for images in: {image_dir}")

    # Read your CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} annotations")
        print("CSV columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())

        # Check required columns
        required_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'image_path', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            return

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Group by image (since each row is one bbox)
    grouped = df.groupby('image_path')
    print(f"\nFound {len(grouped)} unique images")

    # Limit number of samples if specified
    if num_samples:
        image_names = list(grouped.groups.keys())[:num_samples]
        grouped = {name: grouped.get_group(name) for name in image_names}

    transform = create_augmentation_pipeline()

    processed_count = 0
    successful_count = 0
    results = []

    for image_name, group in (grouped.items() if isinstance(grouped, dict) else grouped):
        processed_count += 1

        # Construct full image path
        img_path = os.path.join(image_dir, image_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_h, original_w = image.shape[:2]

            # Extract bboxes and labels for this image
            bboxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            labels = group['label'].tolist()

            print(
                f"\nProcessing {image_name} ({processed_count}/{len(grouped) if not isinstance(grouped, dict) else len(grouped)})")
            print(f"  Image size: {original_w}x{original_h}")
            print(f"  Original bboxes: {len(bboxes)}")

            # Apply augmentation
            augmented = transform(
                image=image,
                bboxes=bboxes,
                labels=labels
            )

            aug_h, aug_w = augmented['image'].shape[:2]
            print(f"  Augmented size: {aug_w}x{aug_h}")
            print(f"  Remaining bboxes: {len(augmented['bboxes'])}")

            # Store results
            result = {
                'original_image_name': image_name,
                'original_image': image,
                'original_bboxes': bboxes,
                'original_labels': labels,
                'augmented_image': augmented['image'],
                'augmented_bboxes': augmented['bboxes'],
                'augmented_labels': augmented['labels']
            }
            results.append(result)

            # Save augmented data if requested
            if save_output and output_dir:
                os.makedirs(output_dir, exist_ok=True)

                # Save augmented image
                aug_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                output_name = f"aug_{image_name}"
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, aug_image_bgr)

                # Save augmented annotations (convert back to absolute coordinates)
                aug_annotations = []
                for bbox, label in zip(augmented['bboxes'], augmented['labels']):
                    # Convert from normalized back to absolute coordinates
                    x_min, y_min, x_max, y_max = bbox[:4]
                    abs_coords = [
                        x_min * aug_w, y_min * aug_h,
                        x_max * aug_w, y_max * aug_h
                    ]
                    aug_annotations.append([output_name] + abs_coords + [label])

                # Save annotations to CSV
                if aug_annotations:
                    ann_df = pd.DataFrame(aug_annotations,
                                          columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
                    ann_path = os.path.join(output_dir, f"annotations_{image_name.split('.')[0]}.csv")
                    ann_df.to_csv(ann_path, index=False)

            successful_count += 1

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Summary ===")
    print(f"Processed: {processed_count} images")
    print(f"Successful: {successful_count} images")

    return results


def visualize_augmentation_results(results, num_to_show=3):
    """Visualize before/after augmentation results."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    num_to_show = min(num_to_show, len(results))

    fig, axes = plt.subplots(num_to_show, 2, figsize=(15, 5 * num_to_show))
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results[:num_to_show]):
        # Original image
        ax_orig = axes[i, 0]
        ax_orig.imshow(result['original_image'])
        ax_orig.set_title(f"Original: {result['original_image_name']}")

        # Draw original bboxes
        for bbox, label in zip(result['original_bboxes'], result['original_labels']):
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax_orig.add_patch(rect)
            ax_orig.text(x_min, y_min - 10, label, color='red', fontsize=8)

        # Augmented image
        ax_aug = axes[i, 1]
        ax_aug.imshow(result['augmented_image'])
        ax_aug.set_title(f"Augmented ({len(result['augmented_bboxes'])} boxes)")

        # Draw augmented bboxes (convert from normalized to absolute)
        h, w = result['augmented_image'].shape[:2]
        for bbox, label in zip(result['augmented_bboxes'], result['augmented_labels']):
            x_min, y_min, x_max, y_max = bbox[:4]
            # Convert from normalized coordinates
            abs_x_min, abs_y_min = x_min, y_min
            abs_x_max, abs_y_max = x_max, y_max

            rect = patches.Rectangle((abs_x_min, abs_y_min),
                                     abs_x_max - abs_x_min, abs_y_max - abs_y_min,
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax_aug.add_patch(rect)
            ax_aug.text(abs_x_min, abs_y_min - 10, label, color='blue', fontsize=8)

        ax_orig.axis('off')
        ax_aug.axis('off')

    plt.tight_layout()
    plt.show()


# Updated test function for your specific dataset
def test_michigan_birds():
    """Test the SmartRandomCrop transform with the Michigan birds dataset."""

    print("Testing SmartRandomCrop with Michigan birds dataset...")

    # Process a few samples
    results = process_michigan_birds_data(
        num_samples=10,  # Process just 5 images for testing
        save_output=True,  # Don't save output for testing
        output_dir='/Volumes/G-DRIVE/Datasets/deep_forest_birds/hasty_style/augmented'  # Specify your output directory if needed
    )

    if results:
        print(f"\nProcessed {len(results)} images successfully!")

        # Show statistics
        total_orig_boxes = sum(len(r['original_bboxes']) for r in results)
        total_aug_boxes = sum(len(r['augmented_bboxes']) for r in results)

        print(f"Total original boxes: {total_orig_boxes}")
        print(f"Total augmented boxes: {total_aug_boxes}")
        print(f"Box retention rate: {total_aug_boxes / total_orig_boxes * 100:.1f}%")

        # Visualize results
        visualize_augmentation_results(results, num_to_show=3)

        return results
    else:
        print("No results to show. Check your file paths.")
        return None


if __name__ == "__main__":
    test_michigan_birds()
