"""
create a flat dataset from a coco dataset
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from collections import defaultdict
import pandas as pd

from playground.saliency.saliency import measure_camouflage, analyze_camouflage_for_dataset

# Your paths
WAID_train = Path(
    '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/waid_coco/train')
WAID_annotations = Path(
    '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/waid_coco/train/_annotations.coco.json')


def load_coco_annotations(annotation_path):
    """Load COCO format annotations from JSON file"""
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def get_category_mapping(coco_data):
    """Create ID to category name mapping"""
    id_to_name = {}
    for category in coco_data['categories']:
        id_to_name[category['id']] = category['name']
    return id_to_name


def get_image_annotations(coco_data, image_id):
    """Get all annotations for a specific image ID"""
    return [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]


def get_image_info(coco_data, image_id):
    """Get image information for a specific image ID"""
    for img in coco_data['images']:
        if img['id'] == image_id:
            return img
    return None


def visualize_annotations(image_path, annotations, id_to_name, figsize=(12, 12)):
    """Visualize image with its annotations"""
    # Load image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Display the image
    ax.imshow(img_np)

    # Plot each bounding box
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ann in annotations:
        # Get coordinates
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox

        # Create a Rectangle patch
        category_id = ann['category_id']
        category_name = id_to_name.get(category_id, f"Unknown ({category_id})")
        color = colors[category_id % len(colors)]

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label
        ax.text(x, y - 5, category_name, color=color, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def explore_dataset(coco_data, images_dir, num_samples=5):
    """Randomly sample and visualize images from the dataset"""
    id_to_name = get_category_mapping(coco_data)

    # Group annotations by image_id
    image_to_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_to_annotations[ann['image_id']].append(ann)

    # Get image ids that have annotations
    image_ids = list(image_to_annotations.keys())

    # Randomly sample images
    if len(image_ids) > num_samples:
        sampled_ids = random.sample(image_ids, num_samples)
    else:
        sampled_ids = image_ids

    for img_id in sampled_ids:
        img_info = get_image_info(coco_data, img_id)
        if img_info:
            img_path = images_dir / img_info['file_name']
            if img_path.exists():
                print(f"Image: {img_info['file_name']}")
                print(f"Dimensions: {img_info['width']} x {img_info['height']}")
                annotations = image_to_annotations[img_id]
                print(f"Annotations: {len(annotations)}")

                # List categories in this image
                categories = [id_to_name.get(ann['category_id'], "Unknown") for ann in annotations]
                print(f"Categories: {', '.join(set(categories))}")
                print("-" * 50)

                visualize_annotations(img_path, annotations, id_to_name)
            else:
                print(f"Image file not found: {img_path}")

def convert_annotations_to_dataframe(coco_data, images_dir):
    """
    Convert COCO annotations to a pandas DataFrame.

    Parameters:
    -----------
    coco_data : dict
        The loaded COCO annotations JSON data
    images_dir : Path
        Directory containing the images

    Returns:
    --------
    pandas.DataFrame
        DataFrame with one row per annotation
    """
    # Create category mapping
    id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create image mapping
    image_info = {img['id']: img for img in coco_data['images']}

    # Initialize lists to store data
    rows = []

    # Process each annotation
    for ann in coco_data['annotations']:
        img_id = ann['image_id']

        # Skip if image info not found
        if img_id not in image_info:
            continue

        img = image_info[img_id]
        file_name = img['file_name']
        img_path = str(images_dir / file_name)

        # Get bbox coordinates
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])

        # Create a row
        row = {
            'annotation_id': ann['id'],
            'image_id': img_id,
            'file_name': file_name,
            'image_path': img_path,
            'category_id': ann['category_id'],
            'category_name': id_to_name.get(ann['category_id'], "Unknown"),
            'bbox_x': x,
            'bbox_y': y,
            'bbox_width': w,
            'bbox_height': h,
            'area': ann.get('area', w * h),
            'iscrowd': ann.get('iscrowd', 0),
            'image_width': img.get('width', 0),
            'image_height': img.get('height', 0)
        }

        # Add segmentation if available
        if 'segmentation' in ann:
            row['has_segmentation'] = True
        else:
            row['has_segmentation'] = False

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df

if __name__ == "__main__":
    # Main execution
    try:

        waid_path = Path("waid_flat_camouflage.csv")
        if not waid_path.exists():
            # Load the COCO annotations
            coco_data = load_coco_annotations(WAID_annotations)
            annotations_df = convert_annotations_to_dataframe(coco_data, WAID_train)

            annotations_df.to_csv(waid_path, index=False)
        else:
            annotations_df = pd.read_csv(waid_path)

        camouflage_df = analyze_camouflage_for_dataset(annotations_df, method='spectral', sample_size=3)

        camouflage_df.to_csv("waid_flat_camouflage.csv", index=False)

        # Print dataset summary
        print("Dataset Summary:")
        print(f"Number of images: {len(coco_data['images'])}")
        print(f"Number of annotations: {len(coco_data['annotations'])}")
        print(f"Number of categories: {len(coco_data['categories'])}")

        # Print category information
        print("\nCategories:")
        for cat in coco_data['categories']:
            print(f"  {cat['id']}: {cat['name']}")

        # Explore sample images
        print("\nExploring sample images with annotations:")
        explore_dataset(coco_data, WAID_train)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the paths to your COCO dataset are correct.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the annotations file.")
    except Exception as e:
        print(f"Unexpected error: {e}")