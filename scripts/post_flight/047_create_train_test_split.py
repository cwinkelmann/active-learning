import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
from sklearn.model_selection import train_test_split


def split_and_copy_data(csv_path, empty_images_dir, output_base_dir,
                        train_size=0.7, val_size=0.1, test_size=0.2,
                        random_state=42):
    """
    Split the dataset (including empty images) and copy to respective folders.

    Args:
        csv_path: Path to the CSV file with annotations
        empty_images_dir: Directory containing empty images
        output_base_dir: Base directory to create train/val/test folders
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)



    print(f"Read {len(df)} annotations from {csv_path}")

    # Get unique image paths from annotations
    annotated_images = df['tile_name'].unique()
    print(f"Found {len(annotated_images)} unique annotated images")

    # Get empty images
    empty_images_dir = Path(empty_images_dir)
    empty_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        empty_images.extend(list(empty_images_dir.glob(ext)))

    # Convert to string paths for consistency
    empty_images = [str(img) for img in empty_images]
    print(f"Found {len(empty_images)} empty images in {empty_images_dir}")

    # Create train, val, test splits for annotated images
    train_annotated, temp_annotated = train_test_split(
        annotated_images, train_size=train_size, random_state=random_state
    )

    # Adjust validation size relative to the temp set
    val_ratio = val_size / (val_size + test_size)
    val_annotated, test_annotated = train_test_split(
        temp_annotated, train_size=val_ratio, random_state=random_state
    )

    # Create splits for empty images with the same ratio
    train_empty, temp_empty = train_test_split(
        empty_images, train_size=train_size, random_state=random_state
    )

    val_empty, test_empty = train_test_split(
        temp_empty, train_size=val_ratio, random_state=random_state
    )

    # Create output directories
    output_base_dir = Path(output_base_dir)
    train_dir = output_base_dir / 'train'
    val_dir = output_base_dir / 'val'
    test_dir = output_base_dir / 'test'

    for directory in [output_base_dir, train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # Create mapping for easier access - annotated images
    image_to_split = {}
    for img in train_annotated:
        image_to_split[img] = 'train'
    for img in val_annotated:
        image_to_split[img] = 'val'
    for img in test_annotated:
        image_to_split[img] = 'test'

    # Add mapping for empty images
    for img in train_empty:
        image_to_split[img] = 'train'
    for img in val_empty:
        image_to_split[img] = 'val'
    for img in test_empty:
        image_to_split[img] = 'test'

    # Add split column to DataFrame for annotated images
    df['split'] = df['tile_name'].map(image_to_split)

    # Copy annotated files to respective directories
    copied_files = 0
    skipped_files = 0

    print(f"\nCopying annotated files to {output_base_dir}...")
    for img_path in annotated_images:
        try:
            source_path = Path(img_path)
            if not source_path.exists():
                print(f"Warning: Source file not found: {source_path}")
                skipped_files += 1
                continue

            split = image_to_split[img_path]
            dest_dir = output_base_dir / split
            dest_path = dest_dir / source_path.name

            shutil.copy2(source_path, dest_path)
            copied_files += 1

        except Exception as e:
            print(f"Error copying {img_path}: {str(e)}")
            skipped_files += 1

    print(f"Copied {copied_files} annotated files, skipped {skipped_files} files")

    # Copy empty files to respective directories
    empty_copied = 0
    empty_skipped = 0

    print(f"\nCopying empty files to {output_base_dir}...")
    for img_path in empty_images:
        try:
            source_path = Path(img_path)
            if not source_path.exists():
                print(f"Warning: Empty source file not found: {source_path}")
                empty_skipped += 1
                continue

            split = image_to_split[img_path]
            dest_dir = output_base_dir / split
            dest_path = dest_dir / source_path.name

            shutil.copy2(source_path, dest_path)
            empty_copied += 1

        except Exception as e:
            print(f"Error copying empty image {img_path}: {str(e)}")
            empty_skipped += 1

    print(f"Copied {empty_copied} empty files, skipped {empty_skipped} files")

    df = df.rename(columns={
        'tile_name': 'images',
        'local_pixel_x': 'x',
        'local_pixel_y': 'y'
    })

    # Create annotation files for each split
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_annotations_path = output_base_dir / f"{split}_annotations.csv"
        split_df.to_csv(split_annotations_path, index=False)
        print(f"Created {split} annotations with {len(split_df)} entries")

    # Create a comprehensive inventory for each split
    all_files = []

    # Gather information about annotated images
    for img_path in annotated_images:
        file_info = {
            'image_path': img_path,
            'split': image_to_split.get(img_path, 'unknown'),
            'has_annotations': True,
            'file_exists': Path(img_path).exists()
        }
        all_files.append(file_info)

    # Gather information about empty images
    for img_path in empty_images:
        file_info = {
            'image_path': img_path,
            'split': image_to_split.get(img_path, 'unknown'),
            'has_annotations': False,
            'file_exists': Path(img_path).exists()
        }
        all_files.append(file_info)

    # Create inventory DataFrame
    inventory_df = pd.DataFrame(all_files)
    inventory_path = output_base_dir / "dataset_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    print(f"\nCreated complete dataset inventory at {inventory_path}")

    # Print summary
    print("\nSplit Summary:")
    print("\nAnnotated Images:")
    print(f"Train: {len(train_annotated)} images ({len(train_annotated) / len(annotated_images) * 100:.1f}%)")
    print(f"Validation: {len(val_annotated)} images ({len(val_annotated) / len(annotated_images) * 100:.1f}%)")
    print(f"Test: {len(test_annotated)} images ({len(test_annotated) / len(annotated_images) * 100:.1f}%)")

    print("\nEmpty Images:")
    print(f"Train: {len(train_empty)} images ({len(train_empty) / len(empty_images) * 100:.1f}%)")
    print(f"Validation: {len(val_empty)} images ({len(val_empty) / len(empty_images) * 100:.1f}%)")
    print(f"Test: {len(test_empty)} images ({len(test_empty) / len(empty_images) * 100:.1f}%)")

    print("\nTotal Images:")
    total_train = len(train_annotated) + len(train_empty)
    total_val = len(val_annotated) + len(val_empty)
    total_test = len(test_annotated) + len(test_empty)
    total_images = len(annotated_images) + len(empty_images)

    print(f"Train: {total_train} images ({total_train / total_images * 100:.1f}%)")
    print(f"Validation: {total_val} images ({total_val / total_images * 100:.1f}%)")
    print(f"Test: {total_test} images ({total_test / total_images * 100:.1f}%)")

    # Count annotations per split
    train_annotations = len(df[df['split'] == 'train'])
    val_annotations = len(df[df['split'] == 'val'])
    test_annotations = len(df[df['split'] == 'test'])

    print("\nAnnotation Distribution:")
    print(f"Train: {train_annotations} annotations ({train_annotations / len(df) * 100:.1f}%)")
    print(f"Validation: {val_annotations} annotations ({val_annotations / len(df) * 100:.1f}%)")
    print(f"Test: {test_annotations} annotations ({test_annotations / len(df) * 100:.1f}%)")

    return df, inventory_df


if __name__ == "__main__":
    csv_path = "/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_analysis/herdnet_annotations.csv"
    output_dir = "dataset"

    empty_images_dir = "/Volumes/2TB/DD_MS_COG_ALL_TILES/herdnet_512/empty"

    # Perform the split and copy files
    print(f"Processing data from {csv_path} and empty images from {empty_images_dir}...")
    df_with_splits, inventory = split_and_copy_data(
        csv_path,
        empty_images_dir,
        output_dir,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2)

    print(f"{df_with_splits} and {inventory}")