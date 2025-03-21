from pathlib import Path
import shutil
import random
import os
from typing import Union, List, Tuple, Optional
import argparse
from tqdm import tqdm


def split_dataset(
        source_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.8,
        copy_files: bool = True,
        random_seed: int = 42,
        valid_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
        max_samples_per_class: Optional[int] = None,
        total_max_samples: Optional[int] = None
) -> Tuple[int, int]:
    """
    Split a dataset into training and validation sets.

    Args:
        source_dir: Source directory containing class folders
        output_dir: Output directory (if None, will create train/val inside source_dir)
        train_ratio: Ratio of training data (0.0 to 1.0)
        copy_files: Whether to copy files (True) or move them (False)
        random_seed: Random seed for reproducibility
        valid_extensions: File extensions to include
        max_samples_per_class: Maximum number of samples to use per class (None for all)
        total_max_samples: Maximum total samples across all classes (None for all)

    Returns:
        Tuple of (num_train_files, num_val_files)
    """
    # Convert to Path objects
    source_dir = Path(source_dir)

    # Set up output directory
    if output_dir is None:
        output_dir = source_dir
    else:
        output_dir = Path(output_dir)

    # Create train and val directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'

    # Ensure output directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Track file counts
    train_count = 0
    val_count = 0

    # Get class directories
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith('.')
                  and d.name not in ['train', 'val']]

    if not class_dirs:
        raise ValueError(f"No class directories found in {source_dir}")

    print(f"Found {len(class_dirs)} classes: {', '.join(d.name for d in class_dirs)}")

    # If total_max_samples is specified, calculate per-class limit
    per_class_limit = None
    if total_max_samples is not None:
        per_class_limit = total_max_samples // len(class_dirs)
        print(f"Limiting to approximately {total_max_samples} total samples ({per_class_limit} per class)")
        # Override max_samples_per_class if it's larger or not specified
        if max_samples_per_class is None or per_class_limit < max_samples_per_class:
            max_samples_per_class = per_class_limit

    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")

        # Create class directories in train and val
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        # Get all valid image files
        image_files = [f for f in class_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in valid_extensions
                       and not f.name.startswith('.')]

        if not image_files:
            print(f"  Warning: No images found in {class_dir}")
            continue

        # Shuffle the files
        random.shuffle(image_files)

        # Limit samples per class if specified
        if max_samples_per_class is not None:
            image_files = image_files[:max_samples_per_class]
            print(f"  Limiting to {len(image_files)} samples for class {class_name}")

        # Split into train and val
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        # Update counts
        train_count += len(train_files)
        val_count += len(val_files)

        # Copy or move files
        op = shutil.copy2 if copy_files else shutil.move
        operation_name = "Copying" if copy_files else "Moving"

        # Process training files
        print(f"  {operation_name} {len(train_files)} files to training set")
        for src_file in tqdm(train_files, desc=f"  Train {class_name}", ncols=80):
            dst_file = train_dir / class_name / src_file.name
            op(src_file, dst_file)

        # Process validation files
        print(f"  {operation_name} {len(val_files)} files to validation set")
        for src_file in tqdm(val_files, desc=f"  Val {class_name}", ncols=80):
            dst_file = val_dir / class_name / src_file.name
            op(src_file, dst_file)

    print(f"\nSplit complete:")
    print(f"  Training set: {train_count} images")
    print(f"  Validation set: {val_count} images")
    print(
        f"  Train/val ratio: {train_count / (train_count + val_count):.2f}/{val_count / (train_count + val_count):.2f}")
    print(f"  Training data: {train_dir}")
    print(f"  Validation data: {val_dir}")

    return train_count, val_count





if __name__ == '__main__':
    split_dataset(
        source_dir=Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224"),
        output_dir=Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224_train_val"),
        train_ratio=0.9,
        copy_files=True,
        random_seed=42,
        max_samples_per_class=5000,

    )