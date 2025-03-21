from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
from typing import Union, Optional, List, Tuple


def resize_image(
        image_path: Path,
        output_path: Path,
        size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Resize a single image and save to output path.

    Args:
        image_path: Source image path
        output_path: Destination path for resized image
        size: Target size (width, height)

    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize with LANCZOS resampling for better quality
            resized_img = img.resize(size, Image.LANCZOS)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with optimization
            resized_img.save(output_path, 'JPEG', quality=95, optimize=True)
            return True
    except (IOError, OSError, Image.DecompressionBombError) as e:
        print(f"Error processing {image_path}: {e}")
        return False


def resize_dataset(
        source_dir: Union[str, Path],
        output_dir: Union[str, Path],
        size: Tuple[int, int] = (224, 224),
        max_workers: Optional[int] = None,
        skip_existing: bool = True
) -> Tuple[int, int]:
    """
    Resize all images in the source directory to the specified size and save to output directory.

    Args:
        source_dir: Source directory containing 'Cat' and 'Dog' folders
        output_dir: Output directory for resized images
        size: Target size (width, height)
        max_workers: Maximum number of parallel workers, None for auto
        skip_existing: Skip images that already exist in the output directory

    Returns:
        Tuple of (number of successfully processed images, total number of images)
    """
    # Convert to Path objects
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all categories (Cat and Dog folders)
    categories = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    # Prepare list of image paths and their output paths
    image_pairs = []
    for category in categories:
        category_output_dir = output_dir / category.name
        category_output_dir.mkdir(exist_ok=True)

        # Get all image files in this category
        image_files = [
            f for f in category.iterdir()
            if f.is_file() and not f.name.startswith('.') and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ]

        for img_path in image_files:
            out_path = category_output_dir / f"{img_path.stem}.jpg"  # Use .jpg for all output files

            # Skip if output exists and we're told to skip
            if skip_existing and out_path.exists():
                continue

            image_pairs.append((img_path, out_path))

    total_images = len(image_pairs)
    successful = 0

    if total_images == 0:
        print("No images to process!")
        return 0, 0

    print(f"Resizing {total_images} images to {size[0]}x{size[1]}...")

    # Process with parallel workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(resize_image, img_path, out_path, size): img_path
            for img_path, out_path in image_pairs
        }

        # Process results with a progress bar
        with tqdm(total=total_images, desc="Resizing images") as pbar:
            for future in futures:
                pbar.update(1)
                if future.result():
                    successful += 1

    print(f"Processed {successful}/{total_images} images successfully!")

    return successful, total_images


if __name__ == "__main__":
    # Example usage
    source_directory = Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages")
    output_directory = Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224")

    # Resize to 224x224
    successful, total = resize_dataset(source_directory, output_directory, size=(224, 224))

    # Print summary
    print(f"Finished resizing: {successful} of {total} images processed successfully")
    print(f"Resized images saved to: {output_directory}")