# Example usage
from pathlib import Path

from active_learning.util.image_quality.image_quality import analyze_image_quality, calculate_laplacian_variance, \
    calculate_gradient_magnitude, calculate_local_contrast, calculate_tenengrad
from active_learning.util.image_quality.image_quality_from_grid import visualise_image_quality

EXTRACT_DATA = True



if __name__ == "__main__":

    # Replace with your image folder path
    # IMAGE_FOLDER = Path("/Users/christian/data/training_data/2025_04_06/images_2024_06_04/FMO03")
    # IMAGE_FOLDER = Path("/Users/christian/data/training_data/2025_04_06/images_2024_06_04/Fer_FCD01-02-03_20122021_single_images")
    # IMAGE_FOLDER = Path("/Users/christian/data/training_data/2025_04_06/images_2024_06_04/Fer_FCD01-02-03_20122021")


    ## The FCD01 Trilogy
    # IMAGE_FOLDER = Path("/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/2020_2021_2022_2023_2024/Fernandina/FCD01_20122021")
    # IMAGE_FOLDER = Path("/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/2020_2021_2022_2023_2024/Fernandina/FCD01_22012023")
    # IMAGE_FOLDER = Path("/Volumes/u235425.your-storagebox.de/Iguanas_From_Above/2020_2021_2022_2023_2024/Fernandina/FCD01_04052024")
    # IMAGE_FOLDER = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Fernandina")
    IMAGE_FOLDER = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Floreana") # Done
    IMAGE_FOLDER = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Isabela") # Done
    IMAGE_FOLDER = Path("/Volumes/G-DRIVE/Iguanas_From_Above/2020_2021_2022_2023_2024/Pinzón")

    grid_size = (9, 12)  # (y, x), 10x10 grid, Aspect Ratio of drone images is 4:3
    # Optional: Specify output directory
    OUTPUT_DIR = Path("/Volumes/G-DRIVE/Iguanas_From_Above/database/image_quality") / Path(f"image_quality_results_{IMAGE_FOLDER.name}")

    # Define metrics
    metrics = {
        'laplacian_variance': calculate_laplacian_variance,
        'gradient_magnitude': calculate_gradient_magnitude,
        'local_contrast': calculate_local_contrast,
        'tenengrad': calculate_tenengrad
    }

    if EXTRACT_DATA:
        # Run the analysis
        results = analyze_image_quality(
            folder_path=IMAGE_FOLDER,
            metrics=metrics,
            grid_size=grid_size,
            max_images=None,  # Process up to N images
            output_dir=OUTPUT_DIR,
        )


    # Run the analysis
    results = visualise_image_quality(
        grid_path=OUTPUT_DIR / "image_quality.parquet",
        metrics=metrics,
        grid_size=grid_size,  # 10x10 grid
        output_dir=OUTPUT_DIR,
    )

    # Show the normalized quality map
    # plt.figure(results['normalized_fig'].number)
    # plt.show()