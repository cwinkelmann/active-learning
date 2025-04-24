import cv2
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import time

from playground.saliency.coco_flatten import load_coco_annotations, convert_annotations_to_dataframe
from playground.saliency.saliency import measure_camouflage


def analyze_camouflage_in_chunks(annotations_df, output_path, chunk_size=100, processes=4,
                                 method='spectral', resume=True):
    """
    Process the dataset in chunks, saving progress after each chunk

    Parameters:
    -----------
    annotations_df : pandas.DataFrame
        DataFrame with annotations
    output_path : Path or str
        Path to save the results
    chunk_size : int
        Number of rows to process in each chunk
    method : str
        Saliency method to use
    resume : bool
        Whether to resume from previous progress

    Returns:
    --------
    pandas.DataFrame
        DataFrame with camouflage metrics added
    """
    output_path = Path(output_path)

    # Initialize metrics columns
    metrics_columns = [
        'saliency_ratio', 'saliency_contrast', 'normalized_saliency',
        'color_hist_similarity', 'edge_ratio', 'texture_similarity',
        'camouflage_score', 'analysis_error'
    ]

    # Check if we're resuming from previous work
    if resume and output_path.exists():
        print(f"Loading existing results from {output_path}")
        results_df = pd.read_csv(output_path)

        # Identify which rows have already been processed
        processed_mask = ~results_df['camouflage_score'].isna() | ~results_df['analysis_error'].isna()
        processed_count = processed_mask.sum()

        print(f"Found {processed_count} already processed rows")

        # Find rows that still need processing
        to_process = results_df[~processed_mask].index.tolist()
    else:
        # Start from scratch
        results_df = annotations_df.copy()
        for col in metrics_columns:
            results_df[col] = None

        to_process = results_df.index.tolist()
        processed_count = 0

    total_rows = len(to_process)
    print(f"Processing {total_rows} rows in chunks of {chunk_size}")

    # Process in chunks
    chunk_indices = [to_process[i:i + chunk_size] for i in range(0, len(to_process), chunk_size)]

    start_time = time.time()
    for chunk_num, indices in enumerate(chunk_indices):
        chunk_start_time = time.time()
        print(f"\nProcessing chunk {chunk_num + 1}/{len(chunk_indices)} ({len(indices)} rows)")

        # Process each row in this chunk
        for idx in tqdm(indices, desc=f"Chunk {chunk_num + 1}"):
            row = results_df.loc[idx]

            # Skip if file doesn't exist
            if not os.path.exists(row['image_path']):
                results_df.loc[idx, 'analysis_error'] = "File not found"
                continue

            try:
                # Load image
                image = cv2.imread(row['image_path'])

                if image is None:
                    results_df.loc[idx, 'analysis_error'] = "Failed to load image"
                    continue

                # Extract bbox
                bbox = (int(row['bbox_x']), int(row['bbox_y']),
                        int(row['bbox_width']), int(row['bbox_height']))

                # Run camouflage analysis
                metrics, _ = measure_camouflage(image, bbox, method=method)

                # Store metrics in DataFrame
                for metric_name, metric_value in metrics.items():
                    results_df.loc[idx, metric_name] = metric_value

            except Exception as e:
                results_df.loc[idx, 'analysis_error'] = str(e)

        # Save progress after each chunk
        results_df.to_csv(output_path, index=False)

        # Report on chunk completion
        chunk_end_time = time.time()
        chunk_duration = chunk_end_time - chunk_start_time
        processed_count += len(indices)

        print(f"Chunk completed in {chunk_duration:.1f} seconds")
        print(f"Progress: {processed_count}/{len(results_df)} rows ({processed_count / len(results_df) * 100:.1f}%)")

        # Estimate remaining time
        elapsed_time = chunk_end_time - start_time
        rows_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
        remaining_rows = len(results_df) - processed_count
        estimated_remaining_time = remaining_rows / rows_per_second if rows_per_second > 0 else float('inf')

        print(f"Processing rate: {rows_per_second:.2f} rows/second")
        print(f"Estimated time remaining: {estimated_remaining_time / 60:.1f} minutes")

    end_time = time.time()
    print(f"\nProcessing complete! Total time: {(end_time - start_time) / 60:.1f} minutes")

    return results_df

if __name__ == "__main__":
    # Usage example
    waid_path = Path("waid_flat_annotations.csv")
    results_path = Path("waid_flat_camouflage.csv")
    WAID_train = Path(
        '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/waid_coco/train')
    WAID_annotations = Path(
        '/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/waid_coco/train/_annotations.coco.json')

    ## iguanas
    waid_path = Path("ifa_flat_annotations.csv")
    results_path = Path("ifa_flat_camouflage.csv")

    WAID_train = Path(
        '/Users/christian/data/training_data/2025_04_18_all/train/crops_640')
    WAID_annotations = Path(
        '/Users/christian/data/training_data/2025_04_18_all/train/coco_format.json')


    if not waid_path.exists():
        # Load the COCO annotations
        coco_data = load_coco_annotations(WAID_annotations)
        annotations_df = convert_annotations_to_dataframe(coco_data, WAID_train)
        annotations_df.to_csv(waid_path, index=False)
    else:
        annotations_df = pd.read_csv(waid_path)

    # Process with chunking, resuming from previous progress if available
    camouflage_df = analyze_camouflage_in_chunks(
        annotations_df,
        results_path,
        chunk_size=350,  # Adjust chunk size based on your machine's performance
        method='spectral',
        resume=True
    )

    # After completion, you might want to do some analysis
    if camouflage_df is not None:
        # Count successful analyses
        success_count = (~camouflage_df['camouflage_score'].isna()).sum()
        error_count = (~camouflage_df['analysis_error'].isna()).sum()

        print(f"\nAnalysis summary:")
        print(f"  - Total rows: {len(camouflage_df)}")
        print(f"  - Successful analyses: {success_count} ({success_count / len(camouflage_df) * 100:.1f}%)")
        print(f"  - Failed analyses: {error_count} ({error_count / len(camouflage_df) * 100:.1f}%)")

        # Basic stats by category
        if 'category_name' in camouflage_df.columns:
            print("\nAverage camouflage score by category:")
            category_scores = camouflage_df.groupby('category_name')['camouflage_score'].agg(
                ['mean', 'std', 'count']).reset_index()

            # Only show categories with at least 5 successful analyses
            valid_categories = category_scores[category_scores['count'] >= 5]
            print(valid_categories.sort_values('mean'))