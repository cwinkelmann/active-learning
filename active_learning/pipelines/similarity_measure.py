"""
calculate the similarity between two images using the Structural Similarity Index (SSIM), FSIM,
 and Mean Squared Error (MSE) metrics
 TODO refactor this
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import piq
import seaborn as sns
import torch
import typing
from pathlib import Path
from skimage.io import imread
from tqdm import tqdm


def laplacian_variance(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.var(laplacian)


# %%
def tenengrad(image, ksize=3):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    return np.mean(magnitude)


@torch.no_grad()
def sim_with_ref(img1_path, img2_path):
    print(f"====================== metrics for {img1_path} & {img2_path}======================")
    dict_data = {}
    dict_data["img1_path"] = img1_path
    dict_data["img2_path"] = img2_path
    # Read RGB image and it's noisy version
    x = torch.tensor(imread(img1_path)).permute(2, 0, 1)[None, ...] / 255.
    y = torch.tensor(imread(img2_path)).permute(2, 0, 1)[None, ...] / 255.

    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        x = x.cuda()
        y = y.cuda()

    print("====================== metrics with reference image ======================")
    # Compute SSIM
    ssim_value = piq.ssim(x, y, data_range=1.)

    print(f"SSIM: {ssim_value.item()}")
    dict_data["ssim"] = ssim_value.item()
    # To compute FSIM as a measure, use lower case function from the library
    fsim_index: torch.Tensor = piq.fsim(x, y, data_range=1., reduction='none')
    # In order to use FSIM as a loss function, use corresponding PyTorch module
    fsim_loss = piq.FSIMLoss(data_range=1., reduction='none')(x, y)
    print(f"FSIM index: {fsim_index.item():0.4f}, loss: {fsim_loss.item():0.4f}")
    dict_data["fsim_index"] = fsim_index.item()
    dict_data["fsim_loss"] = fsim_loss.item()

    return dict_data


@torch.no_grad()
def sim_no_ref(img1_path):
    print(f"====================== metrics for {img1_path} ======================")
    # Read RGB image and it's noisy version
    x = torch.tensor(imread(img1_path)).permute(2, 0, 1)[None, ...] / 255.
    image = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    dict_data = {}
    dict_data["img1_path"] = img1_path
    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        x = x.cuda()

    print(f"====================== metrics with no reference image ======================")

    t_sharpness = tenengrad(image)
    print(f"Tenengrad Sharpness: {t_sharpness}")
    dict_data["t_sharpness"] = t_sharpness
    l_sharpness = laplacian_variance(image)
    print(f"Laplacian Variance (Sharpness): {l_sharpness}")
    dict_data["l_sharpness"] = l_sharpness
    # To compute BRISQUE score as a measure, use lower case function from the library
    brisque_index: torch.Tensor = piq.brisque(x, data_range=1., reduction='none')
    # In order to use BRISQUE as a loss function, use corresponding PyTorch module.
    # Note: the back propagation is not available using torch==1.5.0.
    # Update the environment with latest torch and torchvision.
    brisque_loss: torch.Tensor = piq.BRISQUELoss(data_range=1., reduction='none')(x)
    print(f"BRISQUE index: {brisque_index.item():0.4f}, loss: {brisque_loss.item():0.4f}")
    dict_data["brisque_index"] = brisque_index.item()

    # To compute CLIP-IQA score as a measure, use PyTorch module from the library
    clip_iqa_index: torch.Tensor = piq.CLIPIQA(data_range=1.).to(x.device)(x)
    print(f"CLIP-IQA: {clip_iqa_index.item():0.4f}")
    dict_data["clip_iqa_index"] = clip_iqa_index.item()

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # Calculate total and central energy
    total_energy = np.sum(magnitude_spectrum ** 2)

    def central_energy(magnitude_spectrum, radius):
        center_x, center_y = magnitude_spectrum.shape[1] // 2, magnitude_spectrum.shape[0] // 2
        Y, X = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        circular_mask = dist_from_center <= radius
        return np.sum(magnitude_spectrum[circular_mask] ** 2)

    central_energy = central_energy(magnitude_spectrum, radius=30)  # Adjust radius as needed

    # Calculate the sharpness metric
    sharpness_metric = central_energy / total_energy
    print(f"Fourier Sharpness Metric: {sharpness_metric}")
    dict_data["sharpness_metric"] = sharpness_metric

    return dict_data


@torch.no_grad()
def calculate_pairwise_ssim_cm(images_class_a: typing.List[Path],
                               images_class_b: typing.List[Path],
                               class_a: str,
                               class_b: str,
                               output_csv=None,
                               output_plot=None,
                               vmin: float = 0.0,
                               vmax: float = 1.0,
                               cmap="viridis"):
    """
    Calculate the pairwise SSIM between images from two different classes.

    Parameters:
    images_class_a (list): List of Path objects pointing to images of class A
    images_class_b (list): List of Path objects pointing to images of class B
    class_a (str): Name of class A for plot labels
    class_b (str): Name of class B for plot labels
    output_csv (str, optional): Path to save results as CSV. If None, only returns the DataFrame
    output_plot (str, optional): Path to save heatmap visualization. If None, displays plot
    vmin (float, optional): Minimum value for the colormap. Default is 0.
    vmax (float, optional): Maximum value for the colormap. Default is 1.
    cmap (str, optional): Colormap to use for the heatmap. Default is "viridis".

    Returns:
    pandas.DataFrame: DataFrame containing the pairwise SSIM values
    """
    # Create an empty list to store results
    results = []

    # Check for available device: CUDA, MPS, or CPU
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    print(f"Using device: {device}")

    # Calculate SSIM for each pair of images
    total_pairs = len(images_class_a) * len(images_class_b)
    with tqdm(total=total_pairs, desc="Calculating pairwise SSIM") as pbar:
        for class_a_path in images_class_a:
            # Load class A image once per loop
            class_a_img = torch.tensor(imread(class_a_path)).permute(2, 0, 1)[None, ...] / 255.
            class_a_img = class_a_img.to(device)

            for class_b_path in images_class_b:
                # Load class B image
                class_b_img = torch.tensor(imread(class_b_path)).permute(2, 0, 1)[None, ...] / 255.
                class_b_img = class_b_img.to(device)

                # Compute SSIM
                ssim_value = piq.ssim(class_a_img, class_b_img, data_range=1.)

                # Store the result
                results.append({
                    "class_a_image": class_a_path.name,
                    "class_b_image": class_b_path.name,
                    "ssim": ssim_value.item()
                })

                pbar.update(1)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # # Sort by SSIM value (highest first)
    # df = df.sort_values(by="ssim", ascending=False)
    #
    # # Print statistics to help with scale selection
    # print(f"\nSSIM Statistics:")
    # print(f"Min: {df['ssim'].min():.4f}")
    # print(f"Max: {df['ssim'].max():.4f}")
    # print(f"Mean: {df['ssim'].mean():.4f}")
    # print(f"Median: {df['ssim'].median():.4f}")
    # print(f"Std Dev: {df['ssim'].std():.4f}")
    #
    # # Calculate percentiles for reference
    # percentiles = [5, 25, 50, 75, 95]
    # for p in percentiles:
    #     print(f"{p}th percentile: {df['ssim'].quantile(p / 100):.4f}")

    # Save to CSV if output path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    # Create a visualization of the SSIM matrix
    # Pivot the data to create a matrix
    ssim_matrix = df.pivot(index="class_a_image", columns="class_b_image", values="ssim")

    # Create a heatmap visualization
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(
        ssim_matrix,
        cmap=cmap,
        annot=False,  # Too many cells for annotations
        fmt=".3f",
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax
    )

    # Customize the plot
    plt.title(f"Pairwise SSIM Comparison: {class_a} vs {class_b}", fontsize=16)
    plt.xlabel(f"{class_b} Images", fontsize=14)
    plt.ylabel(f"{class_a} Images", fontsize=14)

    # Adjust tick labels for readability
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    # Add a colorbar legend
    cbar = ax.collections[0].colorbar
    cbar.set_label(f"SSIM Value (range: {vmin:.3f}-{vmax:.3f})", fontsize=12)

    # Add a note about the scale
    if vmin != 0 or vmax != 1:
        plt.figtext(0.5, 0.01, f"Note: Color scale adjusted to range [{vmin:.3f}, {vmax:.3f}] for better visualization",
                    ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    # Tight layout to ensure all elements are visible
    plt.tight_layout()

    # Save or display the plot
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_plot}")
    else:
        plt.show()

    return df


@torch.no_grad()
def calculate_pairwise_ssim(images_class_a: typing.List[Path],
                            images_class_b: typing.List[Path],
                            class_a: str,
                            class_b: str,
                            output_csv=None, output_plot=None):
    """
    Calculate the pairwise SSIM between each iguana image and each empty image.

    Parameters:
    iguana_images (list): List of Path objects pointing to iguana images
    empty_images (list): List of Path objects pointing to empty images
    output_csv (str, optional): Path to save results as CSV. If None, only returns the DataFrame
    output_plot (str, optional): Path to save heatmap visualization. If None, displays plot

    Returns:
    pandas.DataFrame: DataFrame containing the pairwise SSIM values
    """
    # Create an empty list to store results
    results = []

    # Check for available device: CUDA, MPS, or CPU
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    print(f"Using device: {device}")

    # Calculate SSIM for each pair of images
    total_pairs = len(images_class_a) * len(images_class_b)
    with tqdm(total=total_pairs, desc="Calculating pairwise SSIM") as pbar:
        for class_a_path in images_class_a:
            # Load iguana image once per loop
            classify_a_img = torch.tensor(imread(class_a_path)).permute(2, 0, 1)[None, ...] / 255.
            classify_a_img = classify_a_img.to(device)

            for class_b_path in images_class_b:
                # Load empty image
                class_b_img = torch.tensor(imread(class_b_path)).permute(2, 0, 1)[None, ...] / 255.
                class_b_img = class_b_img.to(device)

                # Compute SSIM
                ssim_value = piq.ssim(classify_a_img, class_b_img, data_range=1.)

                # Store the result
                results.append({
                    "class_a_image": class_a_path.name,
                    "class_b_image": class_b_path.name,
                    "ssim": ssim_value.item()
                })

                pbar.update(1)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Sort by SSIM value (highest first)
    df = df.sort_values(by="ssim", ascending=False)

    # Save to CSV if output path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    # Create a visualization of the SSIM matrix
    # Pivot the data to create a matrix
    ssim_matrix = df.pivot(index="class_a_image", columns="class_b_image", values="ssim")

    # Create a heatmap visualization
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        ssim_matrix,
        cmap="viridis",
        annot=False,  # Too many cells for annotations
        fmt=".2f",
        linewidths=0.5,
        vmin=0,
        vmax=1
    )

    # Customize the plot
    plt.title(f"Pairwise SSIM Comparison:{class_a} vs {class_b} B", fontsize=14)
    plt.xlabel(f"{class_a} Images", fontsize=12)
    plt.ylabel(f"{class_b} Images", fontsize=12)

    # Adjust tick labels for readability
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    # Add a colorbar legend
    cbar = ax.collections[0].colorbar
    cbar.set_label("SSIM Value", fontsize=12)

    # Tight layout to ensure all elements are visible
    plt.tight_layout()

    # Save or display the plot
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_plot}")
    else:
        plt.show()

    return df


if __name__ == "__main__":
    imgage_path_iguana = image_path_2 = Path(
        "/Users/christian/data/training_data/2025_02_22_HIT/01_segment_pretraining/segments_12/train/classification/iguana/Fer_FCD01-02-03_20122021_single_images___DJI_0366_x1344_y448.jpg")
    image_path_empty_1 = Path(
        '/Users/christian/data/training_data/2025_02_22_HIT/01_segment_pretraining/segments_12/train/classification/empty/San_STJB06_12012023___STJB06_12012023_Santiago_m_2_7_DJI_0128_x2240_y2688.jpg')

    image_root_dir = Path(
        "/Users/christian/data/training_data/2025_02_22_HIT/03_all_other/val/classification")
    image_cats_and_dogs_root_dir = Path("/Users/christian/Downloads/kagglecatsanddogs_5340/PetImages224")

    class_empty = "empty"
    class_iguana = "iguana_point"
    class_dog = "Dog"
    class_cat = "Cat"

    iguana_images = [i for i in image_root_dir.joinpath(class_iguana).glob("*.jpg") if not str(i.name).startswith(".")]
    cat_images = [i for i in image_cats_and_dogs_root_dir.joinpath(class_cat).glob("*.jpg") if
                  not str(i.name).startswith(".")]
    dog_images = [i for i in image_cats_and_dogs_root_dir.joinpath(class_dog).glob("*.jpg") if
                  not str(i.name).startswith(".")]
    empty_images = [i for i in image_root_dir.joinpath(class_empty).glob("*.jpg") if not str(i.name).startswith(".")]

    threshold = 20
    iguana_images = iguana_images[:threshold]
    dog_images = dog_images[:threshold]
    cat_images = cat_images[:threshold]
    empty_images = empty_images[:threshold]

    # Calculate pairwise SSIM and save results to CSV
    results_df = calculate_pairwise_ssim_cm(
        images_class_a=iguana_images,
        images_class_b=empty_images,
        class_a=class_iguana,
        class_b=class_dog,
        output_csv="pairwise_ssim_results.csv",
        vmin=0.0,
        vmax=0.3,
    )
    # Basic average of all SSIM values
    average_ssim = results_df['ssim'].mean()
    print(f"Average SSIM {class_iguana} vs {class_empty}: {average_ssim:.4f}")

    # Calculate pairwise SSIM and save results to CSV
    results_df = calculate_pairwise_ssim_cm(
        images_class_a=dog_images,
        images_class_b=cat_images,
        class_a=class_dog,
        class_b=class_cat,
        output_csv="pairwise_ssim_results.csv",
        vmin=0.0,
        vmax=0.3,
    )
    # Basic average of all SSIM values
    average_ssim = results_df['ssim'].mean()
    print(f"Average SSIM {class_dog} vs {class_cat}: {average_ssim:.4f}")

    # Calculate pairwise SSIM and save results to CSV
    results_df = calculate_pairwise_ssim_cm(
        images_class_a=cat_images,
        images_class_b=cat_images,
        class_a=class_dog,
        class_b=class_dog,
        output_csv="pairwise_ssim_results.csv",
        vmin=0.0,
        vmax=0.3,
    )
