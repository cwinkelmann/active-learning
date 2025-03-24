import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image


def get_tsn_embeddings(
        embedding_file: Union[str, Path],
        perplexity: int = 30,
        random_state: int = 42,
        title: str = "t-SNE Visualization of Image Embeddings",
        figsize: Tuple[int, int] = (10, 7),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
        palette: str = "Set1",
        alpha: float = 0.8,
        n_components = 2
) -> pd.DataFrame:
    """
    Load embeddings from a CSV file, apply t-SNE dimensionality reduction, and visualize.

    Args:
        embedding_file: Path to CSV file containing embeddings with class labels
        perplexity: Perplexity parameter for t-SNE (affects clustering)
        random_state: Random seed for reproducibility
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        palette: Color palette for visualization
        alpha: Transparency of points
        return_df: Whether to return the DataFrame with 2D embeddings

    Returns:
        DataFrame with 2D embeddings and class labels if return_df is True
    """
    # Convert to Path if string
    embedding_file = Path(embedding_file) if isinstance(embedding_file, str) else embedding_file

    # Load embeddings from CSV
    df = pd.read_csv(embedding_file, index_col=0)



    # Extract feature vectors and labels
    embeddings = df.iloc[:, :-1].values  # All columns except last (features only)
    class_labels = df["class"].values  # Extract class labels

    # TODO where does the variance come from?
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(embeddings)
    print(pca.explained_variance_ratio_)

    # Get the principal components (eigenvectors)
    components = pca.components_

    # The first row of components corresponds to the first principal component
    # (the one with highest variance)
    first_pc = components[0]

    # Find which original dimension has the largest absolute contribution to the first PC
    contributions = np.abs(first_pc)
    most_important_dimension = np.argmax(contributions)


    # Normalize embeddings
    embeddings = StandardScaler().fit_transform(embeddings)

    # Apply t-SNE
    print(f"Applying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state,  max_iter=2000,              # Number of iterations
    n_iter_without_progress=300,  # Early stopping patience
    min_grad_norm=1e-7,       # Convergence criterion
    metric='euclidean',       # Distance metric
    init='pca',               )
    embeddings_2d = tsne.fit_transform(embeddings)

    # import umap
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    # embeddings_2d = reducer.fit_transform(embeddings)

    # Convert to DataFrame for visualization
    df_tsne = pd.DataFrame(embeddings_2d, columns=["X", "Y"])
    df_tsne["class"] = class_labels  # Append class labels
    df_tsne["file_name"] = df.index.values
    # Plot using Seaborn
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=df_tsne,
        x="X",
        y="Y",
        hue="class",
        palette=palette,
        alpha=alpha,
        edgecolor="k"
    )

    # Improve the plot
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)

    # Move legend outside if many classes
    if len(df_tsne["class"].unique()) > 5:
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.legend(title="Class")

    # Add count of samples per class to the plot
    class_counts = df_tsne["class"].value_counts()
    legend_text = "\n".join([f"{cls}: {count} samples" for cls, count in class_counts.items()])
    plt.annotate(
        legend_text,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )

    # Save if path provided
    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved visualization to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return df_tsne


def visualize_embedding_clusters(
        embedding_file: Union[str, Path],
        n_clusters: int = 5,
        figsize: Tuple[int, int] = (16, 8),
        save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create a side-by-side comparison of embeddings colored by class and by KMeans clusters.

    Args:
        embedding_file: Path to CSV file containing embeddings with class labels
        n_clusters: Number of clusters for KMeans
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    from sklearn.cluster import KMeans

    # Get embeddings
    df_tsne = get_tsn_embeddings(embedding_file, show_plot=False, return_df=True)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_tsne[["X", "Y"]])
    df_tsne["cluster"] = clusters

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot by class
    sns.scatterplot(
        data=df_tsne, x="X", y="Y", hue="class",
        palette="Set1", alpha=0.8, edgecolor="k", ax=ax1
    )
    ax1.set_title("t-SNE by Class")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")

    # Plot by cluster
    sns.scatterplot(
        data=df_tsne, x="X", y="Y", hue="cluster",
        palette="viridis", alpha=0.8, edgecolor="k", ax=ax2
    )
    ax2.set_title(f"t-SNE with KMeans (k={n_clusters})")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved cluster comparison to {save_path}")

    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from matplotlib.lines import Line2D


def perform_clustering(
        df_tsne_embeddings: pd.DataFrame,
        n_clusters: int = 5,
        random_state: int = 42,
) -> pd.DataFrame:
    """
    Load embeddings and perform K-means clustering, returning the enriched dataframe.

    Args:
        embedding_file: Path to CSV file containing embeddings with class labels
        n_clusters: Number of clusters for KMeans
        random_state: Random seed for reproducibility
        load_existing_tsne: If True, assumes the embedding file already contains X and Y columns

    Returns:
        DataFrame with original data plus cluster assignments and prediction information
    """
    # Apply KMeans clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(df_tsne_embeddings[["X", "Y"]])
    df_tsne_embeddings["cluster"] = clusters

    # Create cluster to class mapping (assign each cluster to its most common class)
    cluster_to_class = {}
    for cluster in range(n_clusters):
        cluster_data = df_tsne_embeddings[df_tsne_embeddings["cluster"] == cluster]
        if not cluster_data.empty:
            most_common_class = cluster_data["class"].value_counts().idxmax()
            cluster_to_class[cluster] = most_common_class

    # Add "predicted_class" column based on cluster mapping
    df_tsne_embeddings["predicted_class"] = df_tsne_embeddings["cluster"].map(cluster_to_class)

    # Add "is_correct" column (True if predicted class matches actual class)
    df_tsne_embeddings["is_correct"] = df_tsne_embeddings["predicted_class"] == df_tsne_embeddings["class"]

    # Calculate cluster confidence
    cluster_confidence = {}
    for cluster in df_tsne_embeddings["cluster"].unique():
        cluster_data = df_tsne_embeddings[df_tsne_embeddings["cluster"] == cluster]
        predicted_class = cluster_to_class.get(cluster, "unknown")
        if not cluster_data.empty:
            confidence = (cluster_data["class"] == predicted_class).mean()
            cluster_confidence[cluster] = confidence

    # Add confidence score
    df_tsne_embeddings["cluster_confidence"] = df_tsne_embeddings["cluster"].map(cluster_confidence)

    return df_tsne_embeddings


def visualize_clusters(
        df_tsne: pd.DataFrame,
        figsize: Tuple[int, int] = (21, 7),
        save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Visualize the clustering results in a three-panel plot.

    Args:
        df_tsne: DataFrame containing t-SNE coordinates and cluster assignments
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    # Create three-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Get number of clusters for the title
    n_clusters = len(df_tsne["cluster"].unique())

    # Plot 1: by actual class
    sns.scatterplot(
        data=df_tsne, x="X", y="Y", hue="class",
        palette="Set1", alpha=0.8, edgecolor="k", ax=ax1
    )
    ax1.set_title("t-SNE by True Class")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")

    # Plot 2: by cluster
    sns.scatterplot(
        data=df_tsne, x="X", y="Y", hue="cluster",
        palette="viridis", alpha=0.8, edgecolor="k", ax=ax2
    )
    ax2.set_title(f"t-SNE with KMeans (k={n_clusters})")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")

    # Plot 3: by classification correctness
    sns.scatterplot(
        data=df_tsne, x="X", y="Y",
        hue="is_correct", style="class",
        palette={True: "green", False: "red"},
        alpha=0.8, edgecolor="k", ax=ax3
    )
    ax3.set_title("Classification Accuracy")
    ax3.set_xlabel("t-SNE Dimension 1")
    ax3.set_ylabel("t-SNE Dimension 2")

    # Calculate and display metrics
    accuracy = (df_tsne["is_correct"]).mean()

    # Create a confusion matrix per class
    classes = sorted(df_tsne["class"].unique())
    metrics_text = [f"Overall Accuracy: {accuracy:.2f}"]

    # Calculate precision and recall for each class
    for cls in classes:
        true_pos = ((df_tsne["class"] == cls) & (df_tsne["predicted_class"] == cls)).sum()
        false_pos = ((df_tsne["class"] != cls) & (df_tsne["predicted_class"] == cls)).sum()
        false_neg = ((df_tsne["class"] == cls) & (df_tsne["predicted_class"] != cls)).sum()

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_text.append(f"{cls}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

    # Add metrics to the third plot
    ax3.annotate(
        "\n".join(metrics_text),
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )

    # Create a custom legend for the third plot explaining colors
    custom_lines = [
        Line2D([0], [0], color="green", marker="o", linestyle="None", markersize=8),
        Line2D([0], [0], color="red", marker="o", linestyle="None", markersize=8)
    ]
    ax3.legend(custom_lines, ["Correct Classification", "Incorrect Classification"],
               title="Prediction Accuracy", loc="upper right")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved cluster comparison to {save_path}")

    plt.show()


def get_misclassified_items(
        df_tsne: pd.DataFrame,
        sort_by: str = "cluster_confidence",
        ascending: bool = True
) -> pd.DataFrame:
    """
    Get a DataFrame of misclassified items with additional metadata.

    Args:
        df_tsne: DataFrame with clustering results
        sort_by: Column to sort by (default: cluster_confidence)
        ascending: Sort order (default: True, which shows most uncertain first)

    Returns:
        DataFrame containing only misclassified items
    """
    # Get only the misclassified items
    misclassified = df_tsne[~df_tsne["is_correct"]].copy()

    # Rename for clarity
    misclassified = misclassified.rename(columns={"class": "true_class"})

    # Select and order columns
    result = misclassified[[
        "true_class",
        "predicted_class",
        "cluster",
        "cluster_confidence",
        "X",
        "Y",
        "file_name"
    ]]

    # Sort as requested
    if sort_by in result.columns:
        result = result.sort_values(by=sort_by, ascending=ascending)

    return result


def display_misclassified_grid(
        misclassified_df: pd.DataFrame,
        image_dir: Union[str, Path],
        cols: int = 5,
        rows: int = 1,
        figsize: tuple = (15, 3),
        save_path: Optional[Union[str, Path]] = None,
        true_class_column: str = "true_class"
) -> None:
    """
    Display misclassified images in a simple grid.

    Args:
        misclassified_df: DataFrame with misclassified images data
        image_dir: Base directory containing the images
        cols: Number of columns in the grid
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    # Get file names
    file_names = misclassified_df["file_name"].tolist()
    class_names = misclassified_df[true_class_column].tolist()

    # Calculate grid dimensions
    n_images = len(file_names)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array for single row

    # Flatten axes for easy indexing
    axes_flat = axes.flatten()

    # Display images
    for i, (ax, fname, class_name) in enumerate(zip(axes_flat, file_names, class_names)):
        if i < rows * cols:
            try:
                # Construct full path
                img_path = image_dir / class_name / fname

                # Load and display image
                img = Image.open(img_path)
                ax.imshow(img)

                # Get classification info
                true_class = misclassified_df.iloc[i][true_class_column]
                pred_class = misclassified_df.iloc[i]["predicted_class"]

                # Set title with classification info
                ax.set_title(f"True: {true_class}\nPred: {pred_class}", fontsize=8)

                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}",
                        horizontalalignment='center', verticalalignment='center')
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # Hide unused axes
            ax.axis('off')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved grid to {save_path}")

    plt.show()

