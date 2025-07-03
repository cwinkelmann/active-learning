import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_frequency_distribution(df, columns=None, figsize=(15, 10), bins=30, kde=True):
    """
    Create frequency distribution plots for specified columns in a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to visualize
    columns : list or None
        List of column names to visualize. If None, will try to use numeric columns
    figsize : tuple
        Figure size as (width, height) in inches
    bins : int
        Number of bins for the histogram
    kde : bool
        Whether to overlay a kernel density estimate

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    # If no columns specified, use numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    # Calculate numbr of rows and columns for subplot grid
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes array for easier iteration
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Create plots
    for i, column in enumerate(columns):
        if i < len(axes):  # Ensure we don't exceed the number of axes
            sns.histplot(data=df, x=column, kde=kde, bins=bins, ax=axes[i])
            axes[i].set_title(f'Distribution of {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig
