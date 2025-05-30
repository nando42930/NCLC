import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(numeric_df):
    """
    Computes and plots a Pearson Correlation Heatmap from a numeric DataFrame.

    Args:
        numeric_df (pandas.DataFrame): A DataFrame containing only numeric features
                                       for which the correlation heatmap will be plotted.

    Returns:
        None: This function displays the plot and does not return a value.
    """

    correlation_matrix = numeric_df.corr()

    # Rescales correlation values from [-1, 1] to [0, 100] to enhance visual interpretation
    scaled_correlation_matrix = (correlation_matrix + 1) / 2 * 100

    plt.figure(figsize=(10, 10))
    sns.heatmap(scaled_correlation_matrix, annot=True, fmt=".1f", cmap='inferno', linewidths=0.5, square=True,
                cbar_kws={"shrink": .8, 'extend': 'both'}, xticklabels=numeric_df.columns,
                yticklabels=numeric_df.columns)

    plt.title("Pearson Correlation Heatmap", fontsize=20)
    plt.tight_layout()
    plt.show()
