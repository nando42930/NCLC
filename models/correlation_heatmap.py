from utils.feature_encoder import encode_categorical_features
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(df_main, df_imp_language):
    """
    Computes and plots a Pearson Correlation Heatmap combining the main dataset
    and 'Implementation Language' from a secondary dataset.

    Args:
        df_main (pandas.DataFrame): Main dataset (excluding 'article' and 'Implementation Language').
        df_imp_language (pandas.DataFrame): Dataset containing the 'Implementation Language' feature.

    Returns:
        None: This function displays the plot and does not return a value.
    """

    # Encodes categorical features into numeric format
    numeric_main = encode_categorical_features(df_main)
    numeric_implang = encode_categorical_features(df_imp_language)

    # Computes both correlation matrices
    corr_main = numeric_main.corr()
    corr_implang = numeric_implang.corr()

    # Extracts 'Implementation Language' correlations only
    implang_corr = corr_implang['Implementation Language'].drop('Implementation Language')

    # Appends 'Implementation Language' correlations to the main correlation matrix as a new column
    corr_combined = corr_main.copy()
    corr_combined['Implementation Language'] = implang_corr

    # Appends 'Implementation Language' correlations to the main correlation matrix as a new row
    implang_row = implang_corr.copy()
    implang_row['Implementation Language'] = 1.0
    corr_combined.loc['Implementation Language'] = implang_row

    # Rescales correlation values from [-1, 1] to [0, 100] to enhance visual interpretation
    scaled_correlation_matrix = (corr_combined + 1) / 2 * 100

    plt.figure(figsize=(10, 10))
    sns.heatmap(scaled_correlation_matrix, annot=True, fmt=".1f", cmap='inferno', linewidths=0.5, square=True,
                cbar_kws={"orientation": "horizontal", "shrink": .8, 'extend': 'both'},
                xticklabels=corr_combined.columns, yticklabels=corr_combined.columns)

    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=45, va='top', fontsize=9)

    plt.title("Pearson Correlation Heatmap", fontsize=20)
    plt.tight_layout()
    plt.show()
