import itertools
from scipy.stats import spearmanr


def spearman_correlation(numeric_df):
    """
    Computes and prints the Spearman Rank Correlation and p-value for all pairs of numeric features.

    Args:
        numeric_df (pandas.DataFrame): A DataFrame containing only numeric features.

    Returns:
        None: This function prints correlation statistics and does not return a value.
    """

    for var1, var2 in itertools.combinations(numeric_df.columns, 2):
        corr, p = spearmanr(numeric_df[var1], numeric_df[var2])
        print(f"Spearman Rank Correlation between '{var1}' and '{var2}'")
        print(f"Correlation Coefficient (0.8 < c <= 1.0): {corr:.3f}")
        print(f"P-Value (p < 0.05): {p:.4f}\n")
