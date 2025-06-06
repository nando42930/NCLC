import itertools
from scipy.stats import spearmanr


def spearman_correlation(numeric_df, only_with=None):
    """
    Computes and prints the Spearman Rank Correlation and p-value for all pairs of numeric features.

    Args:
        numeric_df (pandas.DataFrame): A DataFrame containing only numeric features.
        only_with (str): If set, only runs tests between this column and others.

    Returns:
        None: This function prints correlation statistics and does not return a value.
    """

    # Determines combinations of variables to compare
    if only_with and only_with in numeric_df.columns:
        pairs = [(only_with, other) for other in numeric_df.columns if other != only_with]
    else:
        pairs = itertools.combinations(numeric_df.columns, 2)

    # Computes and prints Spearman correlation results
    for var1, var2 in pairs:
        corr, p = spearmanr(numeric_df[var1], numeric_df[var2])
        print(f"Spearman Rank Correlation between '{var1}' and '{var2}'")
        print(f"Correlation Coefficient (0.8 < c <= 1.0): {corr:.3f}")
        print(f"P-Value (p < 0.05): {p:.4f}\n")
