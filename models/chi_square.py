from itertools import combinations
import pandas as pd
from scipy.stats import chi2_contingency


def chi_square_test(df, only_with=None):
    """
    Performs Chi-Square tests on combinations of categorical columns in the DataFrame.

    Args:
        df (DataFrame): Input data with categorical features.
        only_with (str): If set, only runs tests between this column and others.

    Returns:
            None: This function prints the results of each Chi-Square test and does not return a value.
    """

    # Ensures this DataFrame contains only categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # If 'only_with' is given, it pairs that column with each other categorical column
    # Otherwise, generates all unique pairs of categorical columns
    if only_with and only_with in categorical_cols:
        test_pairs = [(only_with, col) for col in categorical_cols if col != only_with]
    else:
        test_pairs = list(combinations(categorical_cols, 2))

    # Performs Chi-Square tests
    for col1, col2 in test_pairs:
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        print(f"\nChi-Square Test between '{col1}' and '{col2}'")
        print(f"Chi-Square Statistic: {chi2}, P-Value (p < 0.05): {p}")
