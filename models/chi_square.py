import pandas as pd
from scipy.stats import chi2_contingency


def chi_square_test(df):
    """Performs a Chi-Square test on GUI Model vs. Usability Testing Data."""
    contingency_table = pd.crosstab(df['GUI Model'], df['Usability Testing Data'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("Chi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2}")
    print(f"P-Value (p < 0.05): {p}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies: {expected}\n")
