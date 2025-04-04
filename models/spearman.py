from scipy.stats import spearmanr


def spearman_correlation(df):
    """Computes Spearman’s Rank Correlation between Sample Size and Usability Testing Data."""
    df['Usability Testing Data Numeric'] = df['Usability Testing Data'].map({
        'No': 0,
        'Qualitative Data': 1,
        'Quantitative Data': 2
    })

    spearman_corr, spearman_p = spearmanr(df['Sample Size'], df['Usability Testing Data Numeric'])

    print("Spearman’s Rank Correlation Results:")
    print(f"Correlation Coefficient (0.8 < c <= 1.0): {spearman_corr}")
    print(f"P-Value (p < 0.05): {spearman_p}\n")
