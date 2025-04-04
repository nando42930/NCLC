import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """Creates and displays a Pearson Correlation Heatmap."""
    numeric_df = pd.DataFrame()
    numeric_df['Sample Size'] = df['Sample Size']
    numeric_df['GUI Image'] = df['GUI Image'].apply(lambda x: 1 if x == 'Yes' else 0)
    numeric_df['GUI Model'] = df['GUI Model'].apply(lambda x: 1 if x == 'Yes' else 0)
    numeric_df['Usability Testing Data'] = df['Usability Testing Data'].apply(
        lambda x: 2 if x == 'Quantitative Data' else (1 if x == 'Qualitative Data' else 0))
    numeric_df['Comparison Table Data'] = df['Comparison Table Data'].apply(lambda x: 1 if x == 'Yes' else 0)
    numeric_df['Target System'] = df['Target System'].apply(
        lambda x: 3 if x == 'Information System' else (2 if x == 'ML System' else (1 if x == 'IoT System' else 0)))
    numeric_df['Contribution Focus'] = df['Contribution Focus'].apply(lambda x: 1 if x == 'Visual' else 0)

    correlation_matrix = numeric_df.corr()
    scaled_correlation_matrix = (correlation_matrix + 1) / 2 * 100

    plt.figure(figsize=(10, 10))
    sns.heatmap(scaled_correlation_matrix, annot=True, fmt=".1f", cmap='inferno', linewidths=0.5, square=True,
                cbar_kws={"shrink": .8, 'extend': 'both'}, xticklabels=numeric_df.columns,
                yticklabels=numeric_df.columns)

    plt.title("Pearson Correlation Heatmap", fontsize=20)
    plt.tight_layout()
    plt.show()
