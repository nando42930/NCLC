from utils.data_distribution import normality_tests
from scipy.stats import f_oneway, kruskal


def anova_kruskal_test(df, normality_method='shapiro'):
    """
    Performs statistical tests (ANOVA or Kruskal-Wallis) on all combinations of 
    categorical and numerical column pairs in the DataFrame.

    For each categorical column and numerical column pair:
        - If the numerical values grouped by the categorical column are normally distributed,
          it uses ANOVA.
        - Otherwise, it uses the Kruskal-Wallis test.

    Args:
        df (pandas.DataFrame): The input DataFrame containing both categorical and numerical features.
        normality_method (str): The method for normality testing. Options are 'shapiro' or 'ks'.
                                Default is 'shapiro'.

    Returns:
        None: This function prints the results of each statistical test and does not return a value.
    """

    numerical_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            # Skips if the categorical column has too few groups or too many,
            # avoiding meaningless or overly fragmented tests
            unique_groups = df[cat_col].dropna().unique()
            if len(unique_groups) < 2 or len(unique_groups) > 20:
                continue

            # Builds grouped data
            groups = {
                group: df[df[cat_col] == group][num_col].dropna()
                for group in unique_groups
            }

            # Skips if any group has less than 2 observations
            if any(len(values) < 2 for values in groups.values()):
                continue

            # If p ≥ 0.05, the data is normally distributed (use ANOVA).
            # If p < 0.05, the data is not normally distributed (use Kruskal-Wallis).
            is_normal = normality_tests(groups, method=normality_method)

            if is_normal:
                stat, p = f_oneway(*groups.values())
                test_type = "ANOVA"
            else:
                stat, p = kruskal(*groups.values())
                test_type = "Kruskal-Wallis"

            print(f"{test_type} Test between '{num_col}' and '{cat_col}':")
            print(f"{test_type} Statistic: {stat}, P-Value (p < 0.05): {p}")
