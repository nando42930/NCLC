from scipy.stats import f_oneway, kruskal, shapiro


def normality_tests(data_groups):
    """Performs Shapiro-Wilk normality tests on each group."""
    print("Shapiro-Wilk Test p-values:")
    for name, group in data_groups.items():
        stat, p = shapiro(group)
        print(f"{name}: {p}")

    return all(p >= 0.05 for _, p in [shapiro(group) for group in data_groups.values()])


def perform_tests(df):
    """Runs ANOVA if normal, else Kruskal-Wallis."""
    data_groups = {
        'Quantitative Data': df[df['Usability Testing Data'] == 'Quantitative Data']['Sample Size'],
        'Qualitative Data': df[df['Usability Testing Data'] == 'Qualitative Data']['Sample Size'],
        'No Usability Data': df[df['Usability Testing Data'] == 'No']['Sample Size']
    }

    is_normal = normality_tests(data_groups)

    # If p ≥ 0.05, the data is normally distributed (use ANOVA).
    # If p < 0.05, the data is not normally distributed (use Kruskal-Wallis).
    if is_normal:
        test_type = "ANOVA"
        stat, p_val = f_oneway(*data_groups.values())
    else:
        test_type = "Kruskal-Wallis"
        stat, p_val = kruskal(*data_groups.values())

    print(f"{test_type} Test Results:")
    print(f"P-Value (p < 0.05): {p_val}\n")
