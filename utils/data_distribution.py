import numpy as np
from scipy.stats import shapiro, kstest


def normality_tests(groups, method='shapiro'):
    """
    Performs normality tests (Shapiro-Wilk or Kolmogorov-Smirnov) on each group.

    Args:
        groups (dict-like): A dictionary or groupby-like object where keys are group names and
                                values are arrays or lists of data.
        method (str): The normality test to use, either 'shapiro' or 'ks'. Default is 'shapiro'.

    Returns:
        bool: True if all groups pass the normality test (p >= 0.05), False otherwise.
    """

    normal_flags = []

    for name, values in groups.items():
        values = np.asarray(values)

        # Skips groups with less than 3 observations (insufficient for reliable testing)
        if len(values) < 3:
            print(f"{name}: Not enough data for {method.capitalize()} (n={len(values)})")
            normal_flags.append(False)
            continue

        if method == 'shapiro':
            stat, p = shapiro(values)
        elif method == 'ks':
            # Estimates parameters for normal distribution from the sample
            mean, std = np.mean(values), np.std(values, ddof=1)

            # Cannot perform KS test with zero standard deviation
            if std == 0:
                print(f"{name}: Standard deviation is 0, cannot run KS test.")
                normal_flags.append(False)
                continue

            stat, p = kstest(values, 'norm', args=(mean, std))
        else:
            raise ValueError("Method must be 'shapiro' or 'ks'.")

        # Group is considered normally distributed if p-value ≥ 0.05
        normal_flags.append(p >= 0.05)

    # Returns True only if all groups passed the normality test
    return all(normal_flags)
