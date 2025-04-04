from utils.data_loader import load_data
from models.chi_square import chi_square_test
from models.anova_kruskal import perform_tests
from models.spearman import spearman_correlation
from models.correlation_heatmap import plot_correlation_heatmap

# Load data
df = load_data("./data/data_extraction.csv")

# Perform analyses
chi_square_test(df)
perform_tests(df)
spearman_correlation(df)
plot_correlation_heatmap(df)
