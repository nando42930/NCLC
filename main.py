from utils.data_loader import load_data
from utils.feature_encoder import encode_categorical_features, encode_numeric_features
from models.correlation_heatmap import plot_correlation_heatmap
from models.spearman import spearman_correlation
from models.chi_square import chi_square_test
from models.anova_kruskal import anova_kruskal_test
from models.model_analyses import analyze_relationship


""" DATA LOADER """
df_main, df_imp_language = load_data(
    "./data/data_extraction.csv",
    "./data/data_extraction_IMPLANGUAGE.csv"
)

# REMOVES 'article' AND 'Implementation Language' FEATURES
df_main = df_main.iloc[:, 2:].copy()

# REMOVES 'Id_PT2_Articles' AND 'article' FEATURES
df_imp_language = df_imp_language.iloc[:, 2:].copy()


""" PEARSON CORRELATION HEATMAP (NUMERICAL x NUMERICAL) """
# plot_correlation_heatmap(df_main, df_imp_language)


""" SPEARMAN RANK CORRELATION (NUMERICAL x NUMERICAL) """
# numeric_df = encode_categorical_features(df_main)                 # MODIFIES CATEGORICAL FEATURES TO NUMERIC
# spearman_correlation(numeric_df)

# imp_numeric_df = encode_categorical_features(df_imp_language)     # MODIFIES CATEGORICAL FEATURES TO NUMERIC
# spearman_correlation(imp_numeric_df, only_with='Implementation Language')


""" CHI-SQUARE ANALYSIS (CATEGORICAL x CATEGORICAL) """
# categorical_df = encode_numeric_features(df_main)                 # MODIFIES NUMERIC FEATURES TO CATEGORICAL
# chi_square_test(categorical_df)

# imp_categorical_df = encode_numeric_features(df_imp_language)     # MODIFIES NUMERIC FEATURES TO CATEGORICAL
# chi_square_test(imp_categorical_df, only_with='Implementation Language')


""" ANOVA OR KRUSKAL-WALLIS ANALYSES (CATEGORICAL x NUMERICAL) """
# anova_kruskal_test(df_main, normality_method='shapiro')           # USES SHAPIRO-WILK
# anova_kruskal_test(df_main, normality_method='ks')                # USES KOLMOGOROV-SMIRNOV

# USES SHAPIRO-WILK
# anova_kruskal_test(df_imp_language, normality_method='shapiro', only_with='Implementation Language')
# USES KOLMOGOROV-SMIRNOV
# anova_kruskal_test(df_imp_language, normality_method='ks', only_with='Implementation Language')


""" RANDOM FOREST REGRESSOR OR CLASSIFIER ANALYSES """
# analyze_relationship(df_main, r2_threshold=0.7, accuracy_threshold=0.94)

# analyze_relationship(df_imp_language, r2_threshold=0.7, accuracy_threshold=0.94, only_with='Implementation Language')
