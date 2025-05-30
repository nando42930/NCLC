import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(df):
    """
    Converts categorical features in the given DataFrame to numeric format.

    Args:
        df (pandas.DataFrame): The input DataFrame containing categorical features.

    Returns:
        pandas.DataFrame: A DataFrame with numerical features.
    """

    numeric_df = pd.DataFrame()
    numeric_df['Sample Size'] = df['Sample Size']
    numeric_df['GUI Image'] = df['GUI Image'].map({'Yes': 1, 'No': 0})
    numeric_df['GUI Model'] = df['GUI Model'].map({'Yes': 1, 'No': 0})
    numeric_df['Usability Testing Data'] = df['Usability Testing Data'].map({
        'Quantitative Data': 2,
        'Qualitative Data': 1,
        'No': 0
    })
    numeric_df['Comparison Table Data'] = df['Comparison Table Data'].map({'Yes': 1, 'No': 0})
    numeric_df['Target System'] = df['Target System'].map({
        'Information System': 3,
        'ML System': 2,
        'IoT System': 1,
        'VR/AR System': 0
    })
    numeric_df['Contribution Focus'] = df['Contribution Focus'].map({'Visual': 1, 'Multimodal': 0})
    numeric_df['Year'] = df['Year']

    if 'Implementation Language' in df.columns:
        le = LabelEncoder()
        numeric_df['Implementation Language'] = le.fit_transform(df['Implementation Language'].astype(str))

    return numeric_df


def encode_numeric_features(df):
    """
    Converts numeric features in the given DataFrame to categorical format.

    Args:
        df (pandas.DataFrame): The input DataFrame containing numeric features.

    Returns:
        pandas.DataFrame: A DataFrame with categorical features.
    """

    categorical_df = pd.DataFrame()
    categorical_df['Sample Size'] = df['Sample Size'].map(lambda x: 'Zero' if x == 0 else 'Non-zero')
    categorical_df['GUI Image'] = df['GUI Image']
    categorical_df['GUI Model'] = df['GUI Model']
    categorical_df['Usability Testing Data'] = df['Usability Testing Data']
    categorical_df['Comparison Table Data'] = df['Comparison Table Data']
    categorical_df['Target System'] = df['Target System']
    categorical_df['Contribution Focus'] = df['Contribution Focus']
    categorical_df['Year'] = df['Year'].astype(str)

    if 'Implementation Language' in df.columns:
        categorical_df['Implementation Language'] = df['Implementation Language']

    return categorical_df
