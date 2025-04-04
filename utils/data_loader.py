import pandas as pd


def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pd.read_csv(file_path, delimiter=';')
