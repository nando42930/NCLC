import pandas as pd


def load_data(*file_paths):
    """
    Loads multiple datasets from the given file paths.

    Args:
        *file_paths (str): One or more file paths to CSV files to be loaded.

    Returns:
        list of pandas.DataFrame: A list of DataFrames, each corresponding to a loaded CSV file.
    """

    dataframes = []

    for path in file_paths:
        try:
            df = pd.read_csv(path, delimiter=';')
            dataframes.append(df)
        except Exception as e:
            raise ValueError(f"Error loading {path}: {e}")

    return dataframes
