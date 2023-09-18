import pandas as pd


def load_data(filename: str) -> pd.DataFrame:
    """Load a CSV file and return its data as a DataFrame.

    Args:
        filename (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.

    Example:
        To load a CSV file named 'data.csv,' you can use this function as follows:
        
        ">>> data = load_data('data.csv')
    """
    return pd.read_csv(filename)