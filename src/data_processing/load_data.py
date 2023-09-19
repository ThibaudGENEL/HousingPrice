import pandas as pd
import os

def load_data(filename: str = 'data/house_prices.csv') -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        filename (str, optional): The relative or absolute path to the CSV file. Defaults to 'data/house_prices.csv'.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    # Convert the relative path to an absolute path
    filepath = os.path.abspath(filename)
    
    # Load the data using pd.read_csv
    df = pd.read_csv(filepath)
    
    return df
