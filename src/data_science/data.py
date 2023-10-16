import pandas as pd
import numpy as np


def split_data(df: pd.DataFrame,
               test_ratio: float,
               seed: int) -> tuple[pd.DataFrame]:
    """Divide the dataset into train and test sets by mixing the order of
    data randomly by setting the random seed.

    Args:
        df (pd.DataFrame): Labeled dataframe
        test_ratio (float): The ratio between 0 and 1 for the proportion of examples in the test set.
        seed (int): The seed value for randomization.

    Returns:
        tuple[pd.DataFrame]: X_train, y_train, X_test, y_test
    """



    len_df = len(df)

    np.random.seed(seed)
    
    index = np.random.permutation(df.index)
    split_value= int(len_df * (1-test_ratio))

    train_df, test_df = df.loc[index[:split_value]], df.loc[index[split_value:]]

    X_train = train_df.drop('Price', axis=1)
    y_train = train_df['Price']
    X_test = test_df.drop('Price', axis=1)
    y_test = test_df['Price']

    return X_train, y_train, X_test, y_test





