import pandas as pd
import numpy as np


def split_data(df: pd.DataFrame,
               test_ratio: float,
               seed: int) -> tuple[pd.DataFrame]:
    """Diviser le dataset en set de train et test en mélangeant l'ordre des
    données aléatoirement en fixant la random seed.

    Args:
        df (pd.DataFrame): _description_
        test_ratio (float): _description_
        seed (int): _description_

    Returns:
        tuple[pd.DataFrame]: X_train, y_train, X_test, y_test
    """
    pass