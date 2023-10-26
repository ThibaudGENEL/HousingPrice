import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


import sys
sys.path.append(r"C:\Users\thiba\OneDrive\Documents\GGM Python Proj\projet-python-gaffiero-genel-manzala")
from src.data_science.data import split_data
from config.config import TEST_RATIO, SEED


#Loading data
data = pd.read_pickle("df_ready.pkl")

#Function baseline
def baseline_model(data, variable, train_sizes = [10, 50, 100, 250, 500, 700], seed = SEED, plot_mse = True, add_column=True):
    """
    Creates a baseline model using the average price for each unique value of the 
    given variable. 
    
    It evaluates the model's performance (MAE) across various training sizes and 
    visualizes the results. Optionally, it can append predictions to the input data.
    
    Parameters:
    - data (pd.DataFrame): Input dataframe containing at least the specified variable and "Price".
    - variable (str): Variable to base the baseline model on.
    - train_sizes (list of int, optional): Desired training data sizes. Defaults to [10, 50, 100, 250, 500, 700].
    - add_column (bool, optional): If True, adds predictions as a new column to input data. Defaults to True.

    Returns:
    None. The function prints MAE values and displays a plot.
    
    Example:
    >>> df = pd.DataFrame({'Bedrooms': [1,2,2,3,3,3], 'Price': [100,200,210,310,300,290]})
    >>> baseline_model(df, "Bedrooms", [10, 20, 50, 400, 600], False)
    """

    mae_errors = []
    for train_size in train_sizes:
        # Split
        test_ratio = 1 - train_size / data.shape[0]   # test_ratio based on asked train_size
        X_train, y_train, X_test, y_test = split_data(data, test_ratio, seed)
        data_train = X_train.copy()
        data_train["Price"] = y_train     # Adding Price into training set, we need it to compute the mean
        data_test = X_test.copy()
        data_test["Price"] = y_test
        # So we have a training set and a test set, all columns in both

        #Model
        variable_price_dict = data_train.groupby(variable)["Price"].mean().to_dict()  #{numb of bedrooms : mean price}
        data_test[f"Price_Baseline_{variable}"] = data_test[variable].apply(lambda x: variable_price_dict.get(x, 0))     # Attribution in test set

        # Compute MAE and append to mae_errors
        mae =  np.abs(data_test["Price"] - data_test[f"Price_Baseline_{variable}"]).mean()
        mae_errors.append(mae)
        errors = pd.DataFrame({"train_size": pd.Series(train_sizes), "mae": pd.Series(mae_errors)})

    if plot_mse:
        # Plotting the results
        plt.plot(train_sizes, mae_errors, '-o')
        plt.xlabel('Training Size')
        plt.ylabel('Mean Squared Error (MAE)')
        plt.title(f'Performance of Baseline {variable} Model by Training Size')
        plt.grid(True)
        plt.show()


    # Creating a prediction column in initial data. All data involved, no train/test
    if add_column:
        data[f"Price_Baseline_{variable}"] = data[variable].apply(lambda x: data.loc[data[variable] == x, "Price"].mean())

    return errors

# baseline_model(data= data, variable= "Bedrooms")

def baseline_model_n_times(n, data, variable, train_sizes = [10, 50, 100, 250, 500, 700]):
    all_errors = []
    np.random.seed(SEED)
    for i in range(n):
        errors = baseline_model(data, variable, train_sizes, seed=np.random.randint(1, 100), plot_mse=False, add_column=False)
        all_errors.append(errors)
    
    errors_mixture = pd.concat(all_errors, ignore_index=True)
    grouped_by_size_mean_mae = errors_mixture.groupby("train_size")["mae"].mean()

    # Plotting the results
    plt.rcParams["font.family"] = "Garamond"
    plt.figure(figsize=(12, 8))
    plt.plot(grouped_by_size_mean_mae, '-o')
    plt.xlabel('Training Size')
    plt.ylabel('Mean Squared Error (MAE)')
    plt.title(f'Mean performance of Baseline {variable} Model by Training Size', fontsize=16, weight="bold")
    plt.suptitle(f"\nMean of MAE over {n} iterations of the model with random (thus different) train/test samples", y=0.9, fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return grouped_by_size_mean_mae


baseline_model_n_times(100, data, "Bedrooms", train_sizes= [10, 50, 100, 250, 500, 700, 780])