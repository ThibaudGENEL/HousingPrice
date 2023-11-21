import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from src.data_science.data import split_data
from config.config import TEST_RATIO, SEED


#Loading data
df = pd.read_pickle("data/df_ready.pkl")




def linear_model(data,seed=SEED,train_sizes = [10, 50, 100, 250, 500, 700],plot_mae=True,plot_r2=True,add_column=True):
    mae_errors = []
    r2_errors=[]
    predictions= []
    y_real=[]
    for train_size in train_sizes :
        test_ratio = 1 - train_size / data.shape[0] 
        X_train, y_train, X_test, y_test = split_data(data, test_ratio, seed)

        linear_models = LinearRegression()

        linear_models.fit(X_train, y_train)

        prediction = linear_models.predict(X_test)

        predictions.append(prediction)
        y_real.append(y_test)
        mae = mean_absolute_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)

        mae_errors.append(mae)
        r2_errors.append(r2)

    if plot_mae:
        # Plotting the results
        plt.plot(train_sizes, mae_errors, '-o')
        plt.xlabel('Training Size')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(f'Performance linear Model by Training Size')
        plt.grid(True)
        plt.show()

    if plot_r2:
        plt.plot(train_sizes, r2_errors, '-o')
        plt.xlabel('Training Size')
        plt.ylabel('R2')
        plt.title(f'Performance linear Model by Training Size')
        plt.grid(True)
        plt.show()

    if add_column:
        
        data["Price_Linear_Regression"] = linear_models.predict(data.drop('Price', axis=1))

    errors = pd.DataFrame({"train_size": pd.Series(train_sizes), "mae": pd.Series(mae_errors),"R2": pd.Series(r2_errors)})

    return errors,predictions,y_real

def linear_model_n_times(n, data):
    all_errors = []

    np.random.seed(SEED)
    for i in range(n):
        errors = linear_model(data,seed=np.random.randint(1, 100), plot_mae=False, plot_r2=False,add_column=False)
        all_errors.append(errors[0])

    
    errors_mixture = pd.concat(all_errors, ignore_index=True)
    grouped_by_size_mean_mae = errors_mixture.groupby("train_size")["mae"].mean()
    grouped_by_size_mean_R2 = errors_mixture.groupby("train_size")["R2"].mean()

    # Plotting the results
    plt.rcParams["font.family"] = "Garamond"
    plt.figure(figsize=(12, 8))
    plt.plot(grouped_by_size_mean_mae, '-o')
    plt.xlabel('Training Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'Mean performance of Linear Model by Training Size', fontsize=16, weight="bold")
    plt.suptitle(f"\nMean of MAE over {n} iterations of the model with random (thus different) train/test samples", y=0.9, fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.rcParams["font.family"] = "Garamond"
    plt.figure(figsize=(12, 8))
    plt.plot(grouped_by_size_mean_R2, '-o')
    plt.xlabel('Training Size')
    plt.ylabel('R2')
    plt.title(f'Mean performance of Linear Model by Training Size', fontsize=16, weight="bold")
    plt.suptitle(f"\nMean of MAE over {n} iterations of the model with random (thus different) train/test samples", y=0.9, fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

    return grouped_by_size_mean_mae , grouped_by_size_mean_R2



