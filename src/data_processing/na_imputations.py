import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# from config.config import SEED
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as smf
import pickle


SEED = 13

def df_na_imputed(df, SEED = SEED):
    """
    Impute missing values in this specific DataFrame using various techniques.

    Parameters:
    data (DataFrame): Input DataFrame containing missing values. NB / Function restricted to our House DataFrame, won't work on another one

    Returns:
    DataFrame: DataFrame with missing values imputed.
    """
     
    np.random.seed(SEED)
    data = df.copy()

    #Imputation Bedrooms
    #Based on the distribution of Bedrooms, we impute na's with a random.normal of same mean and std
    data.loc[data["Bedrooms"].isna(), "Bedrooms"] = np.around(np.random.normal(data["Bedrooms"].mean(), data["Bedrooms"].std(), size = len(data.loc[data["Bedrooms"].isna(), "Bedrooms"] )))


    # The columns we impute with the mode
    cols_to_mode_impute = ["Stories", "Bathrooms", "Mainroad", "Hot_Water_Heating"]
    for col in cols_to_mode_impute:
        modeStories = data[col].mode()[0]
        data[col] = data[col].fillna(modeStories)


    #Impute Area
    dataprov = data.copy()
    data_train = dataprov.dropna(subset = ["Area"])  # Data when Area is known
    model = smf.formula.ols(formula="Area ~ Price + Bedrooms + Bathrooms + Stories + Mainroad + Air_Conditioning + Parking + Prefarea ", data = data_train).fit()       #Linear model

    for col in ["Bedrooms", "Bathrooms", "Stories", "Mainroad", "Air_Conditioning", "Parking", "Prefarea"]:
        dataprov[col].fillna(dataprov[col].median(), inplace = True) # To remove the (one) missing data in x_miss
    # x_miss is the matrix of regressors when Area is missing
    x_miss = dataprov.loc[dataprov["Area"].isna() == True, ["Price", "Bedrooms", "Bathrooms", "Stories", "Mainroad", "Air_Conditioning", "Parking", "Prefarea"]]

    data.loc[data["Area"].isna() == True, "Area"] = model.predict(x_miss)   # Affect the prediction to missing data


    # Impute Parking 
    # A Random Forest based on the Price
    data["Parking"] = data["Parking"].astype("category")
    model = RandomForestClassifier(random_state = SEED)
    X_train = pd.DataFrame(data.loc[data["Parking"].notna(), "Price"])
    y_train = pd.Series(data.loc[data["Parking"].notna(), "Parking"])
    X_test = pd.DataFrame(data.loc[data["Parking"].isna(), "Price"])
    model.fit(X_train, y_train)
    data.loc[data["Parking"].isna(), "Parking"] = model.predict(X_test)    


    # Impute other categroical vars

    regressors = ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Mainroad',
                            'Hot_Water_Heating', 'Parking']
    targets = ["Guestroom", "Basement", "Air_Conditioning", "Prefarea", "Furnishing_Status"]

    for target in targets:
        data[target] = data[target].astype("category")
        model = RandomForestClassifier(random_state= SEED)
        X_train = pd.DataFrame(data.loc[data[target].notna(), regressors])
        y_train = pd.Series(data.loc[data[target].notna(), target])
        X_test = pd.DataFrame(data.loc[data[target].isna(), regressors])
        model.fit(X_train, y_train)    
        model.classes_ = data[target].cat.categories    # Explicitly set the categories for the target column
        data.loc[data[target].isna(), target] = model.predict(X_test)  # Affect the predicition to missing data

        regressors.append(target)      # Adding this target to the regressors for next target


    return data   

