import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from config.config import SEED
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as smf



def df_na_imputed(df):
    np.random.seed(SEED)

    # First filtering
    df.drop(labels = ["House_Age"], axis=1, inplace = True, errors="ignore")        # dropping house_Age (Because 99% of Na)
    df.dropna(subset=["Price"], inplace = True)     # dropping na's in Price. We won't impute our target variable


    #Imputation Bedrooms
    #Based on the distribution of Bedrooms, we impute na's with a random.normal of same mean and std
    df.loc[df["Bedrooms"].isna(), "Bedrooms"] = np.around(np.random.normal(df["Bedrooms"].mean(), df["Bedrooms"].std(), size = len(df.loc[df["Bedrooms"].isna(), "Bedrooms"] )))


    # The columns we impute with the mode
    cols_to_mode_impute = ["Stories", "Bathrooms", "Mainroad", "Hot_Water_Heating"]
    for col in cols_to_mode_impute:
        modeStories = df[col].mode()[0]
        df[col] = df[col].fillna(modeStories)


    # Impute Parking 
    # A Random Forest based on the Price ; because at this point Price
    model = RandomForestClassifier(random_state= SEED)
    X_train = pd.DataFrame(df.loc[df["Parking"].notna(), "Price"])
    y_train = pd.DataFrame(df.loc[df["Parking"].notna(), "Parking"])

    X_test = pd.DataFrame(df.loc[df["Parking"].isna(), "Price"])
    model.fit(X_train, y_train)
    df.loc[df["Parking"].isna(), "Parking"] = model.predict(X_test)    


    #Impute Area
    dataprov = df.copy()
    data_train = dataprov.dropna(subset = ["Area"])  # Data when Area is known
    model = smf.formula.ols(formula="Area ~ Price + Bedrooms + Bathrooms + Stories + Mainroad + Air_Conditioning + Parking + Prefarea ", df=data_train).fit()       #Linear model

    for col in ["Bedrooms", "Bathrooms", "Stories", "Mainroad", "Air_Conditioning", "Parking", "Prefarea"]:
        dataprov[col].fillna(dataprov[col].median(), inplace = True) # To remove the (one) missing df in x_miss
    # x_miss is the matrix of regressors when Area is missing
    x_miss = dataprov.loc[dataprov["Area"].isna() == True, ["Price", "Bedrooms", "Bathrooms", "Stories", "Mainroad", "Air_Conditioning", "Parking", "Prefarea"]]

    df.loc[df["Area"].isna() == True, "Area"] = model.predict(x_miss)   # Affect the prediction to missing df


    # Impute other categroical vars

    regressors = ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Mainroad',
                            'Hot_Water_Heating', 'Parking']
    targets = ["Guestroom", "Basement", "Air_Conditioning", "Prefarea", "Furnishing_Status"]

    for target in targets:
        model = RandomForestClassifier(random_state= SEED)
        X_train = pd.DataFrame(df.loc[df[target].notna(), regressors])
        y_train = pd.DataFrame(df.loc[df[target].notna(), target])
        X_test = pd.DataFrame(df.loc[df[target].isna(), regressors])
        model.fit(X_train, y_train)
        df.loc[df[target].isna(), target] = model.predict(X_test)  # Affect the predicition to missing data

        regressors.append(target)      # Adding this target to the regressors for next target


    return df    

