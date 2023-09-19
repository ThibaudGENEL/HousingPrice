import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def dataviz_univariate(df):
    """
    Generate univariate visualizations for each column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to visualize.

    Returns:
    None

    Example:
    --------
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a sample DataFrame
    data = {
        'Age': [25, 30, 35, 40, 45],
        'Weight_kg': [70, 65, 80, 75, 72],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
    }
    df = pd.DataFrame(data)

    # Generate univariate visualizations
    dataviz_univariate(df)
    """

    cols_to_analyze = df.columns

    # PARAM graphes
    sns.set_style("darkgrid")
    plt.rcParams["font.family"] = "Garamond"
    sns.set_palette("hls")

    # Parcourir chaque colonne
    for col in cols_to_analyze:
        # Si colonne est numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            # Créer une figure avec deux sous-graphiques : Histo + Boxplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Histogramme
            sns.histplot(df[col], kde=True, color="black", ax=ax1)
            ax1.set_title(f'{col} distribution', weight='bold')
            if col == "Taille_du_receveur_en_cm":
                ax1.set_xlabel(f'{col} (cm)', weight='bold')
            elif "kg" in col:
                ax1.set_xlabel(f'{col} (kg)', weight='bold')
            else:
                ax1.set_xlabel(f'{col}', weight='bold')
            ax1.text(1, -0.1, f'n = {df[col].count()}',
                     horizontalalignment='right', verticalalignment='top', fontsize=8,
                     transform=ax1.transAxes)

            # Boîte à moustaches
            sns.boxplot(x=df[col], ax=ax2)
            ax2.set_title(f'{col} Boxplot', weight='bold')
            ax2.set_xlabel(f'{col}', weight='bold')
            ax2.axvline(df[col].mean(), color='firebrick')

            # Afficher les deux sous-graphiques côte à côte
            plt.show()
        # Si categorielle    
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):

            # Si le nombre de catégories est inférieur ou égal à 3, un cemembert
            if df[col].nunique() <= 3:
                plt.figure(figsize=(5, 3))
                plt.pie(df[col].value_counts(normalize = True), autopct="%1.1f%%", labels = df[col].value_counts().index) 
                plt.title(f'{col}', weight='bold')
                plt.text(1, -0.1, f'n = {df[col].count()}', 
                horizontalalignment='right', verticalalignment='top', fontsize=8, 
                transform=plt.gca().transAxes)
                plt.show()        
            # Si le nombre de catégories est inférieur ou égal à 10, un diagramme à barres
            elif df[col].nunique() <= 10:
                plt.figure(figsize=(8, 4))
                sns.countplot(x=df[col])
                plt.title(f'{col} Occurences', weight='bold', fontname='Times New Roman')
                plt.xlabel(f'{col}', weight='bold', fontname='Times New Roman')
                plt.xticks(rotation=30)
                plt.text(1, -0.15, f'n = {df[col].count()}', 
                horizontalalignment='right', verticalalignment='top', fontsize=8, 
                transform=plt.gca().transAxes)
                plt.show()


def dataviz_bivariate(data, var):
    """
    Generate bivariate visualizations for comparing a variable with all columns in a DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data to visualize.
    var (str): The variable to compare with all other columns.

    Returns:
    None

    Example:
    --------
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats

    # Create a sample DataFrame
    data = {
        'Age': [25, 30, 35, 40, 45],
        'Weight_kg': [70, 65, 80, 75, 72],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
    }
    df = pd.DataFrame(data)

    # Generate bivariate visualizations
    dataviz_bivariate(df, 'Age')
    """

    cols_to_analyze = data.columns

    for col in cols_to_analyze:  # Pour chaque colonne à comparer

        if pd.api.types.is_numeric_dtype(data[col]):  # SI colonne numérique ; nuage de points
            plt.figure(figsize=(6, 3.5))
            sns.regplot(data=data, x=var, y=col, scatter_kws={"color": "black", "alpha": 0.5, "s": 20},
                        line_kws={"color": "red"})
            plt.ylabel(col, weight='bold')
            plt.xlabel(var)
            plt.title(f"{[col]} vs {var}", weight='bold')
            plt.text(1, -0.12, f'n = {data[col].count()}',
                     horizontalalignment='right', verticalalignment='top', fontsize=8,
                     transform=plt.gca().transAxes)
            plt.show()

            correlation, p_value = stats.pearsonr(data[col].fillna(data[col].median()), data[var].fillna(data[var].median()))
            print("Pearson's correlation:", correlation)
            print("P-value:", p_value, "\n")

        else:  # Si colonne categ
            plt.figure(figsize=(8, 5))  # Set figure size
            sns.boxplot(x=col, y=var, data=data)  # Create a boxplot with the color palette
            plt.xlabel(col, weight='bold', fontsize=12)  # Set x-axis title
            plt.ylabel(var, weight='bold', fontsize=12)  # Set y-axis title
            plt.title(f"{var} By {[col]}", weight='bold', fontsize=14)  # Set plot title
            plt.tight_layout()  # Automatically adjust padding
            plt.text(1, -0.1, f'n = {data[col].count()}',
                     horizontalalignment='right', verticalalignment='top', fontsize=8,
                     transform=plt.gca().transAxes)
            plt.show()

            # test ANOVA
            group_data = [data[data[col] == mod][var].dropna() for mod in data[col].dropna().unique()]
            fval, pval = stats.f_oneway(*group_data)
            print(f"ANOVA Test, {var} on {[col]} (H0: Same mean {var} among modalities):")
            print(f"F-value = {fval:.6f}")
            print(f"P-value = {pval:.6f}")
    


