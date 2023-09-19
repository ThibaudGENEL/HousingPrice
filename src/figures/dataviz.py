import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/house_prices.csv")


def dataviz_univariate(df):
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

# Utilisation de la fonction dataviz avec un DataFrame exemple
# dataviz(df)



dataviz_univariate(df)