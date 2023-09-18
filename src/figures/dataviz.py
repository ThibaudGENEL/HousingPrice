import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("df/house_prices.csv")


def dataviz_univariate(df):
    cols_to_analyze = df.columns

    # PARAM graphes
    sns.set_style("darkgrid")
    plt.rcParams["font.family"] = "Garamond"
    sns.set_palette("hls")

    # Parcourir chaque colonne
    for col in cols_to_analyze:
        # On s'assure que la colonne est numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            # Créer une figure avec deux sous-graphiques
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

# Utilisation de la fonction dataviz avec un DataFrame exemple
# dataviz(df)



dataviz_univariate(df)