def formatting(df):
    
    """Perform data formatting operations on a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be formatted.

    Returns:
        pd.DataFrame: The DataFrame with the specified formatting applied.
        """

    df.columns = [col.lower().capitalize() for col in df.columns]
    df.rename(columns={'Furnishing status':'Furnishing_Status','Air conditioning':'Air_Conditioning','Hotwaterheating':'Hot_Water_Heating','Houseage':'House_Age'},inplace=True)
    
    df['Furnishing_Status']= df['Furnishing_Status'].apply(lambda x : x.lower().capitalize() if pd.notna(x) else x)
    df.replace({'yes':True,'no':False})