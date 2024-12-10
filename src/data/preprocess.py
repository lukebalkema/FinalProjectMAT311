import pandas as pd
import numpy as np


def preprocess_data(df_train_data, df_test_data_original):
    # Clean the income column in test data (remove '.' and extra whitespace)
    df_test_data_original.iloc[:, -1] = df_test_data_original.iloc[:, -1].str.replace('.', '', regex=False).str.strip()
    
    # Concatenate dataframes
    df_total = pd.concat([df_train_data, df_test_data_original], ignore_index=True)
    
    # Define column names
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    df_total.columns = column_names
    
    # Replace '?' with NaN
    df_total.replace('?', np.nan, inplace=True)
    
    # Handle missing values
    for col in ['workclass', 'occupation', 'native-country']:
        df_total[col].fillna('Unknown', inplace=True)
    
    # Clean income column
    df_total['income'] = df_total['income'].str.replace('.', '', regex=False).str.strip()
    df_total['income'] = df_total['income'].map({'<=50K': 0, '>50K': 1})
    
    # Ensure no NaNs in income
    assert not df_total['income'].isnull().any(), "Error: 'income' column contains NaN values after mapping."
    
    return df_total
