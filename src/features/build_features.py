import pandas as pd
from sklearn.model_selection import train_test_split

def build_features(df_total, random_state=123):
    # Identify categorical columns
    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_total, columns=categorical_columns)
    
    # Separate features and target
    X = df_encoded.drop('income', axis=1)
    y = df_encoded['income']
    
    # Split the data
    # 90% temp (train+validation), 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=random_state, stratify=y
    )
    
    # From temp: 70% train, 20% validation (0.2222 of temp ~ 20% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=random_state, stratify=y_temp
    )
    
    # Reset indices
    for df_ in [X_train, X_val, X_test, y_train, y_val, y_test]:
        df_.reset_index(drop=True, inplace=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    df_train = pd.concat([X_train, y_train], axis=1)
    df_validation = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    df_train.to_csv('data/processed/adult.newData.csv', index=False)
    df_validation.to_csv('data/processed/adult.newValidation.csv', index=False)
    df_test.to_csv('data/processed/adult.newTest.csv', index=False)
    
    return df_train, df_validation, df_test
