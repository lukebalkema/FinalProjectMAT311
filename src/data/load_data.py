import pandas as pd

def load_raw_data(train_path='data/raw/adult.data.csv', test_path='data/raw/adult.test.csv'):
    # Load training data
    df_train_data = pd.read_csv(train_path, header=None)
    
    # Load test data, skipping the first row
    df_test_data_original = pd.read_csv(test_path, skiprows=1, header=None)
    
    return df_train_data, df_test_data_original
