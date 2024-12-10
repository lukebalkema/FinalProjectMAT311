import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

def scale_features(X_train, X_val, X_test, numerical_columns):
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    return X_train, X_val, X_test

def train_knn(X_train, y_train, n_neighbors=21, metric='minkowski'):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_train, y_train)
    return knn

def train_nb(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def train_decision_tree(X_train, y_train, max_depth=10, random_state=123):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt

def apply_smote(X_train, y_train, random_state=123, method='tomek'):
    if method == 'tomek':
        smote_method = SMOTETomek(random_state=random_state)
    elif method == 'enn':
        smote_method = SMOTEENN(random_state=random_state)
    else:
        # Default: standard SMOTE if no or invalid method specified
        smote_method = SMOTE(random_state=random_state)
    
    X_train_res, y_train_res = smote_method.fit_resample(X_train, y_train)
    return X_train_res, y_train_res
