from src.utils.helper_functions import set_random_seed
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_models import train_knn, scale_features, apply_smote
from src.models.evaluate_models import evaluate_model
from src.visualizations.visualize import plot_recall_and_precision_vs_k

import pandas as pd

def run_knn_k_experiment():
    random_state = 123
    set_random_seed(random_state)

    # Load and preprocess data
    df_train_data, df_test_data_original = load_raw_data()
    df_total = preprocess_data(df_train_data, df_test_data_original)

    # Split into train/val/test sets
    X_train, X_val, X_test, y_train, y_val, y_test = build_features(df_total, random_state=random_state)

    # Scale features
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test, numerical_columns)

    # Apply SMOTE/SMOTE variant to address class imbalance in the training set
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state=random_state, method='enn')

    # Define a range of k values to test
    k_values = range(1, 31, 2)
    recall_scores = []
    precision_scores = []

    # Train and evaluate k-NN for each k using SMOTE-resampled data
    for k in k_values:
        knn_model = train_knn(X_train_res, y_train_res, n_neighbors=k)
        metrics = evaluate_model(knn_model, X_val, y_val)

        recall_scores.append(metrics['recall'])
        precision_scores.append(metrics['precision'])

        print(f"k={k}, Validation Recall (with SMOTE): {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")

    # Plot both recall and precision vs. k
    plot_recall_and_precision_vs_k(k_values, recall_scores, precision_scores, title='Validation Recall & Precision vs. k (k-NN with SMOTE)')

    # Determine the best k:
    # 1. Find the maximum recall
    max_recall = max(recall_scores)
    # 2. Find all k that achieve this max recall
    candidates = [i for i, r in enumerate(recall_scores) if r == max_recall]
    # 3. Among these candidates, find the one with the highest precision
    best_candidate = candidates[0]
    best_precision = precision_scores[best_candidate]
    for c in candidates[1:]:
        if precision_scores[c] > best_precision:
            best_precision = precision_scores[c]
            best_candidate = c

    best_k = k_values[best_candidate]
    print(f"\nThe best k with SMOTE based on highest recall and then highest precision is {best_k} "
          f"with a validation recall of {max_recall:.4f} and precision of {best_precision:.4f}")
    return best_k

if __name__ == "__main__":
    run_knn_k_experiment()
