from src.utils.helper_functions import set_random_seed
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_models import train_decision_tree, scale_features, apply_smote
from src.models.evaluate_models import evaluate_model
from src.visualizations.visualize import plot_recall_vs_depth, plot_recall_and_precision_vs_depth

import pandas as pd

def run_decision_tree_depth_experiment():
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
    # You used method='enn' previously, adjust as needed
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state=random_state, method='enn')
    
    max_depths = range(1, 31)
    recall_scores = []
    precision_scores = []

    # Train and evaluate Decision Tree for each depth using SMOTE-resampled data
    for depth in max_depths:
        dt_model = train_decision_tree(X_train_res, y_train_res, max_depth=depth, random_state=random_state)
        metrics = evaluate_model(dt_model, X_val, y_val)
        
        recall_scores.append(metrics['recall'])
        precision_scores.append(metrics['precision'])
        
        print(f"max_depth={depth}, Validation Recall (with SMOTE): {metrics['recall']:.4f}, "
              f"Precision: {metrics['precision']:.4f}")

    # Plot both recall and precision vs. depth
    plot_recall_and_precision_vs_depth(max_depths, recall_scores, precision_scores, title='Validation Recall & Precision vs. Max Depth (Decision Tree with SMOTE)')

    # Determine the best depth:
    # 1. Find the maximum recall
    max_recall = max(recall_scores)
    # 2. Find all depths that achieve this max recall
    candidates = [i for i, r in enumerate(recall_scores) if r == max_recall]
    # 3. Among these candidates, find the one with the highest precision
    best_candidate = candidates[0]
    best_precision = precision_scores[best_candidate]
    for c in candidates[1:]:
        if precision_scores[c] > best_precision:
            best_precision = precision_scores[c]
            best_candidate = c

    best_depth = max_depths[best_candidate]
    print(f"\nThe best depth with SMOTE based on highest recall and then highest precision is {best_depth} "
          f"with a validation recall of {max_recall:.4f} and precision of {best_precision:.4f}")
    return best_depth

if __name__ == "__main__":
    run_decision_tree_depth_experiment()
