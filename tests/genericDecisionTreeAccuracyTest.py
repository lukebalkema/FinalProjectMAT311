from src.utils.helper_functions import set_random_seed
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_models import train_decision_tree, scale_features
from src.models.evaluate_models import evaluate_model
from src.visualizations.visualize import plot_accuracy_vs_depth

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

    max_depths = range(1, 31)
    accuracy_scores = []

    # Train and evaluate Decision Tree for each depth
    for depth in max_depths:
        dt_model = train_decision_tree(X_train, y_train, max_depth=depth, random_state=random_state)
        metrics = evaluate_model(dt_model, X_val, y_val)
        accuracy_scores.append(metrics['accuracy'])
        print(f"max_depth={depth}, Validation Accuracy: {metrics['accuracy']:.4f}")

    # Plot the accuracy vs. depth
    plot_accuracy_vs_depth(max_depths, accuracy_scores, title='Validation Accuracy vs. Max Depth (Decision Tree)')

    # Determine the best depth
    best_depth = max_depths[accuracy_scores.index(max(accuracy_scores))]
    print(f"\nThe best depth is {best_depth} with a validation accuracy of {max(accuracy_scores):.4f}")
    return best_depth

if __name__ == "__main__":
    run_decision_tree_depth_experiment()
