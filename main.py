from src.utils.helper_functions import set_random_seed
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features, save_processed_data
from src.models.train_models import scale_features, train_knn, train_nb, train_decision_tree, apply_smote
from src.models.evaluate_models import evaluate_model
from src.visualizations.visualize import plot_accuracy_vs_k, plot_roc_curves, plot_accuracy_vs_depth, plot_metrics_histogram
from tests.genericDecisionTree_R_SMOTE import run_decision_tree_depth_experiment  
from tests.KnnFindBestK import run_knn_k_experiment  # <-- Importing the KNN best k finder

import pandas as pd

def main():
    print("You are now running main.py")
    #------------------------------------------------------------------------------------------------------
    # Set random seed
    #------------------------------------------------------------------------------------------------------
    random_state = 123
    set_random_seed(random_state)
    
    #------------------------------------------------------------------------------------------------------
    # Load raw data
    #------------------------------------------------------------------------------------------------------
    print("I am now loading the raw data from the data/raw folder")
    df_train_data, df_test_data_original = load_raw_data()
    
    #------------------------------------------------------------------------------------------------------
    # Preprocess data
    #------------------------------------------------------------------------------------------------------
    df_total = preprocess_data(df_train_data, df_test_data_original)
    print("----------------------------------------------")
    print("We took the data and we split it into three parts, here is how we split it:")
    total_size = len(df_total)
    print(f"Total Size: {total_size}")
    print("----------------------------------------------")
    
    #------------------------------------------------------------------------------------------------------
    # Build features and split
    #------------------------------------------------------------------------------------------------------
    print("We are now building the features and splitting the data")
    X_train, X_val, X_test, y_train, y_val, y_test = build_features(df_total, random_state=random_state)
    print("----------------------------------------------")
    
    #------------------------------------------------------------------------------------------------------
    # Save processed data
    #------------------------------------------------------------------------------------------------------
    print("We are now saving the processed data")
    df_train, df_val, df_test_split = save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    train_size, val_size, test_size = len(df_train), len(df_val), len(df_test_split)
    print("----------------------------------------------")
    print(f"Train Size: {train_size} ({train_size/total_size*100:.2f}%)")
    print(f"Validation Size: {val_size} ({val_size/total_size*100:.2f}%)")
    print(f"Test Size: {test_size} ({test_size/total_size*100:.2f}%)")
    print("----------------------------------------------")
    
    #------------------------------------------------------------------------------------------------------
    # Scale Features
    #------------------------------------------------------------------------------------------------------
    print("We are now scaling the features")
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test, numerical_columns)
    print("----------------------------------------------")

    #------------------------------------------------------------------------------------------------------
    # Apply SMOTE to the training data
    #------------------------------------------------------------------------------------------------------
    print("Applying SMOTE to the training data to address class imbalance.")
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state=random_state, method='enn')
    print("----------------------------------------------")

    #------------------------------------------------------------------------------------------------------
    # KNN MODEL
    #------------------------------------------------------------------------------------------------------
    print("Determining the best k for the KNN model using SMOTE-based recall and precision experiment...")
    best_k = run_knn_k_experiment()
    print(f"The best k determined (based on recall and precision with SMOTE) is: {best_k}")
    print("----------------------------------------------")

    print("We are now training a KNN model (with SMOTE) and evaluating it on the validation set")
    knn_model = train_knn(X_train_res, y_train_res, n_neighbors=best_k)
    knn_metrics = evaluate_model(knn_model, X_val, y_val)
    print(f"\nKNN (k={best_k}, SMOTE applied) Validation Metrics:")
    print(knn_metrics['report'])
    print("----------------------------------------------")

    # Visualize KNN validation metrics as a histogram
    plot_metrics_histogram(knn_metrics, title=f'KNN Validation Metrics (k={best_k}, SMOTE)')


    #------------------------------------------------------------------------------------------------------
    # Gaussian Naive Bayes Model
    #------------------------------------------------------------------------------------------------------
    print("We are now training a Gaussian Naive Bayes model (with SMOTE) and evaluating it on the validation set")
    nb_model = train_nb(X_train_res, y_train_res)
    nb_metrics = evaluate_model(nb_model, X_val, y_val)
    print("\nGaussian Naive Bayes (SMOTE applied) Validation Metrics:")
    print(nb_metrics['report'])
    print("----------------------------------------------")
    
    # Visualize NB validation metrics as a histogram
    plot_metrics_histogram(nb_metrics, title='Gaussian Naive Bayes Validation Metrics (SMOTE)')


    #------------------------------------------------------------------------------------------------------
    # Decision Tree Model
    #------------------------------------------------------------------------------------------------------
    print("Determining the best depth for the decision tree using SMOTE-based recall and precision experiment...")
    best_depth = run_decision_tree_depth_experiment()
    print(f"The best depth determined (based on recall with SMOTE) is: {best_depth}")
    print("----------------------------------------------")

    print("We are now training a decision tree (with SMOTE) and evaluating it on the validation set")
    dt_model = train_decision_tree(X_train_res, y_train_res, best_depth, random_state=random_state)
    dt_metrics = evaluate_model(dt_model, X_val, y_val)
    print(f"\nDecision Tree (Depth={best_depth}, SMOTE applied) Validation Metrics:")
    print(dt_metrics['report'])
    print("----------------------------------------------")
    
    # Visualize validation metrics as a histogram
    plot_metrics_histogram(dt_metrics, title=f'Decision Tree Validation Metrics (Depth={best_depth}, SMOTE)')

    #------------------------------------------------------------------------------------------------------
    # Validation Test Metrics
    # This will run all the models on the validation set and figure out which model has the highest AUC.
    # We will use the AUC score to determine which model is the best. Then we will just use that one single model on the test set.
    #------------------------------------------------------------------------------------------------------
    # Combine metrics into a dictionary for ROC plotting
    models_metrics = {
        f'KNN (k={best_k})': knn_metrics,
        'Naive Bayes': nb_metrics,
        f'Decision Tree (Depth={best_depth})': dt_metrics
    }

    # Plot ROC curves for all models on the validation set
    plot_roc_curves(models_metrics)

    # Determine which model has the highest AUC
    best_model_name = None
    best_auc = 0.0
    best_model = None

    # We'll need access to the trained model objects and their metrics:
    # knn_model, nb_model, dt_model are the models; knn_metrics, nb_metrics, dt_metrics are their metrics
    model_objects = {
        f'KNN (k={best_k})': (knn_model, knn_metrics),
        'Naive Bayes': (nb_model, nb_metrics),
        f'Decision Tree (Depth={best_depth})': (dt_model, dt_metrics)
    }

    for model_name, (model_obj, metrics) in model_objects.items():
        if 'roc_auc' in metrics and metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model_name = model_name
            best_model = model_obj

    print("----------------------------------------------")
    print(f"The best model based on the validation AUC is: {best_model_name} with AUC={best_auc:.4f}")
    print("----------------------------------------------")


    #------------------------------------------------------------------------------------------------------
    # Final Test
    # This will take the chosen model with the best AUC value and run it on the test set.
    # Then it will return the ROC curve and also the histogram that shows precision, accuracy, recall, and F1-score.
    #------------------------------------------------------------------------------------------------------
    print(f"Evaluating the best model ({best_model_name}) on the Test Set")
    final_test_metrics = evaluate_model(best_model, X_test, y_test)
    print(f"\n{best_model_name} Test Set Metrics:")
    print(final_test_metrics['report'])
    print("----------------------------------------------")

    # Visualize test metrics as a histogram
    plot_metrics_histogram(final_test_metrics, title=f'{best_model_name} Test Metrics')
    
    # Plot the ROC curve for the final model on the test set
    # Even though we have a plot for validation, let's show the final test ROC:
    final_model_metrics = {best_model_name: final_test_metrics}
    plot_roc_curves(final_model_metrics)



if __name__ == "__main__":
    main()
