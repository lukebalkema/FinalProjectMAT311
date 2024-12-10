README
Project Overview
This project showcases a complete end-to-end machine learning pipeline applied to the Adult Income dataset. The primary objective is to predict whether an individual’s income exceeds $50K per year, using classification algorithms and techniques to handle class imbalance.


data/: For raw and processed data files.

notebooks/: Contains Jupyter notebooks for exploration and initial experimentation.

src/: Contains source code modules for data loading, preprocessing, feature engineering, training, evaluation, and visualization.

tests/: Holds test scripts that run experiments on model hyperparameters.

Reproducible Data Pipeline:
The pipeline begins by reading raw input data from the data/raw/ directory. It then:
- Cleans and preprocesses the dataset (handling missing values, encoding categorical variables).
- Splits the data into training, validation, and test sets.
- Saves processed outputs to data/processed/, ensuring future reproducibility and quick re-runs.

Feature Engineering & Scaling:
- Feature transformations like one-hot encoding and numeric feature scaling are applied. This ensures models receive appropriately formatted input and can better differentiate between classes.

Handling Class Imbalance with SMOTE:
The dataset is imbalanced (fewer individuals earning more than $50K). The project applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes. This improves the model’s recall for the minority class and helps the model generalize better.

Modeling Techniques:
Three main classification algorithms are showcased:

Decision Tree: Explores various tree depths to find the optimal complexity that balances recall and precision.
k-Nearest Neighbors (k-NN): Tests different values of k to select the one that maximizes recall and precision.
Gaussian Naive Bayes: Provides a probabilistic baseline model, trained directly on SMOTE-resampled data without hyperparameter tuning.
By comparing these models, the project demonstrates how to systematically search for the best hyperparameters for complex models and evaluate simpler models easily.

Model Evaluation & Selection:
Multiple metrics are considered:

Accuracy, Precision, Recall, F1-Score: Offer insights into model performance from various perspectives.
ROC Curve & AUC: Used to compare models and select the best one. The model that achieves the highest AUC on the validation set is chosen as the final model.
Visualizations such as histograms of metrics and combined ROC curves for all models allow for straightforward model comparisons.

Final Deployment & Testing:
After selecting the top-performing model based on validation results, the model is evaluated one last time on the held-out test set to confirm its generalization ability. The final results include:

Confusion Matrix and classification report for a comprehensive performance summary.
- ROC curve and AUC on the test set, providing a clear visualization of the model’s predictive capability.

How to Run the Project
Set Up the Environment:
- Create and activate a Python virtual environment (we used venv for the environment).
- Install dependencies using pip install -r requirements.txt.


Directory Structure: Ensure the directory structure is as follows:
project/
├── data/
│   ├── raw/
│   │   ├── adult.data.csv
│   │   └── adult.test.csv
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualizations/
│   └── utils/
├── tests/
├── main.py
├── requirements.txt

Run the Pipeline: Simply run:
`python main.py`

This will:
- Preprocess data.
- Apply SMOTE.
- Perform model selection and hyperparameter tuning for Decision Tree and k-NN.
- Train and evaluate Gaussian Naive Bayes.
- Compare models on the validation set and choose the best one.
- Evaluate the chosen model on the test set and produce final metrics and visualizations.

Results and Insights
The project demonstrates:
- How balancing the dataset with SMOTE can improve recall, ensuring the minority class is better represented.
- The importance of hyperparameter tuning (max_depth for Decision Trees, k for k-NN) in improving precision and overall   predictive performance.
- The value of multiple metrics and ROC/AUC analysis to select a model that balances sensitivity and specificity.
- By the end of the pipeline, you have a model that performs well not only on the validation set but also on the unseen test set, giving confidence that the solution generalizes.

Future Improvements
- Experimenting with other resampling techniques (e.g., SMOTEENN, SMOTETomek) to further refine class balance.
- Using ensemble methods (e.g., Random Forests, Gradient Boosted Trees) to potentially improve the trade-off between recall and precision.
- Implementing cross-validation for more robust hyperparameter tuning and performance estimation.


Conclusion
This project provides a clear blueprint for organizing a machine learning classification project, including data preprocessing, feature engineering, class imbalance handling, model selection, and final evaluation. By following the provided steps and code, you can adapt this pipeline to new datasets and classification tasks with minimal effort.
