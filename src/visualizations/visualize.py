import matplotlib.pyplot as plt

def plot_accuracy_vs_k(k_values, accuracy_scores, title='Validation Accuracy vs. k'):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.title(title)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

def plot_recall_and_precision_vs_k(k_values, recall_scores, precision_scores, title='Recall and Precision vs. k'):
    """
    Plots both recall and precision scores against the number of neighbors (k) for a k-NN classifier.

    Parameters:
    - k_values: A sequence of k-values (e.g., range(1,31))
    - recall_scores: A sequence of recall values corresponding to each k
    - precision_scores: A sequence of precision values corresponding to each k
    - title: The title of the plot (optional)
    """

    plt.figure(figsize=(10, 6))

    # Plot recall scores
    plt.plot(k_values, recall_scores, marker='o', label='Recall', color='orange')
    # Plot precision scores
    plt.plot(k_values, precision_scores, marker='o', label='Precision', color='blue')

    plt.title(title)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Score')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()  # Show the legend to distinguish recall from precision
    plt.show()


def plot_roc_curves(models_metrics):
    plt.figure(figsize=(10, 8))
    for model_name, metrics in models_metrics.items():
        if 'fpr' in metrics and 'tpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'], label=f'{model_name} (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_accuracy_vs_depth(depth_values, accuracy_scores, title='Validation Accuracy vs. Max Depth'):
    plt.figure(figsize=(10, 6))
    plt.plot(depth_values, accuracy_scores, marker='o')
    plt.title(title)
    plt.xlabel('Max Depth')
    plt.ylabel('Validation Accuracy')
    plt.xticks(depth_values)
    plt.grid(True)
    plt.show()

def plot_recall_vs_depth(depth_values, recall_scores, title='Validation Recall vs. Max Depth'):
    plt.figure(figsize=(10, 6))
    plt.plot(depth_values, recall_scores, marker='o', color='orange')
    plt.title(title)
    plt.xlabel('Max Depth')
    plt.ylabel('Validation Recall')
    plt.xticks(depth_values)
    plt.grid(True)
    plt.show()

    import matplotlib.pyplot as plt

def plot_recall_and_precision_vs_depth(depth_values, recall_scores, precision_scores, title='Recall and Precision vs. Max Depth'):
    """
    Plots both recall and precision scores against the max depth values.

    Parameters:
    - depth_values: A sequence of depths (e.g., range(1,31))
    - recall_scores: A sequence of recall values corresponding to each depth
    - precision_scores: A sequence of precision values corresponding to each depth
    - title: The title of the plot (optional)
    """

    plt.figure(figsize=(10, 6))

    # Plot recall scores
    plt.plot(depth_values, recall_scores, marker='o', label='Recall', color='orange')
    # Plot precision scores
    plt.plot(depth_values, precision_scores, marker='o', label='Precision', color='blue')

    plt.title(title)
    plt.xlabel('Max Depth')
    plt.ylabel('Score')
    plt.xticks(depth_values)
    plt.grid(True)
    plt.legend()  # Show the legend to distinguish recall from precision
    plt.show()



def plot_metrics_histogram(metrics, title='Model Performance Metrics'):
    """
    Plots a bar chart showing Accuracy, Recall, Precision, and F1-Score from the given metrics dictionary.
    
    Parameters:
    - metrics (dict): A dictionary containing at least the keys 'accuracy', 'recall', 'precision', 'f1_score'.
                      The values should be floats (0 to 1).
    - title (str): The title of the plot.
    """
    # Extract the metric values in a consistent order
    metric_names = ['Accuracy', 'Recall', 'Precision', 'F1-Score']
    metric_keys = ['accuracy', 'recall', 'precision', 'f1_score']
    
    # Ensure all required keys are present
    for key in metric_keys:
        if key not in metrics:
            raise ValueError(f"Missing metric '{key}' in metrics dictionary.")
    
    values = [metrics[k] for k in metric_keys]
    
    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metric_names, values, color='skyblue')
    
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.ylim(0, 1.1)  # Slightly above 1 to give space for text labels
    plt.grid(axis='y')
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height:.2f}", ha='center', va='bottom')
    
    plt.show()
