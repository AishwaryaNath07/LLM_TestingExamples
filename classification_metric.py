import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics for binary or multiclass classification.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
    
    Returns:
        Dictionary containing accuracy, precision, recall, and f1 score
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def calculate_perplexity(probabilities):
    """
    Calculate perplexity for a language model.
    
    Args:
        probabilities: List of probabilities assigned to true words
    
    Returns:
        Perplexity score
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    log_probabilities = np.log(np.array(probabilities) + epsilon)
    perplexity = np.exp(-np.mean(log_probabilities))
    return perplexity

# Example 1: Classification Metrics (Binary Classification)
print("Binary Classification Example:")
true_labels_binary = [0, 1, 1, 0, 1, 0, 0, 1]
predicted_labels_binary = [0, 1, 0, 0, 1, 1, 0, 1]
binary_metrics = calculate_classification_metrics(true_labels_binary, predicted_labels_binary)

for metric, value in binary_metrics.items():
    print(f"{metric}: {value:.4f}")

# Example 2: Classification Metrics (Multiclass Classification)
print("\nMulticlass Classification Example:")
true_labels_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2]
predicted_labels_multi = [0, 1, 1, 0, 2, 2, 0, 1, 2]
multi_metrics = calculate_classification_metrics(true_labels_multi, predicted_labels_multi)

for metric, value in multi_metrics.items():
    print(f"{metric}: {value:.4f}")

# Example 3: Perplexity Calculation (Language Model Evaluation)
print("\nPerplexity Example:")
# Probabilities assigned by the model to the correct words in a sequence
word_probabilities = [0.8, 0.7, 0.9, 0.6, 0.95, 0.3, 0.85]
perplexity = calculate_perplexity(word_probabilities)
print(f"Perplexity: {perplexity:.4f}")

# Combined example with all metrics
print("\nCombined Example with All Metrics:")
print("Classification Metrics:")
for metric, value in multi_metrics.items():
    print(f"  {metric}: {value:.4f}")
print(f"Perplexity: {perplexity:.4f}")