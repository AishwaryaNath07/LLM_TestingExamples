from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example: Binary classification task
y_true = [1, 0, 1, 1, 0, 1, 0]   # ground truth labels
y_pred = [1, 0, 1, 0, 0, 1, 1]   # predicted labels

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print results
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
