# exp 8 accuracy metrics
import numpy as np
from sklearn.metrics import roc_curve, auc

# Set a random seed for reproducibility
np.random.seed(42)

# Generate sample data: 1000 true binary labels (30% positives)
n = 1000
y_true = np.random.binomial(1, 0.3, n)

# Generate predicted probabilities by adding some noise to the true labels,
# then making sure they stay between 0 and 1.
y_pred = np.clip(y_true + np.random.normal(0, 0.3, n), 0, 1)

# Compute ROC curve: fpr (false positive rates), tpr (true positive rates) and thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# Calculate the Area Under the ROC Curve (AUC)
roc_auc = auc(fpr, tpr)

# Define a function to calculate metrics at a given threshold
def compute_metrics(threshold):
    # Convert probabilities to binary predictions using the threshold
    predictions = (y_pred >= threshold).astype(int)
    
    # Count True Positives (TP), False Positives (FP), True Negatives (TN) and False Negatives (FN)
    tp = np.sum((predictions == 1) & (y_true == 1))
    fp = np.sum((predictions == 1) & (y_true == 0))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))
    
    # Calculate Accuracy, Precision, Recall and F1 Score
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

# Find the best threshold based on the highest F1 score
best_threshold = 0
best_f1 = 0
for th in thresholds:
    _, _, _, f1 = compute_metrics(th)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th

# Calculate the metrics using the best threshold found
accuracy, precision, recall, f1 = compute_metrics(best_threshold)

# Print the results
print("ROC AUC: {:.3f}".format(roc_auc))
print("Optimal Threshold: {:.3f}".format(best_threshold))
print("Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1 Score: {:.3f}".format(accuracy, precision, recall, f1))
