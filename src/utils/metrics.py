import numpy as np
import scipy.stats as stats
from .engine import POS_LABEL, NEG_LABEL

def concordance_index(y_true, y_pred):
    """
    Calculate the Concordance Index (CI) for DTI prediction.
    CI is the proportion of pairs of interactions where the predicted values 
    have the same order as the observed values.
    """
    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]
    i = len(y_true)-1
    count = 0
    num_pairs = 0
    while i > 0:
        j = i - 1
        while j >= 0:
            if y_true[i] > y_true[j]:
                num_pairs += 1
                if y_pred[i] > y_pred[j]:
                    count += 1
                elif y_pred[i] == y_pred[j]:
                    count += 0.5
            j -= 1
        i -= 1
    return count / num_pairs if num_pairs > 0 else 1.0

def pearson_correlation(y_true, y_pred):
    """Calculate Pearson correlation coefficient."""
    return stats.pearsonr(y_true, y_pred)[0]

def mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))

def class_metrics(y_true, y_pred):
    """
    Calculate classification metrics (Accuracy, F1, Precision, Recall, etc.)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    TP = np.sum((y_pred == POS_LABEL) & (y_true == POS_LABEL))
    TN = np.sum((y_pred == NEG_LABEL) & (y_true == NEG_LABEL))
    FP = np.sum((y_pred == POS_LABEL) & (y_true == NEG_LABEL))
    FN = np.sum((y_pred == NEG_LABEL) & (y_true == POS_LABEL))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 1.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall      = sensitivity
    accuracy    = (TP + TN) / max(1, (TP + TN + FP + FN))
    f1          = (2*TP) / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 1.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }

def all_dti_metrics(y_true, y_prob):
    """
    Unified function for all DTI metrics (classification + regression-style).
    y_true: binary labels (for class metrics)
    y_prob: raw logits or probabilities (for AUPRC, AUROC, CI, etc.)
    """
    y_pred = (y_prob >= 0.5).astype(int)
    results = class_metrics(y_true, y_pred)
    
    # These metrics are traditionally for regression but often applied to scores in DTI research
    results["mse"] = float(mse(y_true, y_prob))
    results["rmse"] = float(rmse(y_true, y_prob))
    results["pearson"] = float(pearson_correlation(y_true, y_prob))
    results["ci"] = float(concordance_index(y_true, y_prob))
    
    return results
