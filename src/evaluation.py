import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_regression_model(y_true, y_pred, num_features):
    """
    Calculate Regression Metrics: MAE, MSE, RMSE, R2, Adjusted R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    p = num_features
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else r2
    
    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Adj_R2": round(adj_r2, 4)
    }


def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """
    Calculate Classification Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception:
            auc = None
            
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(auc, 4) if auc is not None else "N/A"
    }


def plot_regression_leaderboard(results_df, output_path="images/regression_leaderboard.png"):
    """
    Plot bar chart comparing Regression models by R2 score.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    sorted_df = results_df.sort_values(by="R2", ascending=True)
    bars = plt.barh(sorted_df["Model"], sorted_df["R2"], color="skyblue")
    
    plt.xlabel("R² Score", fontsize=12, fontweight="bold")
    plt.title("Model Comparison Leaderboard (Regression R² Score)", fontsize=14, fontweight="bold")
    plt.xlim(0, 1.0)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.4f}", ha="left", va="center", fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Regression Leaderboard chart to: {output_path}")


def plot_classification_leaderboard(results_df, output_path="images/classification_leaderboard.png"):
    """
    Plot bar chart comparing Classification models by F1-Score & Accuracy.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    sorted_df = results_df.sort_values(by="F1_Score", ascending=True)
    bars = plt.barh(sorted_df["Model"], sorted_df["F1_Score"], color="coral")
    
    plt.xlabel("F1 Score (Weighted)", fontsize=12, fontweight="bold")
    plt.title("Model Comparison Leaderboard (Classification F1 Score)", fontsize=14, fontweight="bold")
    plt.xlim(0, 1.0)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.4f}", ha="left", va="center", fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Classification Leaderboard chart to: {output_path}")


def plot_confusion_matrix(y_true, y_pred, model_name="Best Model", output_path="images/confusion_matrix.png"):
    """
    Plot confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
    plt.title(f"Confusion Matrix ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix heatmap to: {output_path}")
