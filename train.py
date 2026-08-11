import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import load_dataset, clean_dataset, save_joblib_model
from src.eda import run_eda
from src.preprocessing import create_full_pipeline
from src.models import get_regression_models, get_classification_models
from src.evaluation import (
    evaluate_regression_model,
    evaluate_classification_model,
    plot_regression_leaderboard,
    plot_classification_leaderboard,
    plot_confusion_matrix
)


def run_master_training():
    """
    Main training workflow for both Regression and Classification models.
    """
    print("=" * 70)
    print("MASTER MACHINE LEARNING PIPELINE TRAINING")
    print("=" * 70)
    
    # STEP 1: LOAD & CLEAN DATA
    raw_df = load_dataset("data/insurance.csv")
    df = clean_dataset(raw_df)
    
    # Save cleaned data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/insurance_cleaned.csv", index=False)
    
    # STEP 2: EDA & VISUALIZATIONS
    print("\n--- Generating EDA Visualizations ---")
    run_eda(df, output_dir="images")
    
    # STEP 3: REGRESSION MODELS BENCHMARK
    print("\n" + "=" * 50)
    print("TASK 1: REGRESSION (PREDICTING INSURANCE CHARGES)")
    print("=" * 50)
    
    X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
    y_reg = df["charges"]
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    print(f"Regression Train set shape: {X_train_reg.shape}, Test set shape: {X_test_reg.shape}")
    
    reg_models = get_regression_models(random_state=42)
    reg_results = []
    fitted_reg_pipelines = {}
    
    for name, estimator in reg_models.items():
        # Scikit-Learn Pipeline combining ColumnTransformer (StandardScaler + OneHotEncoder) + Estimator
        pipeline = create_full_pipeline(estimator, scaler_type="standard")
        pipeline.fit(X_train_reg, y_train_reg)
        
        y_pred = pipeline.predict(X_test_reg)
        metrics = evaluate_regression_model(y_test_reg, y_pred, num_features=X_train_reg.shape[1])
        
        metrics["Model"] = name
        reg_results.append(metrics)
        fitted_reg_pipelines[name] = pipeline
        
        print(f"-> {name:<22} | R²: {metrics['R2']:.4f} | RMSE: ${metrics['RMSE']:,.2f} | MAE: ${metrics['MAE']:,.2f}")
        
    reg_df = pd.DataFrame(reg_results)[["Model", "R2", "Adj_R2", "RMSE", "MAE", "MSE"]]
    reg_df.sort_values(by="R2", ascending=False, inplace=True)
    
    os.makedirs("models", exist_ok=True)
    reg_df.to_csv("models/regression_leaderboard.csv", index=False)
    plot_regression_leaderboard(reg_df, output_path="images/regression_leaderboard.png")
    
    best_reg_name = reg_df.iloc[0]["Model"]
    best_reg_pipeline = fitted_reg_pipelines[best_reg_name]
    best_reg_score = reg_df.iloc[0]["R2"]
    print(f"\n🏆 CHAMPION REGRESSOR: {best_reg_name} (R² = {best_reg_score:.4f})")
    
    save_joblib_model(
        pipeline=best_reg_pipeline,
        metadata={"model_name": best_reg_name, "task": "Regression", "r2_score": float(best_reg_score)},
        model_path="models/best_regression_pipeline.joblib",
        metadata_path="models/best_regression_metadata.json"
    )
    
    # STEP 4: CLASSIFICATION MODELS BENCHMARK (STRATIFIED SPLIT)
    print("\n" + "=" * 50)
    print("TASK 2: CLASSIFICATION (PREDICTING HIGH INSURANCE RISK TIER)")
    print("=" * 50)
    
    y_cls = df["is_high_risk"]
    
    # STRATIFIED SPLIT for classification balanced targets
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    print(f"Stratified Classification Train set shape: {X_train_cls.shape}, Test set shape: {X_test_cls.shape}")
    
    cls_models = get_classification_models(random_state=42)
    cls_results = []
    fitted_cls_pipelines = {}
    
    for name, estimator in cls_models.items():
        pipeline = create_full_pipeline(estimator, scaler_type="standard")
        pipeline.fit(X_train_cls, y_train_cls)
        
        y_pred = pipeline.predict(X_test_cls)
        y_prob = pipeline.predict_proba(X_test_cls) if hasattr(pipeline, "predict_proba") else None
        
        metrics = evaluate_classification_model(y_test_cls, y_pred, y_prob)
        metrics["Model"] = name
        cls_results.append(metrics)
        fitted_cls_pipelines[name] = pipeline
        
        auc_str = f"{metrics['ROC_AUC']:.4f}" if isinstance(metrics['ROC_AUC'], float) else metrics['ROC_AUC']
        print(f"-> {name:<22} | F1: {metrics['F1_Score']:.4f} | Acc: {metrics['Accuracy']:.4f} | ROC-AUC: {auc_str}")
        
    cls_df = pd.DataFrame(cls_results)[["Model", "F1_Score", "Accuracy", "Precision", "Recall", "ROC_AUC"]]
    cls_df.sort_values(by="F1_Score", ascending=False, inplace=True)
    
    cls_df.to_csv("models/classification_leaderboard.csv", index=False)
    plot_classification_leaderboard(cls_df, output_path="images/classification_leaderboard.png")
    
    best_cls_name = cls_df.iloc[0]["Model"]
    best_cls_pipeline = fitted_cls_pipelines[best_cls_name]
    best_cls_score = cls_df.iloc[0]["F1_Score"]
    print(f"\n🏆 CHAMPION CLASSIFIER: {best_cls_name} (F1 = {best_cls_score:.4f})")
    
    # Plot Confusion Matrix for champion classifier
    best_cls_pred = best_cls_pipeline.predict(X_test_cls)
    plot_confusion_matrix(y_test_cls, best_cls_pred, model_name=best_cls_name, output_path="images/confusion_matrix.png")
    
    save_joblib_model(
        pipeline=best_cls_pipeline,
        metadata={"model_name": best_cls_name, "task": "Classification", "f1_score": float(best_cls_score)},
        model_path="models/best_classification_pipeline.joblib",
        metadata_path="models/best_classification_metadata.json"
    )
    
    # Also save fallback legacy path for compatibility
    save_joblib_model(
        pipeline=best_reg_pipeline,
        metadata={"model_name": best_reg_name},
        model_path="model/insurance_model.pkl",
        metadata_path="model/metadata.json"
    )
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE! ALL MODELS BENCHMARKED & Persisted.")
    print("=" * 70)


if __name__ == "__main__":
    run_master_training()
