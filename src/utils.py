import os
import json
import joblib
import pandas as pd
import numpy as np


def load_dataset(data_path="data/insurance.csv"):
    """
    Load raw insurance dataset and perform data validation.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at path: {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def clean_dataset(df):
    """
    Clean dataset:
    - Drop duplicate records
    - Handle missing values if any
    - Create classification target feature 'is_high_risk' / 'charge_tier'
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_shape = df_clean.shape
    df_clean.drop_duplicates(inplace=True)
    dedup_shape = df_clean.shape
    print(f"[Data Cleaning] Removed {initial_shape[0] - dedup_shape[0]} duplicate rows. New shape: {dedup_shape}")
    
    # Handle potential missing values
    if df_clean.isnull().sum().sum() > 0:
        print("[Data Cleaning] Imputing missing values...")
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    # Add classification target based on charge median/75th percentile split
    # Target 1 (Regression): 'charges' (continuous)
    # Target 2 (Classification): 'is_high_risk' (binary 0/1: 1 if charges >= median)
    threshold = df_clean["charges"].median()
    df_clean["is_high_risk"] = (df_clean["charges"] >= threshold).astype(int)
    
    return df_clean


def save_joblib_model(pipeline, metadata, model_path, metadata_path):
    """
    Save fitted Scikit-Learn pipeline and metadata using joblib.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[Model Persistence] Saved pipeline model to: {model_path}")
    print(f"[Model Persistence] Saved metadata to: {metadata_path}")


def load_joblib_model(model_path):
    """
    Load fitted Scikit-Learn pipeline using joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    return joblib.load(model_path)
