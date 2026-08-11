import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df, output_dir="images"):
    """
    Perform Exploratory Data Analysis (EDA) and save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    
    # 1. BMI Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["bmi"], kde=True, color="teal", bins=30)
    plt.title("Body Mass Index (BMI) Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("BMI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bmi_histogram.png"), dpi=300)
    plt.close()
    
    # 2. Age vs Charges Scatter Plot (Colored by Smoker)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="age", y="charges", hue="smoker", data=df, palette="Set1", alpha=0.8)
    plt.title("Age vs Charges (by Smoker Status)", fontsize=14, fontweight="bold")
    plt.xlabel("Age")
    plt.ylabel("Insurance Charges ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "age_vs_charges.png"), dpi=300)
    plt.close()
    
    # 3. BMI vs Charges (Colored by Smoker)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, palette="Dark2", alpha=0.8)
    plt.title("BMI vs Charges (by Smoker Status)", fontsize=14, fontweight="bold")
    plt.xlabel("BMI")
    plt.ylabel("Insurance Charges ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bmi_vs_charges.png"), dpi=300)
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(9, 6))
    df_encoded = df.copy()
    if 'sex' in df_encoded.columns:
        df_encoded['sex'] = df_encoded['sex'].map({'female': 1, 'male': 0})
    if 'smoker' in df_encoded.columns:
        df_encoded['smoker'] = df_encoded['smoker'].map({'yes': 1, 'no': 0})
    num_df = df_encoded.select_dtypes(include=[np.number])
    corr = num_df.corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 5. Charges Boxplot (Outlier Analysis)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="smoker", y="charges", hue="sex", data=df, palette="Pastel1")
    plt.title("Charges Outlier Boxplot by Smoker & Sex", fontsize=14, fontweight="bold")
    plt.xlabel("Smoker")
    plt.ylabel("Insurance Charges ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "charges_boxplot.png"), dpi=300)
    plt.close()
    
    print(f"[EDA] Generated & saved all visualizations to '{output_dir}/'")


if __name__ == "__main__":
    df = pd.read_csv("data/insurance.csv")
    run_eda(df)
