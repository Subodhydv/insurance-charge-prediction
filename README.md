# 🏥 Master Insurance Charge & Risk Prediction System

A production-grade, end-to-end Machine Learning portfolio project demonstrating **both Regression and Classification workflows** across 19 algorithm implementations, automated Scikit-Learn Pipelines, ColumnTransformer preprocessing, Stratified Splits, hyperparameter tuning, model evaluation suites, Joblib persistence, and an interactive Web Dashboard.

---

## 🌟 Key Highlights & Covered ML Concepts

This repository provides a comprehensive reference implementation covering the complete Machine Learning lifecycle:

### 🧠 Machine Learning Algorithms (19 Benchmark Variations)

#### 📉 Regression Task: Predicting Continuous Insurance Charges ($)
1. **Linear Regression**: Ordinary Least Squares baseline linear modeling.
2. **K-Nearest Neighbors Regressor (KNN)**: Non-parametric distance-weighted regression.
3. **Decision Tree Regressor**: Non-linear tree splitting based on MSE reduction.
4. **Random Forest Regressor**: Ensemble of randomized decision trees with variance reduction.
5. **Bagging Regressor**: Bootstrap Aggregation ensemble modeling.
6. **AdaBoost Regressor**: Adaptive sequential boosting focused on residual errors.
7. **Gradient Boosting Regressor**: Stage-wise additive model minimizing loss gradient.
8. **XGBoost Regressor**: Extreme Gradient Boosting with L1/L2 regularization and tree pruning.
9. **Support Vector Regressor (SVR)**: Epsilon-insensitive support vector machine with RBF kernel.

#### 🎯 Classification Task: Predicting High Insurance Risk Tier (0: Standard, 1: High Risk)
1. **Logistic Regression**: Sigmoid-based probabilistic classification.
2. **K-Nearest Neighbors Classifier (KNN)**: Distance-based majority voting classification.
3. **Decision Tree Classifier**: Tree-based classification using Gini Impurity / Information Gain.
4. **Random Forest Classifier**: Bagged tree ensemble classification.
5. **Bagging Classifier**: Bootstrap aggregated classifiers.
6. **AdaBoost Classifier**: Adaptive boosting re-weighting misclassified samples.
7. **Gradient Boosting Classifier**: Loss-gradient decision tree classifier.
8. **XGBoost Classifier**: State-of-the-art scalable gradient tree classifier.
9. **Naive Bayes (GaussianNB)**: Probabilistic classifier based on Bayes' Theorem with Gaussian distribution assumption.
10. **Support Vector Classifier (SVC)**: Maximum-margin hyper-plane classifier with RBF kernel and probability calibration.

---

### ⚙️ Preprocessing & Data Pipeline Engineering
- **EDA (Exploratory Data Analysis)**: Automated chart generation (Histograms, Scatterplots, Heatmaps, Boxplots).
- **Data Cleaning**: Outlier detection via IQR method, duplicate removal, missing value imputation.
- **One-Hot Encoding**: `sklearn.preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Feature Scaling**: `sklearn.preprocessing.StandardScaler` & `MinMaxScaler`.
- **Train/Test & Stratified Split**: `train_test_split` with `stratify=y` for class balance preservation.
- **ColumnTransformer**: `sklearn.compose.ColumnTransformer` cleanly separating continuous & categorical channels.
- **Scikit-Learn Pipelines**: `sklearn.pipeline.Pipeline` bundling data preprocessors + ML estimators to eliminate data leakage.
- **Model Evaluation**:
  - **Regression Metrics**: MAE, MSE, RMSE, R², Adjusted R².
  - **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
- **Joblib Model Serialization**: Fitted pipelines and metadata exported via `joblib.dump()` and loaded for inference via `joblib.load()`.
- **Interactive UI**: Real-time Streamlit dashboard (`app.py`) & interactive Jupyter Notebooks.

---

## 📐 Mathematical Foundations

### 1. Linear Regression
$$\hat{y} = \beta_0 + \sum_{i=1}^p \beta_i x_i$$

### 2. Logistic Regression (Sigmoid Function)
$$P(Y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where } z = \beta_0 + \mathbf{\beta}^T \mathbf{X}$$

### 3. Decision Tree Gini Impurity
$$Gini(D) = 1 - \sum_{i=1}^C p_i^2$$

### 4. Support Vector Machine (SVM Margin Optimization)
$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to } y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i$$

### 5. Regression Evaluation Metrics
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 📂 Project Architecture

```text
insurance-charge-prediction/
├── data/
│   ├── insurance.csv                      # Raw insurance dataset
│   └── insurance_cleaned.csv              # Processed dataset
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb      # EDA, missing values, outlier detection
│   ├── 02_preprocessing_and_pipelines.ipynb# ColumnTransformer & Pipeline engineering
│   ├── 03_regression_models.ipynb          # 9 Regressors benchmarked
│   ├── 04_classification_models.ipynb      # 10 Classifiers benchmarked (Stratified Split)
│   └── 05_model_comparison_and_saving.ipynb# Leaderboard comparison & Joblib loading
├── src/
│   ├── __init__.py
│   ├── eda.py                             # Automated EDA visualization module
│   ├── preprocessing.py                    # ColumnTransformer & Scikit-Learn Pipeline constructor
│   ├── models.py                          # 19 ML algorithms registry
│   ├── evaluation.py                      # Evaluation metrics suite & benchmark plotter
│   └── utils.py                           # Joblib IO & data cleaning utility functions
├── models/                                # Persisted Joblib pipelines & leaderboards
│   ├── best_regression_pipeline.joblib
│   ├── best_classification_pipeline.joblib
│   ├── regression_leaderboard.csv
│   └── classification_leaderboard.csv
├── images/                                # Generated visual chart artifacts
│   ├── correlation_heatmap.png
│   ├── age_vs_charges.png
│   ├── bmi_histogram.png
│   ├── charges_boxplot.png
│   ├── regression_leaderboard.png
│   ├── classification_leaderboard.png
│   └── confusion_matrix.png
├── app.py                                 # Streamlit Web Dashboard
├── train.py                               # CLI pipeline execution script
├── predict.py                             # Interactive CLI prediction tool
├── requirements.txt                       # Python package dependencies
└── README.md                              # Comprehensive documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train All Models & Generate Artifacts
Run the master training script to fit all 19 models, evaluate performance metrics, output leaderboards, and dump the champion Joblib pipelines:
```bash
python train.py
```

### 3. Predict Insurance Charges & Risk Tier (CLI)
Run interactive inference using saved Joblib model pipelines:
```bash
python predict.py
```

### 4. Launch Interactive Web Dashboard
Launch the Streamlit web application:
```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Subodh Kumar Yadav**  
B.Tech CSE, NIT Silchar  
Machine Learning & Data Science Portfolio