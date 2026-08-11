import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Insurance Charge & Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Insurance Charge & Risk Prediction System")
st.markdown("### Master Machine Learning Portfolio Project (Regression & Classification)")

# Sidebar Navigation
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Select Dashboard View", ["🔮 Interactive Predictor", "📊 EDA & Insights", "🏆 Model Leaderboard"])

@st.cache_resource
def load_models():
    reg_path = "models/best_regression_pipeline.joblib" if os.path.exists("models/best_regression_pipeline.joblib") else "model/insurance_model.pkl"
    cls_path = "models/best_classification_pipeline.joblib" if os.path.exists("models/best_classification_pipeline.joblib") else "model/insurance_model.pkl"
    reg_model = joblib.load(reg_path)
    cls_model = joblib.load(cls_path)
    return reg_model, cls_model

try:
    reg_pipeline, cls_pipeline = load_models()
except Exception as e:
    st.error(f"Error loading Joblib models: {e}. Please run 'python train.py' first!")
    st.stop()

if page == "🔮 Interactive Predictor":
    st.subheader("📋 Enter Customer Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", options=["female", "male"])
        
    with col2:
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.5, step=0.1)
        children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])
        
    with col3:
        smoker = st.selectbox("Smoker Status", options=["no", "yes"])
        region = st.selectbox("Region", options=["southwest", "southeast", "northwest", "northeast"])
        
    if st.button("🚀 Calculate Insurance Charge & Risk", use_container_width=True):
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])
        
        pred_charge = reg_pipeline.predict(input_data)[0]
        pred_risk = cls_pipeline.predict(input_data)[0]
        risk_label = "HIGH RISK TIER" if pred_risk == 1 else "LOW RISK TIER"
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Predicted Insurance Charge ($)", value=f"${pred_charge:,.2f}")
            
        with res_col2:
            st.metric(
                label="Risk Tier Classification",
                value=risk_label,
                delta="High Cost Potential" if pred_risk == 1 else "Standard Tier",
                delta_color="inverse" if pred_risk == 1 else "normal"
            )
            
elif page == "📊 EDA & Insights":
    st.subheader("Exploratory Data Analysis")
    
    if os.path.exists("data/insurance.csv"):
        df = pd.read_csv("data/insurance.csv")
        st.write("### Dataset Preview", df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("images/bmi_histogram.png"):
                st.image("images/bmi_histogram.png", caption="BMI Distribution")
            if os.path.exists("images/age_vs_charges.png"):
                st.image("images/age_vs_charges.png", caption="Age vs Charges")
                
        with col2:
            if os.path.exists("images/correlation_heatmap.png"):
                st.image("images/correlation_heatmap.png", caption="Correlation Heatmap")
            if os.path.exists("images/charges_boxplot.png"):
                st.image("images/charges_boxplot.png", caption="Outlier Boxplot")

elif page == "🏆 Model Leaderboard":
    st.subheader("Machine Learning Model Benchmarks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 📈 Regression Leaderboard")
        if os.path.exists("models/regression_leaderboard.csv"):
            reg_df = pd.read_csv("models/regression_leaderboard.csv")
            st.dataframe(reg_df, use_container_width=True)
        if os.path.exists("images/regression_leaderboard.png"):
            st.image("images/regression_leaderboard.png")
            
    with col2:
        st.write("#### 🎯 Classification Leaderboard")
        if os.path.exists("models/classification_leaderboard.csv"):
            cls_df = pd.read_csv("models/classification_leaderboard.csv")
            st.dataframe(cls_df, use_container_width=True)
        if os.path.exists("images/classification_leaderboard.png"):
            st.image("images/classification_leaderboard.png")
            
    if os.path.exists("images/confusion_matrix.png"):
        st.write("#### 🧩 Champion Classification Confusion Matrix")
        st.image("images/confusion_matrix.png", width=450)
