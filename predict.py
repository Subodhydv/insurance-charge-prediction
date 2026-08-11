import os
import pandas as pd
import joblib


def load_pipeline(model_path="models/best_regression_pipeline.joblib"):
    """
    Load saved Joblib model pipeline.
    """
    if not os.path.exists(model_path):
        # Fallback legacy path
        model_path = "model/insurance_model.pkl"
    print(f"Loading model pipeline from: {model_path}")
    return joblib.load(model_path)


def predict_insurance(age, sex, bmi, children, smoker, region):
    """
    Predict insurance charge & risk tier for a new customer record.
    """
    input_df = pd.DataFrame([{
        "age": float(age),
        "sex": str(sex).lower(),
        "bmi": float(bmi),
        "children": int(children),
        "smoker": str(smoker).lower(),
        "region": str(region).lower()
    }])
    
    # Load Regressor Pipeline
    reg_pipeline = load_pipeline("models/best_regression_pipeline.joblib")
    predicted_charge = reg_pipeline.predict(input_df)[0]
    
    # Load Classifier Pipeline
    cls_pipeline = load_pipeline("models/best_classification_pipeline.joblib")
    risk_pred = cls_pipeline.predict(input_df)[0]
    risk_label = "HIGH RISK" if risk_pred == 1 else "LOW RISK"
    
    print("\n" + "=" * 50)
    print("INSURANCE CHARGE & RISK PREDICTION RESULT")
    print("=" * 50)
    print(f"Input Profile     : Age: {age}, Sex: {sex}, BMI: {bmi}, Children: {children}, Smoker: {smoker}, Region: {region}")
    print(f"Predicted Charge  : ${predicted_charge:,.2f}")
    print(f"Risk Classification: {risk_label}")
    print("=" * 50 + "\n")
    
    return predicted_charge, risk_label


if __name__ == "__main__":
    # Test sample profile
    predict_insurance(
        age=30,
        sex="female",
        bmi=27.5,
        children=2,
        smoker="no",
        region="southeast"
    )