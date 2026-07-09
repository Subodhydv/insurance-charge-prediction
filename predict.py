import joblib
import pandas as pd

model = joblib.load("model/insurance_model.pkl")

new_customer = pd.DataFrame({
    "age":[30],
    "is_female":[1],
    "bmi":[27],
    "children":[2],
    "is_smoker":[0],
    "region_northwest":[0],
    "region_southeast":[1],
    "region_southwest":[0],
    "bmi_category_Healthy":[0],
    "bmi_category_Overweight":[1],
    "bmi_category_Obese":[0],
    "age_after_5":[35]
})

prediction = model.predict(new_customer)

print(f"Predicted Insurance Charges: ₹{prediction[0]:.2f}")