import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# STEP 1 : LOAD DATASET

df = pd.read_csv("data/insurance.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nInfo")
print(df.info())

print("\nDescribe")
print(df.describe())


# STEP 2 : CHECK MISSING VALUES


print("\nMissing Values")
print(df.isnull())

print("\nMissing Values Count")
print(df.isnull().sum())


# STEP 3 : CHECK DUPLICATES


print("\nDuplicate Rows")
print(df.duplicated())

print("\nDuplicate Count")
print(df.duplicated().sum())

# Remove duplicates
df_cleaned = df.copy()

df_cleaned.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates")
print(df_cleaned.shape)

# STEP 4 : VALUE COUNTS


print("\nSex Count")
print(df_cleaned["sex"].value_counts())

print("\nSmoker Count")
print(df_cleaned["smoker"].value_counts())

print("\nRegion Count")
print(df_cleaned["region"].value_counts())


# STEP 5 : ENCODING


df_cleaned["sex"] = df_cleaned["sex"].map({
    "male":0,
    "female":1
})

df_cleaned["smoker"] = df_cleaned["smoker"].map({
    "no":0,
    "yes":1
})

df_cleaned.rename(columns={
    "sex":"is_female",
    "smoker":"is_smoker"
},inplace=True)

# One Hot Encoding
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=["region"],
    drop_first=True
)


# STEP 6 : FEATURE ENGINEERING


df_cleaned["bmi_category"] = pd.cut(
    df_cleaned["bmi"],
    bins=[0,18.5,25,30,100],
    labels=[
        "Underweight",
        "Healthy",
        "Overweight",
        "Obese"
    ]
)

# Encode BMI Category
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=["bmi_category"],
    drop_first=True
)

print(df_cleaned.head())

# STEP 7 : FILTERING


print("\nAge > 40")

print(
    df_cleaned[
        df_cleaned["age"]>40
    ]
)

print("\nCharges > 30000")

print(
    df_cleaned[
        df_cleaned["charges"]>30000
    ]
)

print("\nAge > 40 AND Smoker")

print(
    df_cleaned[
        (df_cleaned["age"]>40) &
        (df_cleaned["is_smoker"]==1)
    ]
)


# STEP 8 : SORTING

print("\nHighest Charges")

print(
    df_cleaned.sort_values(
        by="charges",
        ascending=False
    ).head(10)
)

# STEP 9 : GROUPBY

print("\nAverage Charges by Smoker")

print(
    df_cleaned.groupby(
        "is_smoker"
    )["charges"].mean()
)

print("\nAverage Charges by Gender")

print(
    df_cleaned.groupby(
        "is_female"
    )["charges"].mean()
)

# STEP 10 : APPLY()

df_cleaned["age_after_5"] = df_cleaned["age"].apply(
    lambda x:x+5
)

print(df_cleaned.head())

# STEP 11 : VISUALIZATION

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(df["bmi"],kde=True)
plt.title("BMI Distribution")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(
    x="age",
    y="charges",
    data=df
)
plt.title("Age vs Charges")
plt.show()

# Line Plot
plt.figure(figsize=(6,4))
sns.lineplot(
    x="age",
    y="charges",
    data=df
)
plt.title("Age vs Charges")
plt.show()

# Count Plot
plt.figure(figsize=(6,4))
sns.countplot(
    x="is_smoker",
    data=df_cleaned
)
plt.title("Smoker Count")
plt.show()

# Bar Plot
plt.figure(figsize=(6,4))
sns.barplot(
    x="is_smoker",
    y="charges",
    data=df_cleaned
)
plt.title("Average Charges by Smoker")
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(
    x="is_smoker",
    y="charges",
    data=df_cleaned
)
plt.title("Charges Distribution")
plt.show()

# STEP 12 : CORRELATION

plt.figure(figsize=(12,8))

corr = df_cleaned.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

# STEP 13 : MACHINE LEARNING

X = df_cleaned.drop("charges",axis=1)

y = df_cleaned["charges"]

# STEP 14 : TRAIN TEST SPLIT

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape")

print(X_train.shape)

print("\nTesting Shape")

print(X_test.shape)

# STEP 15 : LINEAR REGRESSION

model = LinearRegression()

model.fit(
    X_train,
    y_train
)
# STEP 16 : PREDICTION
prediction = model.predict(X_test)

print("\nPredicted Values")

print(prediction[:10])

print("\nActual Values")

print(y_test.values[:10])

#step 17: Evalution
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(y_test, prediction)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, prediction)

print("="*50)
print("Model Evaluation")
print("="*50)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print(X_train.columns)
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
predicted_charge = model.predict(new_customer)

print("Predicted Insurance Charge : ₹", predicted_charge[0])
import joblib

joblib.dump(model,"model/insurance_model.pkl")
model = joblib.load("model/insurance_model.pkl")

print("Model Saved Successfully!")
plt.savefig("images/age_vs_charges.png", dpi=300)