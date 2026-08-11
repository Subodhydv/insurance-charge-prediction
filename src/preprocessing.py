from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_preprocessor(scaler_type="standard"):
    """
    Construct a Scikit-Learn ColumnTransformer for numeric & categorical features.
    
    Parameters:
    - scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler, or None.
    
    Returns:
    - ColumnTransformer instance.
    """
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]
    
    # Numeric Pipeline: Imputation + Scaling
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaler_type == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler_type == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
        
    num_pipeline = Pipeline(steps=num_steps)
    
    # Categorical Pipeline: Imputation + One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    
    # Bundle into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="passthrough"
    )
    
    return preprocessor


def create_full_pipeline(estimator, scaler_type="standard"):
    """
    Wrap preprocessor and estimator into a Scikit-Learn Pipeline.
    """
    preprocessor = get_preprocessor(scaler_type=scaler_type)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])
    return pipeline
