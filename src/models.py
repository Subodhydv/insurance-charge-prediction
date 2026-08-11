from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    BaggingRegressor, BaggingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier


def get_regression_models(random_state=42):
    """
    Returns a dictionary of configured Regression models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state),
        "Bagging Regressor": BaggingRegressor(n_estimators=50, random_state=random_state),
        "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=random_state),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=random_state, verbosity=0),
        "SVM Regressor (SVR)": SVR(C=1000.0, epsilon=0.1, kernel="rbf")
    }
    return models


def get_classification_models(random_state=42):
    """
    Returns a dictionary of configured Classification models.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state),
        "Bagging Classifier": BaggingClassifier(n_estimators=50, random_state=random_state),
        "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state, eval_metric="logloss"),
        "Naive Bayes": GaussianNB(),
        "SVM Classifier (SVC)": SVC(probability=True, kernel="rbf", C=1.0, random_state=random_state)
    }
    return models
