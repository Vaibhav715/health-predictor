import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# =========================================================
# 🔷 DIABETES MODEL
# =========================================================

# Load dataset
diabetes = pd.read_csv("diabetes.csv")

# Columns with invalid 0 values
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Replace 0 with median
for col in cols:
    diabetes[col] = diabetes[col].replace(0, diabetes[col].median())

# Split features and target
X_d = diabetes.drop("Outcome", axis=1)
y_d = diabetes["Outcome"]

# Scaling
scaler = StandardScaler()
X_d_scaled = scaler.fit_transform(X_d)

# Train-test split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d_scaled, y_d, test_size=0.2, random_state=42
)

# XGBoost Model (Optimized)
diabetes_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train model
diabetes_model.fit(X_train_d, y_train_d)

# Accuracy (for debugging/logs)
y_pred_d = diabetes_model.predict(X_test_d)
print("✅ XGBoost Diabetes Accuracy:", accuracy_score(y_test_d, y_pred_d))


# =========================================================
# 🔷 HEART DISEASE MODEL
# =========================================================

# Load dataset
heart = pd.read_csv("heart.csv")

# Split features and target
X_h = heart.drop("target", axis=1)
y_h = heart["target"]

# Train-test split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

# Random Forest Model
heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train_h, y_train_h)

# Accuracy (for debugging/logs)
y_pred_h = heart_model.predict(X_test_h)
print("✅ Heart Disease Accuracy:", accuracy_score(y_test_h, y_pred_h))


# =========================================================
# 🔷 PREDICTION FUNCTIONS (CLEAN & FINAL)
# =========================================================

# -------- Diabetes Probability --------
def predict_diabetes_proba(data):
    """
    Returns probability of diabetes (0 to 1)
    """
    data_scaled = scaler.transform([data])
    return diabetes_model.predict_proba(data_scaled)[0][1]


# -------- Diabetes Prediction --------
def predict_diabetes(data):
    """
    Returns 0 (No Diabetes) or 1 (Diabetes)
    """
    data_scaled = scaler.transform([data])
    return diabetes_model.predict(data_scaled)[0]


# -------- Heart Probability --------
def predict_heart_proba(data):
    """
    Returns probability of heart disease (0 to 1)
    """
    return heart_model.predict_proba([data])[0][1]


# -------- Heart Prediction --------
def predict_heart(data):
    """
    Returns 0 (No Disease) or 1 (Disease)
    """
    return heart_model.predict([data])[0]
